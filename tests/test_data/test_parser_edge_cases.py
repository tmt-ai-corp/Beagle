import unittest
from types import SimpleNamespace

import torch

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import ChatTemplate


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = None
        self.unk_token_id = 0
        self.chat_template = "dummy"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        if tools and not any(message["role"] == "user" for message in messages):
            raise ValueError(
                "Cannot put tools in the first user message when there's no first user message!"
            )

        rendered = []
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            if role == "system":
                rendered.append(f"<system>{content}</system>")
            elif role == "user":
                rendered.append(f"<user>{content}</user>")
            elif role == "assistant":
                rendered.append(f"<assistant>{content}<eot>")
            elif role == "tool":
                rendered.append(f"<tool>{content}</tool>")
        return "".join(rendered)

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        token_count = len(text)
        if max_length is not None:
            token_count = min(token_count, max_length)
        return SimpleNamespace(
            input_ids=torch.arange(token_count, dtype=torch.long).unsqueeze(0)
        )

    def encode(
        self,
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=None,
    ):
        token_count = len(text)
        if max_length is not None:
            token_count = min(token_count, max_length)
        return list(range(token_count))


class TestParserEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.chat_template = ChatTemplate(
            assistant_header="<assistant>",
            user_header="<user>",
            system_prompt="You are a helpful assistant.",
            end_of_turn_token="<eot>",
        )
        self.tools = [
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Seoul"},
                    },
                }
            ]
        ]

    def test_drops_leading_assistant_before_first_user(self):
        conversations = [
            [
                {"role": "assistant", "content": "orphan assistant turn"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        ]

        results = preprocess_conversations(
            tokenizer=self.tokenizer,
            conversations=conversations,
            chat_template=self.chat_template,
            max_length=1024,
            tools=self.tools,
        )

        self.assertEqual(len(results["input_ids"]), 1)
        self.assertTrue(torch.any(results["loss_mask"][0] == 1))

    def test_skips_conversation_without_any_user_turn(self):
        conversations = [
            [
                {"role": "assistant", "content": "orphan assistant turn"},
                {"role": "tool", "content": '{"temperature": 20}'},
            ]
        ]

        results = preprocess_conversations(
            tokenizer=self.tokenizer,
            conversations=conversations,
            chat_template=self.chat_template,
            max_length=1024,
            tools=self.tools,
        )

        self.assertEqual(len(results["input_ids"]), 0)
        self.assertEqual(len(results["loss_mask"]), 0)
        self.assertEqual(len(results["attention_mask"]), 0)


if __name__ == "__main__":
    unittest.main()
