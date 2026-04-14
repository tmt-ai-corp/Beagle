import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaForCausalLMEagle3,
    LlamaMLP,
    LlamaRMSNorm,
)
from specforge.modeling.draft.onebit import OneBitLinear

# from model_module import LlamaForCausalLMEagle3


class TestLlamaForCausalLMEagle3Loading(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()

        config_dict = {
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.28.1",
            "use_cache": True,
            "vocab_size": 128256,
            "draft_vocab_size": 32000,
        }

        self.config = LlamaConfig(**config_dict)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        model = LlamaForCausalLMEagle3(self.config)

        self.assertIsInstance(model.midlayer.self_attn, LlamaAttention)
        self.assertIsInstance(model.midlayer.mlp, LlamaMLP)
        self.assertIsInstance(model.midlayer.hidden_norm, LlamaRMSNorm)
        self.assertIsInstance(model.midlayer.input_layernorm, LlamaRMSNorm)
        self.assertIsInstance(model.midlayer.post_attention_layernorm, LlamaRMSNorm)
        self.assertEqual(model.midlayer.hidden_size, self.config.hidden_size)

    def test_save_pretrained(self):
        """Test the model's save_pretrained functionality."""
        model = LlamaForCausalLMEagle3(self.config)

        self.config.save_pretrained(self.temp_dir)

        model_path = os.path.join(self.temp_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)

        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))
        self.assertTrue(os.path.exists(model_path))

    @patch("transformers.modeling_utils.PreTrainedModel.from_pretrained")
    def test_from_pretrained_mock(self, mock_from_pretrained):
        """mock"""
        mock_model = LlamaForCausalLMEagle3(self.config)
        mock_from_pretrained.return_value = mock_model

        loaded_model = LlamaForCausalLMEagle3.from_pretrained(self.temp_dir)
        mock_from_pretrained.assert_called_once_with(self.temp_dir)
        self.assertIsInstance(loaded_model, LlamaForCausalLMEagle3)

    def test_model_forward_pass(self):
        """forward"""
        model = LlamaForCausalLMEagle3(self.config)
        model.eval()

        batch_size = 2
        seq_len = 10

        input_emb = torch.randn(batch_size, seq_len, self.config.hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size * 3)
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=input_emb,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        self.assertEqual(outputs.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_state_dict_compatibility(self):
        model1 = LlamaForCausalLMEagle3(self.config)
        model2 = LlamaForCausalLMEagle3(self.config)

        state_dict = model1.state_dict()

        model2.load_state_dict(state_dict)

        for name, param1 in model1.named_parameters():
            param2 = dict(model2.named_parameters())[name]
            self.assertTrue(torch.equal(param1, param2))

    def test_config_validation(self):
        invalid_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=127,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        with self.assertRaises(AttributeError):
            LlamaForCausalLMEagle3(invalid_config)

    def test_bita_helpers(self):
        config = LlamaConfig(
            vocab_size=1024,
            draft_vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=1,
            max_position_embeddings=128,
            pad_token_id=0,
            bita_mask_num=3,
            bita_mask_diff=True,
            bita_prompt_num=5,
            bita_prefix_hidden_size=64,
            bita_prefix_projection=True,
        )

        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        self.assertTrue(model.supports_bita_training)

        mask_embeds = model.get_bita_mask_embeddings(
            groups=2, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(mask_embeds.shape, (6, config.hidden_size))

        prompt_k, prompt_v = model.get_bita_prompt_key_values(
            batch_size=2, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(
            prompt_k.shape,
            (
                2,
                config.num_key_value_heads,
                config.bita_prompt_num,
                config.hidden_size // config.num_attention_heads,
            ),
        )
        self.assertEqual(prompt_v.shape, prompt_k.shape)

    def test_bita_backbone_forward(self):
        config = LlamaConfig(
            vocab_size=1024,
            draft_vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=1,
            max_position_embeddings=128,
            pad_token_id=0,
            bita_mask_num=2,
            bita_prompt_num=4,
        )

        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        model.eval()

        batch_size = 2
        seq_len = 6
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        input_embeds = model.embed_input_ids(input_ids)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        seq_attention_mask = model.prepare_decoder_attention_mask(
            attention_mask=torch.ones(batch_size, seq_len),
            hidden_states=hidden_states,
            batch_size=batch_size,
            seq_length=seq_len,
            past_key_values_length=0,
        )
        prompt_kv = model.get_bita_prompt_key_values(
            batch_size=batch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        prompt_len = prompt_kv[0].shape[-2]
        prompt_attention_mask = torch.zeros(batch_size, 1, seq_len, prompt_len)
        attention_mask = torch.cat([prompt_attention_mask, seq_attention_mask], dim=-1)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            outputs = model.backbone(
                input_embeds=input_embeds,
                hidden_states=hidden_states,
                cache_hidden=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                prompt_key_values=prompt_kv,
                use_cache=False,
            )

        self.assertEqual(outputs.shape, (batch_size, seq_len, config.hidden_size))

    def test_bita_freeze_non_bita_parameters(self):
        config = LlamaConfig(
            vocab_size=1024,
            draft_vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=1,
            max_position_embeddings=128,
            pad_token_id=0,
            bita_mask_num=2,
            bita_prompt_num=4,
        )

        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        model.freeze_non_bita_parameters()

        for name, param in model.named_parameters():
            if name.startswith("bita_"):
                self.assertTrue(param.requires_grad, name)
            else:
                self.assertFalse(param.requires_grad, name)

    def test_bita_save_and_load(self):
        config = LlamaConfig(
            vocab_size=1024,
            draft_vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=1,
            max_position_embeddings=128,
            pad_token_id=0,
            bita_mask_num=2,
            bita_prompt_num=4,
            bita_prefix_hidden_size=64,
        )

        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        for name, param in model.named_parameters():
            if name.startswith("bita_"):
                torch.nn.init.normal_(param)

        adapter_dir = os.path.join(self.temp_dir, "bita_adapter")
        model.save_bita_pretrained(adapter_dir)

        base_config = LlamaConfig(
            vocab_size=1024,
            draft_vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=1,
            max_position_embeddings=128,
            pad_token_id=0,
        )
        reloaded = LlamaForCausalLMEagle3(base_config, attention_backend="sdpa")
        reloaded.load_bita_pretrained(adapter_dir)

        self.assertEqual(reloaded.bita_mask_num, 2)
        self.assertEqual(reloaded.bita_prompt_num, 4)
        for name, param in model.named_parameters():
            if name.startswith("bita_"):
                self.assertTrue(torch.equal(param, dict(reloaded.named_parameters())[name]))

    def test_onebit_conversion_helpers(self):
        config = LlamaConfig(
            vocab_size=1024,
            draft_vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=1,
            max_position_embeddings=128,
            pad_token_id=0,
        )

        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        model.convert_linear_layers_to_onebit(do_train=True)

        self.assertTrue(model.use_onebit)
        self.assertIsInstance(model.fc, OneBitLinear)
        self.assertIsInstance(model.midlayer.self_attn.q_proj, OneBitLinear)
        self.assertIsInstance(model.midlayer.mlp.gate_proj, OneBitLinear)
        self.assertIsInstance(model.lm_head, torch.nn.Linear)
        self.assertTrue(hasattr(model.fc, "in_channel_scale"))
        self.assertTrue(hasattr(model.fc, "out_channel_scale"))

    def test_onebit_save_and_load(self):
        config = LlamaConfig(
            vocab_size=1024,
            draft_vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=1,
            max_position_embeddings=128,
            pad_token_id=0,
            use_onebit=True,
            onebit_quant_func="STEBinary",
            onebit_add_layernorm=True,
        )

        model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
        torch.nn.init.normal_(model.fc.weight)
        torch.nn.init.normal_(model.fc.in_channel_scale)
        torch.nn.init.normal_(model.fc.out_channel_scale)

        model_dir = os.path.join(self.temp_dir, "onebit_model")
        model.save_pretrained(model_dir)

        reloaded = LlamaForCausalLMEagle3.from_pretrained(
            model_dir, attention_backend="sdpa"
        )

        self.assertTrue(reloaded.use_onebit)
        self.assertIsInstance(reloaded.fc, OneBitLinear)
        self.assertTrue(torch.equal(model.fc.weight, reloaded.fc.weight))
        self.assertTrue(
            torch.equal(model.fc.in_channel_scale, reloaded.fc.in_channel_scale)
        )
        self.assertTrue(
            torch.equal(model.fc.out_channel_scale, reloaded.fc.out_channel_scale)
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestLlamaForCausalLMEagle3Loading))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
