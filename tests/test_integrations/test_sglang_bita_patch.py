import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from specforge.integrations.sglang_bita_patch import (
    _get_bita_runtime_hidden_bias,
    apply_sglang_bita_patch,
)


class TestSGLangBiTAPatch(unittest.TestCase):

    def test_server_args_accepts_bita_cli_flag(self):
        apply_sglang_bita_patch()

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model-path",
                str(
                    Path(__file__).resolve().parents[2]
                    / "models"
                    / "Eagle3-Llama-3.1-8B-Intruct"
                ),
                "--speculative-bita-model-path",
                "/tmp/bita_adapter",
            ]
        )
        server_args = ServerArgs.from_cli_args(args)

        self.assertEqual(
            getattr(server_args, "speculative_bita_model_path", None),
            "/tmp/bita_adapter",
        )

    def test_llama_eagle3_model_is_patched_for_bita_loading(self):
        apply_sglang_bita_patch()

        from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
        from sglang.srt.model_executor.model_runner import ModelRunner

        self.assertTrue(hasattr(LlamaForCausalLMEagle3, "load_bita_pretrained"))
        self.assertTrue(hasattr(LlamaForCausalLMEagle3, "BITA_CONFIG_KEYS"))
        self.assertTrue(hasattr(LlamaForCausalLMEagle3, "get_bita_runtime_hidden_bias"))
        self.assertIn("bita_prompt_num", LlamaForCausalLMEagle3.BITA_CONFIG_KEYS)
        self.assertTrue(getattr(ModelRunner, "_specforge_bita_patch_applied", False))

    def test_runtime_hidden_bias_is_available_when_bita_prompt_exists(self):
        class DummyPrefixEncoder(nn.Module):
            def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
                batch_size, prompt_num = prefix_tokens.shape
                output_dim = 2 * 1 * 4 * 16
                return torch.randn(batch_size, prompt_num, output_dim)

        dummy = SimpleNamespace(
            bita_runtime_enabled=True,
            bita_prompt_num=4,
            bita_runtime_alpha=0.25,
            bita_prefix_encoder=DummyPrefixEncoder(),
            bita_prefix_tokens=torch.arange(4, dtype=torch.long),
            config=SimpleNamespace(
                num_hidden_layers=1,
                num_key_value_heads=4,
                num_attention_heads=8,
            ),
            model=SimpleNamespace(
                midlayer=SimpleNamespace(
                    self_attn=SimpleNamespace(
                        head_dim=16,
                    )
                )
            ),
        )
        ref_hidden = torch.randn(6, 128)

        runtime_bias = _get_bita_runtime_hidden_bias(dummy, ref_hidden)

        self.assertIsNotNone(runtime_bias)
        self.assertEqual(runtime_bias.shape, (1, 128))

    def test_runtime_hidden_bias_is_none_without_bita_prompt(self):
        dummy = SimpleNamespace(
            bita_runtime_enabled=False,
            bita_prompt_num=0,
            bita_runtime_alpha=0.25,
        )
        ref_hidden = torch.randn(6, 128)

        runtime_bias = _get_bita_runtime_hidden_bias(dummy, ref_hidden)

        self.assertIsNone(runtime_bias)


if __name__ == "__main__":
    unittest.main(verbosity=2)
