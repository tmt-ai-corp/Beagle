from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BITA_CONFIG_KEYS = (
    "bita_mask_num",
    "bita_mask_diff",
    "bita_prompt_num",
    "bita_prefix_projection",
    "bita_prefix_hidden_size",
    "bita_prefix_dropout_prob",
    "bita_loss_weight",
    "bita_max_groups",
    "bita_window_strategy",
)


class PrefixEncoder(nn.Module):
    def __init__(
        self,
        *,
        prompt_num: int,
        hidden_size: int,
        output_dim: int,
        prefix_projection: bool,
        prefix_hidden_size: int,
    ) -> None:
        super().__init__()
        self.prefix_projection = prefix_projection

        if prefix_projection:
            self.embedding = nn.Embedding(prompt_num, hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(hidden_size, prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(prefix_hidden_size, output_dim),
            )
        else:
            self.embedding = nn.Embedding(prompt_num, output_dim)

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        prefix_embeds = self.embedding(prefix_tokens)
        if self.prefix_projection:
            return self.trans(prefix_embeds)
        return prefix_embeds


def _apply_bita_config(self, source: Any) -> None:
    defaults = {
        "bita_mask_num": 0,
        "bita_mask_diff": False,
        "bita_prompt_num": 0,
        "bita_prefix_projection": True,
        "bita_prefix_hidden_size": self.config.hidden_size,
        "bita_prefix_dropout_prob": 0.0,
        "bita_loss_weight": 1.0,
        "bita_max_groups": 1,
        "bita_window_strategy": "random",
    }
    for key, default_value in defaults.items():
        value = (
            source.get(key, default_value)
            if isinstance(source, dict)
            else getattr(source, key, default_value)
        )
        setattr(self, key, value)
        setattr(self.config, key, value)

    self.bita_mask_num = int(self.bita_mask_num or 0)
    self.bita_mask_diff = bool(self.bita_mask_diff)
    self.bita_prompt_num = int(self.bita_prompt_num or 0)
    self.bita_prefix_projection = bool(self.bita_prefix_projection)
    self.bita_prefix_hidden_size = int(self.bita_prefix_hidden_size)
    self.bita_prefix_dropout_prob = float(self.bita_prefix_dropout_prob)
    self.bita_loss_weight = float(self.bita_loss_weight)
    self.bita_max_groups = int(self.bita_max_groups or 1)
    self.bita_window_strategy = str(self.bita_window_strategy)


def _clear_bita_modules(self) -> None:
    for module_name in (
        "bita_mask_tokens",
        "bita_prefix_encoder",
        "bita_prefix_dropout",
    ):
        if hasattr(self, module_name):
            delattr(self, module_name)
    if "bita_prefix_tokens" in self._buffers:
        del self._buffers["bita_prefix_tokens"]


def _move_bita_modules_to_model_device(self) -> None:
    ref_param = next(
        (param for param in self.parameters() if param.is_floating_point()),
        None,
    )
    if ref_param is None:
        return

    device = ref_param.device
    dtype = ref_param.dtype
    for module_name in ("bita_mask_tokens", "bita_prefix_encoder", "bita_prefix_dropout"):
        module = getattr(self, module_name, None)
        if module is not None:
            module.to(device=device, dtype=dtype)

    if "bita_prefix_tokens" in self._buffers:
        self._buffers["bita_prefix_tokens"] = self._buffers["bita_prefix_tokens"].to(
            device=device
        )


def _init_bita_modules(self) -> None:
    self._clear_bita_modules()

    if self.bita_mask_num > 0:
        mask_vocab = self.bita_mask_num if self.bita_mask_diff else 1
        self.bita_mask_tokens = nn.Embedding(mask_vocab, self.config.hidden_size)

    if self.bita_prompt_num > 0:
        prefix_output_dim = (
            2
            * self.config.num_hidden_layers
            * self.config.num_key_value_heads
            * self.model.midlayer.self_attn.head_dim
        )
        self.bita_prefix_encoder = PrefixEncoder(
            prompt_num=self.bita_prompt_num,
            hidden_size=self.config.hidden_size,
            output_dim=prefix_output_dim,
            prefix_projection=self.bita_prefix_projection,
            prefix_hidden_size=self.bita_prefix_hidden_size,
        )
        self.register_buffer(
            "bita_prefix_tokens",
            torch.arange(self.bita_prompt_num, dtype=torch.long),
            persistent=False,
        )
        if self.bita_prefix_dropout_prob > 0:
            self.bita_prefix_dropout = nn.Dropout(self.bita_prefix_dropout_prob)

    self._move_bita_modules_to_model_device()


def _get_bita_config_dict(self) -> dict:
    return {key: getattr(self, key) for key in BITA_CONFIG_KEYS}


def _load_bita_pretrained(self, input_dir: str, map_location: str = "cpu") -> None:
    config_path = os.path.join(input_dir, "bita_config.json")
    state_path = os.path.join(input_dir, "bita_state.pt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"BiTA config not found: {config_path}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"BiTA state not found: {state_path}")

    with open(config_path, "r") as file:
        bita_config = json.load(file)

    self._apply_bita_config(bita_config)
    self._init_bita_modules()

    state_dict = torch.load(state_path, map_location=map_location)
    missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
    unexpected_keys = [key for key in unexpected_keys if key.startswith("bita_")]
    missing_keys = [key for key in missing_keys if key.startswith("bita_")]

    if unexpected_keys:
        raise RuntimeError(
            "Unexpected BiTA adapter keys while loading into SGLang draft model: "
            + ", ".join(unexpected_keys)
        )
    if missing_keys:
        raise RuntimeError(
            "Missing BiTA adapter keys while loading into SGLang draft model: "
            + ", ".join(missing_keys)
        )
    self.bita_runtime_enabled = True


def _get_bita_runtime_hidden_bias(
    self, reference_hidden: torch.Tensor
) -> torch.Tensor | None:
    if (
        not getattr(self, "bita_runtime_enabled", False)
        or self.bita_prompt_num <= 0
        or not hasattr(self, "bita_prefix_encoder")
    ):
        return None

    prompt_tokens = self.bita_prefix_tokens.unsqueeze(0).to(reference_hidden.device)
    prompt_kv = self.bita_prefix_encoder(prompt_tokens)

    if hasattr(self, "bita_prefix_dropout"):
        prompt_kv = self.bita_prefix_dropout(prompt_kv)

    head_dim = self.model.midlayer.self_attn.head_dim
    prompt_kv = prompt_kv.view(
        1,
        self.bita_prompt_num,
        2,
        self.config.num_hidden_layers,
        self.config.num_key_value_heads,
        head_dim,
    )
    prompt_k = prompt_kv[:, :, 0, 0].mean(dim=1)
    prompt_v = prompt_kv[:, :, 1, 0].mean(dim=1)
    prompt_bias = 0.5 * (prompt_k + prompt_v)

    repeat_factor = max(
        1, self.config.num_attention_heads // self.config.num_key_value_heads
    )
    prompt_bias = prompt_bias.repeat_interleave(repeat_factor, dim=1).reshape(1, -1)
    prompt_bias = prompt_bias[..., : reference_hidden.shape[-1]]
    prompt_bias = prompt_bias.to(
        device=reference_hidden.device,
        dtype=reference_hidden.dtype,
    )

    prompt_bias_rms = (
        prompt_bias.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(prompt_bias.dtype)
    )
    prompt_bias = prompt_bias / (prompt_bias_rms + 1e-6)

    ref_rms = (
        reference_hidden.float()
        .pow(2)
        .mean(dim=-1, keepdim=True)
        .sqrt()
        .mean()
        .to(reference_hidden.dtype)
    )
    return prompt_bias * ref_rms * self.bita_runtime_alpha


def _patch_server_args() -> None:
    from sglang.srt.server_args import ServerArgs

    if getattr(ServerArgs, "_specforge_bita_patch_applied", False):
        return

    ServerArgs.speculative_bita_model_path = None

    original_add_cli_args = ServerArgs.add_cli_args
    original_from_cli_args = ServerArgs.from_cli_args

    def add_cli_args(parser):
        original_add_cli_args(parser)
        if any(
            "--speculative-bita-model-path" in action.option_strings
            for action in parser._actions
        ):
            return
        parser.add_argument(
            "--speculative-bita-model-path",
            type=str,
            default=None,
            help="Path to a standalone BiTA adapter checkpoint that should be attached to the speculative draft model.",
        )

    @classmethod
    def from_cli_args(cls, args):
        server_args = original_from_cli_args(args)
        setattr(
            server_args,
            "speculative_bita_model_path",
            getattr(args, "speculative_bita_model_path", None),
        )
        return server_args

    ServerArgs.add_cli_args = staticmethod(add_cli_args)
    ServerArgs.from_cli_args = from_cli_args
    ServerArgs._specforge_bita_patch_applied = True


def _patch_llama_eagle3() -> None:
    from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3

    if getattr(LlamaForCausalLMEagle3, "_specforge_bita_patch_applied", False):
        return

    original_init = LlamaForCausalLMEagle3.__init__

    def patched_init(self, config, quant_config=None, prefix: str = "") -> None:
        original_init(self, config, quant_config=quant_config, prefix=prefix)
        self._apply_bita_config(config)
        self._init_bita_modules()
        self.bita_runtime_alpha = float(
            os.environ.get("SPECFORGE_BITA_RUNTIME_ALPHA", "0.25")
        )
        self.bita_runtime_enabled = False

    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ):
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                input_embeds = self.model.embed_tokens(input_ids)
            prompt_bias = self.get_bita_runtime_hidden_bias(input_embeds)
            if prompt_bias is not None:
                input_embeds = input_embeds + prompt_bias

        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            return self.pooler(hidden_states, forward_batch)

        return hidden_states

    LlamaForCausalLMEagle3.BITA_CONFIG_KEYS = BITA_CONFIG_KEYS
    LlamaForCausalLMEagle3.__init__ = patched_init
    LlamaForCausalLMEagle3.forward = patched_forward
    LlamaForCausalLMEagle3._apply_bita_config = _apply_bita_config
    LlamaForCausalLMEagle3._clear_bita_modules = _clear_bita_modules
    LlamaForCausalLMEagle3._move_bita_modules_to_model_device = (
        _move_bita_modules_to_model_device
    )
    LlamaForCausalLMEagle3._init_bita_modules = _init_bita_modules
    LlamaForCausalLMEagle3.get_bita_config_dict = _get_bita_config_dict
    LlamaForCausalLMEagle3.load_bita_pretrained = _load_bita_pretrained
    LlamaForCausalLMEagle3.get_bita_runtime_hidden_bias = _get_bita_runtime_hidden_bias
    LlamaForCausalLMEagle3._specforge_bita_patch_applied = True


def _patch_model_runner() -> None:
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_specforge_bita_patch_applied", False):
        return

    original_load_model = ModelRunner.load_model

    def patched_load_model(self, *args, **kwargs):
        original_load_model(self, *args, **kwargs)

        bita_path = getattr(self.server_args, "speculative_bita_model_path", None)
        if not self.is_draft_worker or not bita_path:
            return

        if not hasattr(self.model, "load_bita_pretrained"):
            raise RuntimeError(
                f"Draft model class {type(self.model).__name__} does not support standalone BiTA loading. "
                "Currently the runtime patch supports Llama EAGLE3 draft models."
            )

        self.model.load_bita_pretrained(bita_path)
        logger.info(
            "Loaded standalone BiTA adapter from %s into draft model %s",
            bita_path,
            type(self.model).__name__,
        )

    ModelRunner.load_model = patched_load_model
    ModelRunner._specforge_bita_patch_applied = True


def apply_sglang_bita_patch() -> bool:
    try:
        _patch_server_args()
        _patch_llama_eagle3()
        _patch_model_runner()
    except ImportError:
        return False
    return True
