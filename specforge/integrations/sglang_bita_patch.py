from __future__ import annotations

import glob
import json
import logging
import os
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from specforge.modeling.draft.onebit import patch_linears_to_onebit

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

ONEBIT_CONFIG_KEYS = (
    "use_onebit",
    "onebit_quant_func",
    "onebit_is_po2",
    "onebit_include_lm_head",
    "onebit_add_layernorm",
)

SPECFORGE_SGLANG_MODEL_PACKAGE = "specforge.sglang_models"


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


def _apply_onebit_config(self, source: Any) -> None:
    defaults = {
        "use_onebit": False,
        "onebit_quant_func": "STEBinary",
        "onebit_is_po2": False,
        "onebit_include_lm_head": False,
        "onebit_add_layernorm": True,
    }
    for key, default_value in defaults.items():
        value = (
            source.get(key, default_value)
            if isinstance(source, dict)
            else getattr(source, key, default_value)
        )
        setattr(self, key, value)
        setattr(self.config, key, value)

    self.use_onebit = bool(self.use_onebit)
    self.onebit_quant_func = str(self.onebit_quant_func)
    self.onebit_is_po2 = bool(self.onebit_is_po2)
    self.onebit_include_lm_head = bool(self.onebit_include_lm_head)
    self.onebit_add_layernorm = bool(self.onebit_add_layernorm)


def _resolve_checkpoint_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(
        repo_id=model_path,
        allow_patterns=["*.json", "*.safetensors", "*.bin"],
    )


def _ensure_external_model_package() -> None:
    current = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE", "")
    if current == SPECFORGE_SGLANG_MODEL_PACKAGE:
        return
    if current and current != SPECFORGE_SGLANG_MODEL_PACKAGE:
        logger.warning(
            "Overriding SGLANG_EXTERNAL_MODEL_PACKAGE=%s with %s so SpecForge's "
            "custom speculative draft models can be loaded.",
            current,
            SPECFORGE_SGLANG_MODEL_PACKAGE,
        )
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = SPECFORGE_SGLANG_MODEL_PACKAGE


def _load_model_config_dict(model_path: str) -> dict:
    model_dir = _resolve_checkpoint_dir(model_path)
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as file:
        return json.load(file)


def _is_peagle_draft_model(model_path: str | None) -> bool:
    if not model_path:
        return False
    try:
        config = _load_model_config_dict(model_path)
    except Exception:
        logger.exception("Failed to inspect draft model config from %s", model_path)
        return False

    architectures = config.get("architectures") or []
    if "LlamaForCausalLMPeagle" in architectures:
        return True
    return bool(config.get("specforge_parallel_drafting", False))


def _load_selected_checkpoint_tensors(
    model_path: str,
    tensor_names: list[str],
    map_location: str = "cpu",
) -> dict[str, torch.Tensor]:
    model_dir = _resolve_checkpoint_dir(model_path)
    selected = {}
    remaining = set(tensor_names)
    if not remaining:
        return selected

    index_json_paths = glob.glob(os.path.join(model_dir, "*.index.json"))
    if index_json_paths:
        if len(index_json_paths) != 1:
            raise FileNotFoundError(f"Multiple index.json files found in {model_dir}")
        with open(index_json_paths[0], "r") as file:
            index_json = json.load(file)

        files_to_keys: dict[str, list[str]] = {}
        for key in list(remaining):
            weight_file = index_json["weight_map"].get(key)
            if weight_file is None:
                continue
            files_to_keys.setdefault(weight_file, []).append(key)

        for file_name, keys in files_to_keys.items():
            file_path = os.path.join(model_dir, file_name)
            if file_name.endswith(".safetensors"):
                with safe_open(file_path, framework="pt") as handle:
                    for key in keys:
                        selected[key] = handle.get_tensor(key)
                        remaining.discard(key)
            else:
                state_dict = torch.load(file_path, map_location=map_location)
                for key in keys:
                    if key in state_dict:
                        selected[key] = state_dict[key]
                        remaining.discard(key)
        return selected

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        with safe_open(safetensors_path, framework="pt") as handle:
            for key in list(remaining):
                if key in handle.keys():
                    selected[key] = handle.get_tensor(key)
                    remaining.discard(key)
        return selected

    pytorch_model_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(pytorch_model_path):
        state_dict = torch.load(pytorch_model_path, map_location=map_location)
        for key in list(remaining):
            if key in state_dict:
                selected[key] = state_dict[key]
                remaining.discard(key)
        return selected

    raise FileNotFoundError(
        f"No index.json, model.safetensors, or pytorch_model.bin found in {model_dir}"
    )


def _copy_tensor_into_parameter(
    parameter: nn.Parameter,
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    if parameter.shape != tensor.shape:
        raise RuntimeError(
            f"Shape mismatch while loading OneBit tensor '{name}': "
            f"expected {tuple(parameter.shape)}, got {tuple(tensor.shape)}"
        )
    parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def _shard_qkv_output_tensor(module, tensor: torch.Tensor, shard_id: str) -> torch.Tensor:
    shard_size_mapping = {
        "q": module.q_proj_shard_size,
        "k": module.kv_proj_shard_size,
        "v": module.v_proj_shard_size,
    }
    shard_size = shard_size_mapping[shard_id]
    if shard_id == "q":
        shard_index = module.tp_rank
    else:
        shard_index = module.tp_rank // module.num_kv_head_replicas
    start = shard_index * shard_size
    return tensor.narrow(0, start, shard_size).contiguous()


def _shard_merged_output_tensor(
    module, tensor: torch.Tensor, output_size: int, shard_id: int
) -> torch.Tensor:
    shard_size = output_size // module.tp_size
    start = module.tp_rank * shard_size
    return tensor.narrow(0, start, shard_size).contiguous()


def _shard_row_input_tensor(module, tensor: torch.Tensor) -> torch.Tensor:
    shard_size = module.input_size_per_partition
    start = module.tp_rank * shard_size
    return tensor.narrow(0, start, shard_size).contiguous()


def _split_bias_or_none(bias: torch.Tensor | None, sizes: list[int]) -> list[torch.Tensor | None]:
    if bias is None:
        return [None for _ in sizes]
    return list(bias.split(sizes, dim=0))


def _onebit_sign(weight: torch.Tensor) -> torch.Tensor:
    signed = weight.sign()
    signed[signed == 0] = 1
    return signed


def _apply_onebit_column_branch(
    x: torch.Tensor,
    weight: torch.Tensor,
    in_scale: torch.Tensor,
    out_scale: torch.Tensor,
    layernorm: nn.Module,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    output = F.linear(x * in_scale, _onebit_sign(weight), bias=None)
    output = output * out_scale
    if bias is not None:
        output = output + bias
    return layernorm(output)


def _register_onebit_vector_parameter(
    module: nn.Module,
    name: str,
    size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if hasattr(module, name):
        return
    parameter = nn.Parameter(
        torch.ones(size, device=device, dtype=dtype), requires_grad=False
    )
    module.register_parameter(name, parameter)


def _register_onebit_layernorm(
    module: nn.Module,
    name: str,
    hidden_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    enabled: bool,
) -> None:
    if hasattr(module, name):
        return
    layernorm = nn.LayerNorm(hidden_size).to(device=device, dtype=dtype)
    if not enabled:
        layernorm = nn.Identity()
    if isinstance(layernorm, nn.LayerNorm):
        for parameter in layernorm.parameters():
            parameter.requires_grad = False
    module.add_module(name, layernorm)


def _patch_qkv_parallel_linear_for_onebit(module, *, add_layernorm: bool) -> None:
    if getattr(module, "_specforge_onebit_patched", False):
        return

    dtype = module.weight.dtype
    device = module.weight.device
    branch_specs = {
        "q": module.q_proj_shard_size,
        "k": module.kv_proj_shard_size,
        "v": module.v_proj_shard_size,
    }
    for branch_name, branch_size in branch_specs.items():
        _register_onebit_vector_parameter(
            module,
            f"{branch_name}_in_channel_scale",
            module.input_size,
            dtype=dtype,
            device=device,
        )
        _register_onebit_vector_parameter(
            module,
            f"{branch_name}_out_channel_scale",
            branch_size,
            dtype=dtype,
            device=device,
        )
        _register_onebit_layernorm(
            module,
            f"{branch_name}_layernorm",
            branch_size,
            dtype=dtype,
            device=device,
            enabled=add_layernorm,
        )

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        bias_chunks = _split_bias_or_none(
            bias,
            [self.q_proj_shard_size, self.kv_proj_shard_size, self.v_proj_shard_size],
        )
        q_weight, k_weight, v_weight = self.weight.split(
            [self.q_proj_shard_size, self.kv_proj_shard_size, self.v_proj_shard_size],
            dim=0,
        )
        q_output = _apply_onebit_column_branch(
            input_,
            q_weight,
            self.q_in_channel_scale,
            self.q_out_channel_scale,
            self.q_layernorm,
            bias_chunks[0],
        )
        k_output = _apply_onebit_column_branch(
            input_,
            k_weight,
            self.k_in_channel_scale,
            self.k_out_channel_scale,
            self.k_layernorm,
            bias_chunks[1],
        )
        v_output = _apply_onebit_column_branch(
            input_,
            v_weight,
            self.v_in_channel_scale,
            self.v_out_channel_scale,
            self.v_layernorm,
            bias_chunks[2],
        )
        output_parallel = torch.cat([q_output, k_output, v_output], dim=-1)

        if self.gather_output:
            from sglang.srt.layers.linear import tensor_model_parallel_all_gather

            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    module.forward = MethodType(forward, module)
    module._specforge_onebit_patched = True


def _patch_merged_column_parallel_linear_for_onebit(
    module, *, add_layernorm: bool
) -> None:
    if getattr(module, "_specforge_onebit_patched", False):
        return

    dtype = module.weight.dtype
    device = module.weight.device
    branch_names = ("gate", "up")
    for branch_name, branch_partition_size in zip(
        branch_names, module.output_partition_sizes, strict=True
    ):
        _register_onebit_vector_parameter(
            module,
            f"{branch_name}_in_channel_scale",
            module.input_size,
            dtype=dtype,
            device=device,
        )
        _register_onebit_vector_parameter(
            module,
            f"{branch_name}_out_channel_scale",
            branch_partition_size,
            dtype=dtype,
            device=device,
        )
        _register_onebit_layernorm(
            module,
            f"{branch_name}_layernorm",
            branch_partition_size,
            dtype=dtype,
            device=device,
            enabled=add_layernorm,
        )

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        bias_chunks = _split_bias_or_none(bias, self.output_partition_sizes)
        gate_weight, up_weight = self.weight.split(self.output_partition_sizes, dim=0)
        gate_output = _apply_onebit_column_branch(
            input_,
            gate_weight,
            self.gate_in_channel_scale,
            self.gate_out_channel_scale,
            self.gate_layernorm,
            bias_chunks[0],
        )
        up_output = _apply_onebit_column_branch(
            input_,
            up_weight,
            self.up_in_channel_scale,
            self.up_out_channel_scale,
            self.up_layernorm,
            bias_chunks[1],
        )
        output_parallel = torch.cat([gate_output, up_output], dim=-1)

        if self.gather_output:
            from sglang.srt.layers.linear import tensor_model_parallel_all_gather

            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    module.forward = MethodType(forward, module)
    module._specforge_onebit_patched = True


def _patch_row_parallel_linear_for_onebit(module, *, add_layernorm: bool) -> None:
    if getattr(module, "_specforge_onebit_patched", False):
        return

    dtype = module.weight.dtype
    device = module.weight.device
    _register_onebit_vector_parameter(
        module,
        "in_channel_scale",
        module.input_size_per_partition,
        dtype=dtype,
        device=device,
    )
    _register_onebit_vector_parameter(
        module,
        "out_channel_scale",
        module.output_size,
        dtype=dtype,
        device=device,
    )
    _register_onebit_layernorm(
        module,
        "layernorm",
        module.output_size,
        dtype=dtype,
        device=device,
        enabled=add_layernorm,
    )

    def forward(self, input_, skip_all_reduce: bool = False):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            from sglang.srt.layers.linear import split_tensor_along_last_dim

            input_parallel = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )[self.tp_rank].contiguous()

        output_parallel = F.linear(
            input_parallel * self.in_channel_scale,
            _onebit_sign(self.weight),
            bias=None,
        )

        if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
            from sglang.srt.layers.linear import (
                get_attention_tp_group,
                tensor_model_parallel_all_reduce,
            )

            if self.use_dp_attention_reduce:
                output = get_attention_tp_group().all_reduce(output_parallel)
            else:
                output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output = output * self.out_channel_scale
        if self.bias is not None and not self.skip_bias_add:
            output = output + self.bias
        output = self.layernorm(output)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    module.forward = MethodType(forward, module)
    module._specforge_onebit_patched = True


def _patch_sglang_draft_model_for_onebit(model) -> None:
    if getattr(model, "_specforge_onebit_runtime_patched", False):
        return

    if getattr(model.config, "onebit_include_lm_head", False):
        raise RuntimeError(
            "OneBit SGLang serving currently supports dense lm_head only. "
            "Please train or export the draft with onebit_include_lm_head=false."
        )

    patch_linears_to_onebit(
        model.model,
        do_train=False,
        quant_func_name=getattr(model.config, "onebit_quant_func", "STEBinary"),
        is_po2=getattr(model.config, "onebit_is_po2", False),
        include_lm_head=False,
        add_layernorm=getattr(model.config, "onebit_add_layernorm", True),
        exclude_names=("bita_",),
    )
    _patch_qkv_parallel_linear_for_onebit(
        model.model.midlayer.self_attn.qkv_proj,
        add_layernorm=getattr(model.config, "onebit_add_layernorm", True),
    )
    _patch_merged_column_parallel_linear_for_onebit(
        model.model.midlayer.mlp.gate_up_proj,
        add_layernorm=getattr(model.config, "onebit_add_layernorm", True),
    )
    _patch_row_parallel_linear_for_onebit(
        model.model.midlayer.self_attn.o_proj,
        add_layernorm=getattr(model.config, "onebit_add_layernorm", True),
    )
    _patch_row_parallel_linear_for_onebit(
        model.model.midlayer.mlp.down_proj,
        add_layernorm=getattr(model.config, "onebit_add_layernorm", True),
    )
    model._specforge_onebit_runtime_patched = True


def _load_onebit_state_into_sglang_model(model, model_path: str) -> None:
    tensor_names = [
        "fc.in_channel_scale",
        "fc.out_channel_scale",
        "fc.layernorm.weight",
        "fc.layernorm.bias",
        "midlayer.self_attn.q_proj.in_channel_scale",
        "midlayer.self_attn.q_proj.out_channel_scale",
        "midlayer.self_attn.q_proj.layernorm.weight",
        "midlayer.self_attn.q_proj.layernorm.bias",
        "midlayer.self_attn.k_proj.in_channel_scale",
        "midlayer.self_attn.k_proj.out_channel_scale",
        "midlayer.self_attn.k_proj.layernorm.weight",
        "midlayer.self_attn.k_proj.layernorm.bias",
        "midlayer.self_attn.v_proj.in_channel_scale",
        "midlayer.self_attn.v_proj.out_channel_scale",
        "midlayer.self_attn.v_proj.layernorm.weight",
        "midlayer.self_attn.v_proj.layernorm.bias",
        "midlayer.self_attn.o_proj.in_channel_scale",
        "midlayer.self_attn.o_proj.out_channel_scale",
        "midlayer.self_attn.o_proj.layernorm.weight",
        "midlayer.self_attn.o_proj.layernorm.bias",
        "midlayer.mlp.gate_proj.in_channel_scale",
        "midlayer.mlp.gate_proj.out_channel_scale",
        "midlayer.mlp.gate_proj.layernorm.weight",
        "midlayer.mlp.gate_proj.layernorm.bias",
        "midlayer.mlp.up_proj.in_channel_scale",
        "midlayer.mlp.up_proj.out_channel_scale",
        "midlayer.mlp.up_proj.layernorm.weight",
        "midlayer.mlp.up_proj.layernorm.bias",
        "midlayer.mlp.down_proj.in_channel_scale",
        "midlayer.mlp.down_proj.out_channel_scale",
        "midlayer.mlp.down_proj.layernorm.weight",
        "midlayer.mlp.down_proj.layernorm.bias",
    ]
    state = _load_selected_checkpoint_tensors(model_path, tensor_names)

    _copy_tensor_into_parameter(
        model.model.fc.in_channel_scale,
        state["fc.in_channel_scale"],
        name="fc.in_channel_scale",
    )
    _copy_tensor_into_parameter(
        model.model.fc.out_channel_scale,
        state["fc.out_channel_scale"],
        name="fc.out_channel_scale",
    )
    if hasattr(model.model.fc.layernorm, "weight"):
        _copy_tensor_into_parameter(
            model.model.fc.layernorm.weight,
            state["fc.layernorm.weight"],
            name="fc.layernorm.weight",
        )
        _copy_tensor_into_parameter(
            model.model.fc.layernorm.bias,
            state["fc.layernorm.bias"],
            name="fc.layernorm.bias",
        )

    qkv_module = model.model.midlayer.self_attn.qkv_proj
    for branch_name, shard_id in (("q", "q"), ("k", "k"), ("v", "v")):
        _copy_tensor_into_parameter(
            getattr(qkv_module, f"{branch_name}_in_channel_scale"),
            state[f"midlayer.self_attn.{branch_name}_proj.in_channel_scale"],
            name=f"midlayer.self_attn.{branch_name}_proj.in_channel_scale",
        )
        _copy_tensor_into_parameter(
            getattr(qkv_module, f"{branch_name}_out_channel_scale"),
            _shard_qkv_output_tensor(
                qkv_module,
                state[f"midlayer.self_attn.{branch_name}_proj.out_channel_scale"],
                shard_id,
            ),
            name=f"midlayer.self_attn.{branch_name}_proj.out_channel_scale",
        )
        layernorm = getattr(qkv_module, f"{branch_name}_layernorm")
        if hasattr(layernorm, "weight"):
            _copy_tensor_into_parameter(
                layernorm.weight,
                _shard_qkv_output_tensor(
                    qkv_module,
                    state[f"midlayer.self_attn.{branch_name}_proj.layernorm.weight"],
                    shard_id,
                ),
                name=f"midlayer.self_attn.{branch_name}_proj.layernorm.weight",
            )
            _copy_tensor_into_parameter(
                layernorm.bias,
                _shard_qkv_output_tensor(
                    qkv_module,
                    state[f"midlayer.self_attn.{branch_name}_proj.layernorm.bias"],
                    shard_id,
                ),
                name=f"midlayer.self_attn.{branch_name}_proj.layernorm.bias",
            )

    gate_up_module = model.model.midlayer.mlp.gate_up_proj
    merged_specs = (
        ("gate", 0, gate_up_module.output_sizes[0]),
        ("up", 1, gate_up_module.output_sizes[1]),
    )
    for branch_name, shard_id, output_size in merged_specs:
        _copy_tensor_into_parameter(
            getattr(gate_up_module, f"{branch_name}_in_channel_scale"),
            state[f"midlayer.mlp.{branch_name}_proj.in_channel_scale"],
            name=f"midlayer.mlp.{branch_name}_proj.in_channel_scale",
        )
        _copy_tensor_into_parameter(
            getattr(gate_up_module, f"{branch_name}_out_channel_scale"),
            _shard_merged_output_tensor(
                gate_up_module,
                state[f"midlayer.mlp.{branch_name}_proj.out_channel_scale"],
                output_size,
                shard_id,
            ),
            name=f"midlayer.mlp.{branch_name}_proj.out_channel_scale",
        )
        layernorm = getattr(gate_up_module, f"{branch_name}_layernorm")
        if hasattr(layernorm, "weight"):
            _copy_tensor_into_parameter(
                layernorm.weight,
                _shard_merged_output_tensor(
                    gate_up_module,
                    state[f"midlayer.mlp.{branch_name}_proj.layernorm.weight"],
                    output_size,
                    shard_id,
                ),
                name=f"midlayer.mlp.{branch_name}_proj.layernorm.weight",
            )
            _copy_tensor_into_parameter(
                layernorm.bias,
                _shard_merged_output_tensor(
                    gate_up_module,
                    state[f"midlayer.mlp.{branch_name}_proj.layernorm.bias"],
                    output_size,
                    shard_id,
                ),
                name=f"midlayer.mlp.{branch_name}_proj.layernorm.bias",
            )

    row_module_specs = (
        (
            model.model.midlayer.self_attn.o_proj,
            "midlayer.self_attn.o_proj",
        ),
        (
            model.model.midlayer.mlp.down_proj,
            "midlayer.mlp.down_proj",
        ),
    )
    for row_module, prefix in row_module_specs:
        _copy_tensor_into_parameter(
            row_module.in_channel_scale,
            _shard_row_input_tensor(row_module, state[f"{prefix}.in_channel_scale"]),
            name=f"{prefix}.in_channel_scale",
        )
        _copy_tensor_into_parameter(
            row_module.out_channel_scale,
            state[f"{prefix}.out_channel_scale"],
            name=f"{prefix}.out_channel_scale",
        )
        if hasattr(row_module.layernorm, "weight"):
            _copy_tensor_into_parameter(
                row_module.layernorm.weight,
                state[f"{prefix}.layernorm.weight"],
                name=f"{prefix}.layernorm.weight",
            )
            _copy_tensor_into_parameter(
                row_module.layernorm.bias,
                state[f"{prefix}.layernorm.bias"],
                name=f"{prefix}.layernorm.bias",
            )


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
        self._apply_onebit_config(config)
        self.bita_runtime_alpha = float(
            os.environ.get("SPECFORGE_BITA_RUNTIME_ALPHA", "0.25")
        )
        self.bita_runtime_enabled = False
        self.onebit_runtime_enabled = False

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
    LlamaForCausalLMEagle3.ONEBIT_CONFIG_KEYS = ONEBIT_CONFIG_KEYS
    LlamaForCausalLMEagle3.__init__ = patched_init
    LlamaForCausalLMEagle3.forward = patched_forward
    LlamaForCausalLMEagle3._apply_bita_config = _apply_bita_config
    LlamaForCausalLMEagle3._apply_onebit_config = _apply_onebit_config
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

        if not self.is_draft_worker:
            return

        draft_model_path = getattr(self.server_args, "speculative_draft_model_path", None)
        if getattr(self.model.config, "use_onebit", False):
            if not draft_model_path:
                raise RuntimeError(
                    "OneBit draft runtime was requested, but speculative_draft_model_path is missing."
                )
            _patch_sglang_draft_model_for_onebit(self.model)
            _load_onebit_state_into_sglang_model(self.model, draft_model_path)
            self.model.onebit_runtime_enabled = True
            logger.info(
                "Enabled OneBit runtime for draft model %s from %s",
                type(self.model).__name__,
                draft_model_path,
            )

        bita_path = getattr(self.server_args, "speculative_bita_model_path", None)
        if not bita_path:
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


def _patch_speculative_worker_selection() -> None:
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if getattr(SpeculativeAlgorithm, "_specforge_peagle_patch_applied", False):
        return

    original_create_worker = SpeculativeAlgorithm.create_worker

    def patched_create_worker(self, server_args):
        if self.is_eagle() and _is_peagle_draft_model(
            getattr(server_args, "speculative_draft_model_path", None)
        ):
            if not server_args.disable_overlap_schedule:
                raise ValueError(
                    "P-EAGLE runtime in SpecForge currently requires "
                    "--disable-overlap-schedule."
                )
            from specforge.integrations.sglang_peagle_worker import PEagleWorker

            return PEagleWorker
        return original_create_worker(self, server_args)

    SpeculativeAlgorithm.create_worker = patched_create_worker
    SpeculativeAlgorithm._specforge_peagle_patch_applied = True


def apply_sglang_bita_patch() -> bool:
    try:
        _ensure_external_model_package()
        _patch_server_args()
        _patch_llama_eagle3()
        _patch_model_runner()
        _patch_speculative_worker_selection()
    except ImportError:
        return False
    return True
