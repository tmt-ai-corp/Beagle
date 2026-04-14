from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def adaptive_po2(x: torch.Tensor) -> torch.Tensor:
    """Placeholder to preserve the LittleBit3 scale-quantization interface."""
    return x


class _STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        deriv = (x > -1) & (x < 1)
        return grad_output * deriv


class _SmoothSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
        ctx.alpha = alpha
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output * alpha * (1 - torch.tanh(alpha * x) ** 2)
        return grad_input, None


STEBinary = _STEBinary.apply
SmoothSign = _SmoothSign.apply

ONEBIT_QUANT_FUNCS = {
    "STEBinary": STEBinary,
    "SmoothSign": SmoothSign,
}


def get_onebit_quant_func(name: str):
    if name not in ONEBIT_QUANT_FUNCS:
        raise ValueError(
            f"Unknown OneBit quant function '{name}'. "
            f"Expected one of: {sorted(ONEBIT_QUANT_FUNCS)}"
        )
    return ONEBIT_QUANT_FUNCS[name]


class OneBitLinear(nn.Module):
    """
    In-place replacement for ``nn.Linear`` used by the Eagle3 draft model.

    This follows the LittleBit3 training-time conversion style: keep the dense
    weight parameter, learn per-channel scales, and apply a binary quantizer in
    the forward pass.
    """

    def __onebit_convert__(
        self,
        do_train: bool,
        quant_func_name: str = "STEBinary",
        is_po2: bool = False,
        add_layernorm: bool = True,
        layernorm_eps: float = 1e-5,
        **_: object,
    ) -> None:
        self.onebit_quant_func_name = quant_func_name
        self.quant_func = get_onebit_quant_func(quant_func_name)
        self.is_po2 = is_po2
        self.onebit_add_layernorm = add_layernorm
        self._binarized = False

        if add_layernorm:
            self.layernorm = nn.LayerNorm(self.out_features, eps=layernorm_eps)
        else:
            self.layernorm = nn.Identity()

        dtype = self.weight.data.dtype
        device = self.weight.data.device

        if do_train:
            in_channel_scale, out_channel_scale = _initialize_onebit_scales(
                self.weight.data
            )
            in_channel_scale = in_channel_scale.to(device=device, dtype=dtype)
            out_channel_scale = out_channel_scale.to(device=device, dtype=dtype)
        else:
            in_channel_scale = torch.ones(
                self.in_features, device=device, dtype=dtype
            )
            out_channel_scale = torch.ones(
                self.out_features, device=device, dtype=dtype
            )

        self.register_parameter(
            "in_channel_scale", nn.Parameter(in_channel_scale, requires_grad=True)
        )
        self.register_parameter(
            "out_channel_scale", nn.Parameter(out_channel_scale, requires_grad=True)
        )

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self._binarized:
            return x
        return self.quant_func(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])

        in_channel_scale = self.in_channel_scale
        out_channel_scale = self.out_channel_scale
        if self.is_po2:
            in_channel_scale = adaptive_po2(in_channel_scale)
            out_channel_scale = adaptive_po2(out_channel_scale)

        hidden_states = (
            (x * in_channel_scale)
            @ self.quantize(self.weight.to(dtype=x.dtype)).t()
        ) * out_channel_scale

        if self.bias is not None:
            hidden_states = hidden_states + self.bias

        hidden_states = self.layernorm(hidden_states)
        return hidden_states.reshape(*original_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        params = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias is not None,
            "quant_func": getattr(self, "onebit_quant_func_name", "STEBinary"),
        }
        return ", ".join(f"{key}={value}" for key, value in params.items())


def _initialize_onebit_scales(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = weight.dtype
    original_device = weight.device
    working_weight = weight.float()
    calc_device = (
        torch.device("cuda")
        if torch.cuda.is_available() and original_device.type != "cuda"
        else original_device
    )

    try:
        working_weight = working_weight.to(calc_device)
        U, S, Vh = torch.linalg.svd(torch.abs(working_weight), full_matrices=False)
        sqrt_S_diag = torch.sqrt(torch.diag(S))
        out_channel_scale = (U @ sqrt_S_diag[:, 0:1]).view(-1)
        in_channel_scale = (sqrt_S_diag[0:1, :] @ Vh).view(-1)
    except RuntimeError:
        abs_weight = torch.abs(working_weight)
        out_channel_scale = abs_weight.mean(dim=1).clamp_min(1e-6)
        in_channel_scale = abs_weight.mean(dim=0).clamp_min(1e-6)

    return (
        in_channel_scale.to(device=original_device, dtype=dtype),
        out_channel_scale.to(device=original_device, dtype=dtype),
    )


def patch_linears_to_onebit(
    model: nn.Module,
    *,
    do_train: bool,
    quant_func_name: str = "STEBinary",
    is_po2: bool = False,
    include_lm_head: bool = False,
    add_layernorm: bool = True,
    exclude_names: Iterable[str] = ("bita_",),
) -> nn.Module:
    for name, module in model.named_modules():
        if any(exclude_name in name for exclude_name in exclude_names):
            continue
        if not include_lm_head and name.endswith("lm_head"):
            continue
        if type(module) is nn.Linear:
            module.__class__ = OneBitLinear
            module.__onebit_convert__(
                do_train=do_train,
                quant_func_name=quant_func_name,
                is_po2=is_po2,
                add_layernorm=add_layernorm,
            )
    return model
