import torch
import torch.nn as nn


class PrefixEncoder(nn.Module):
    """
    Lightweight prefix encoder used by the BiTA-style prompt pathway.

    The encoder outputs a packed tensor that can be reshaped into per-layer
    key/value tensors for attention.
    """

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
