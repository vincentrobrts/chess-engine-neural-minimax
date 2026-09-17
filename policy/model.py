"""Small residual policy network for CPU-friendly move ordering."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from policy.encoding import INPUT_PLANES


@dataclass(frozen=True)
class PolicyModelConfig:
    input_planes: int = INPUT_PLANES
    channels: int = 64
    residual_blocks: int = 3
    policy_hidden_channels: int = 16
    vocab_size: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ChessPolicyNet(nn.Module):
    def __init__(self, config: PolicyModelConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("PolicyModelConfig.vocab_size must be positive.")

        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_planes, config.channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, config.channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *(ResidualBlock(config.channels) for _ in range(config.residual_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.channels, config.policy_hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8 * config.policy_hidden_channels, config.vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.policy_head(x)


def build_policy_model(vocab_size: int, **overrides: int) -> ChessPolicyNet:
    config = PolicyModelConfig(vocab_size=vocab_size, **overrides)
    return ChessPolicyNet(config)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

