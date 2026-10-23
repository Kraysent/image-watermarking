from collections.abc import Sequence

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, *, message_bits: int, channels: Sequence[int], residual_scale: float) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must contain at least one layer width")
        self.residual_scale = residual_scale
        layers: list[nn.Module] = []
        in_channels = 3 + message_bits
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        layers.extend(
            [
                nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        message_map = message.view(batch, -1, 1, 1).expand(-1, -1, height, width)
        return self.residual_scale * self.net(torch.cat([image, message_map], dim=1))


class Decoder(nn.Module):
    def __init__(self, *, message_bits: int, channels: Sequence[int]) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must contain at least one layer width")
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(in_channels, message_bits)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image).flatten(1))
