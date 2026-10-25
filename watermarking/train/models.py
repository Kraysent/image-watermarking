from collections.abc import Sequence

import torch
import torch.nn as nn


def _conv_norm_relu(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        message_bits: int,
        channels: Sequence[int],
        residual_scale: float,
        message_layer: int = 0,
    ) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must contain at least one layer width")
        if not 0 <= message_layer < len(channels):
            raise ValueError(f"message_layer must be in [0, {len(channels)}), got {message_layer}")
        self.residual_scale = residual_scale

        in_channels = 3
        stem: list[nn.Module] = []
        for out_channels in channels[:message_layer]:
            stem.extend(_conv_norm_relu(in_channels, out_channels))
            in_channels = out_channels
        self.stem = nn.Sequential(*stem) if stem else nn.Identity()

        in_channels = in_channels + message_bits
        body: list[nn.Module] = []
        for out_channels in channels[message_layer:]:
            body.extend(_conv_norm_relu(in_channels, out_channels))
            in_channels = out_channels
        body.extend(
            [
                nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.body = nn.Sequential(*body)

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        features = self.stem(image)
        batch, _, height, width = features.shape
        message_map = (2.0 * message - 1.0).view(batch, -1, 1, 1).expand(-1, -1, height, width)
        return self.residual_scale * self.body(torch.cat([features, message_map], dim=1))


class Decoder(nn.Module):
    def __init__(self, *, message_bits: int, channels: Sequence[int]) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must contain at least one layer width")
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(_conv_norm_relu(in_channels, out_channels))
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(in_channels, message_bits)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image).flatten(1))
