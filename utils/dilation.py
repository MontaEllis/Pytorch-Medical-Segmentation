import torch.nn as nn
from residual import ResidualBlock


class DilationBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dilation,
            dimensions,
            layers_per_block=2,
            num_residual_blocks=3,
            batch_norm=True,
            instance_norm=False,
            residual=True,
            padding_mode='constant',
            ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            residual_block = ResidualBlock(
                in_channels,
                out_channels,
                layers_per_block,
                dilation,
                dimensions,
                batch_norm=batch_norm,
                instance_norm=instance_norm,
                residual=residual,
                padding_mode=padding_mode,
            )
            residual_blocks.append(residual_block)
            in_channels = out_channels
        self.dilation_block = nn.Sequential(*residual_blocks)

    def forward(self, x):
        return self.dilation_block(x)