import torch
import torch.nn as nn

from convolution import ConvolutionalBlock


BATCH_DIM = 0
CHANNELS_DIM = 1


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_layers,
            dilation,
            dimensions,
            batch_norm=True,
            instance_norm=False,
            residual=True,
            residual_type='pad',
            padding_mode='constant',
            ):
        assert residual_type in ('pad', 'project')
        super().__init__()
        self.residual = residual
        self.change_dimension = in_channels != out_channels
        self.residual_type = residual_type
        self.dimensions = dimensions
        if self.change_dimension:
            if residual_type == 'project':
                conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
                self.change_dim_layer = conv_class(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    dilation=dilation,
                    bias=False,  # as in NiftyNet and PyTorch's ResNet model
                )

        conv_blocks = nn.ModuleList()
        for _ in range(num_layers):
            conv_block = ConvolutionalBlock(
                in_channels,
                out_channels,
                dilation,
                dimensions,
                batch_norm=batch_norm,
                instance_norm=instance_norm,
                padding_mode=padding_mode,
            )
            conv_blocks.append(conv_block)
            in_channels = out_channels
        self.residual_block = nn.Sequential(*conv_blocks)

    def forward(self, x):
        """
        From the original ResNet paper, page 4:
        "When the dimensions increase, we consider two options:
        (A) The shortcut still performs identity mapping,
        with extra zero entries padded for increasing dimensions.
        This option introduces no extra parameter
        (B) The projection shortcut in Eqn.(2) is used to
        match dimensions (done by 1x1 convolutions).
        For both options, when the shortcuts go across feature maps of
        two sizes, they are performed with a stride of 2."
        """
        out = self.residual_block(x)
        if self.residual:
            if self.change_dimension:
                if self.residual_type == 'project':
                    x = self.change_dim_layer(x)
                elif self.residual_type == 'pad':
                    batch_size = x.shape[BATCH_DIM]
                    x_channels = x.shape[CHANNELS_DIM]
                    out_channels = out.shape[CHANNELS_DIM]
                    spatial_dims = x.shape[2:]
                    diff_channels = out_channels - x_channels
                    zeros_half = x.new_zeros(
                        batch_size, diff_channels // 2, *spatial_dims)
                    x = torch.cat((zeros_half, x, zeros_half),
                                  dim=CHANNELS_DIM)
            out = x + out
        return out