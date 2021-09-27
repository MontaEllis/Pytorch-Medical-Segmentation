import torch
import torch.nn as nn
import os

from ...utils.dilation import DilationBlock
from ...utils.convolution import ConvolutionalBlock

# copy from https://github.com/fepegar/highresnet, Thank you to [fepegar](https://github.com/fepegar) for your generosity!

__all__ = ['HighResNet', 'HighRes2DNet', 'HighRes3DNet']


class HighResNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dimensions=None,
            initial_out_channels_power=4,
            layers_per_residual_block=2,
            residual_blocks_per_dilation=3,
            dilations=3,
            batch_norm=True,
            instance_norm=False,
            residual=True,
            padding_mode='constant',
            add_dropout_layer=False,
            ):
        assert dimensions in (2, 3)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_residual_block = layers_per_residual_block
        self.residual_blocks_per_dilation = residual_blocks_per_dilation
        self.dilations = dilations

        # List of blocks
        blocks = nn.ModuleList()

        # Add first conv layer
        initial_out_channels = 2 ** initial_out_channels_power
        first_conv_block = ConvolutionalBlock(
            in_channels=self.in_channels,
            out_channels=initial_out_channels,
            dilation=1,
            dimensions=dimensions,
            batch_norm=batch_norm,
            instance_norm=instance_norm,
            preactivation=False,
            padding_mode=padding_mode,
        )
        blocks.append(first_conv_block)

        # Add dilation blocks
        in_channels = out_channels = initial_out_channels
        dilation_block = None  # to avoid pylint errors
        for dilation_idx in range(dilations):
            if dilation_idx >= 1:
                in_channels = dilation_block.out_channels
            dilation = 2 ** dilation_idx
            dilation_block = DilationBlock(
                in_channels,
                out_channels,
                dilation,
                dimensions,
                layers_per_block=layers_per_residual_block,
                num_residual_blocks=residual_blocks_per_dilation,
                batch_norm=batch_norm,
                instance_norm=instance_norm,
                residual=residual,
                padding_mode=padding_mode,
            )
            blocks.append(dilation_block)
            out_channels *= 2
        out_channels = out_channels // 2

        # Add dropout layer as in NiftyNet
        if add_dropout_layer:
            in_channels = out_channels
            out_channels = 80
            dropout_conv_block = ConvolutionalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=1,
                dimensions=dimensions,
                batch_norm=batch_norm,
                instance_norm=instance_norm,
                preactivation=False,
                kernel_size=1,
            )
            blocks.append(dropout_conv_block)
            blocks.append(nn.Dropout3d())

        # Add classifier
        classifier = ConvolutionalBlock(
            in_channels=out_channels,
            out_channels=self.out_channels,
            dilation=1,
            dimensions=dimensions,
            batch_norm=batch_norm,
            instance_norm=instance_norm,
            preactivation=False,
            kernel_size=1,
            activation=False,
            padding_mode=padding_mode,
        )

        blocks.append(classifier)
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

    @property
    def num_parameters(self):
        # pylint: disable=not-callable
        return sum(torch.prod(torch.tensor(p.shape)) for p in self.parameters())

    @property
    def receptive_field(self):
        """
        B: number of convolutional layers per residual block
        N: number of residual blocks per dilation factor
        D: number of different dilation factors
        """
        B = self.layers_per_residual_block
        D = self.dilations
        N = self.residual_blocks_per_dilation
        d = torch.arange(D)
        input_output_diff = (3 - 1) + torch.sum(B * N * 2 ** (d + 1))
        receptive_field = input_output_diff + 1
        return receptive_field

    def get_receptive_field_world(self, spacing=1):
        return self.receptive_field * spacing




class HighRes2DNet(HighResNet):
    def __init__(self, *args, **kwargs):
        kwargs['dimensions'] = 2
        super().__init__(*args, **kwargs)




if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 64
    x = torch.Tensor(1, 1, image_size, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))
    
    model = HighRes3DNet(in_channels=1, out_channels=1)
    
    out = model(x)
    print("out size: {}".format(out.size()))