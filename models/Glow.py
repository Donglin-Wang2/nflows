import numpy as np

import torch

from nflows.nn import nets 
from nflows import transforms

from nflows import distributions
from nflows import flows



class Glow(flows.Flow):
    def __init__(self, img_shape, levels, hidden_channels, num_bits, alpha):
        
        mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)

        def create_mask(c):
            mask = torch.zeros(c)
            mask[::2] = 1
            return mask

        def create_convnet(in_channels, out_channels):
            net = nets.ConvResidualNet(in_channels=in_channels,
                                        out_channels=out_channels,
                                        hidden_channels=hidden_channels, num_blocks=3)
            return net

        c, h, w = img_shape
        for _ in range(levels):
            squeeze_layer = transforms.SqueezeTransform()
            c, h, w = squeeze_layer.get_output_shape(c, h, w)
            composite_transform = transforms.CompositeTransform(
                [squeeze_layer, 
                transforms.ActNorm(c),
                transforms.OneByOneConvolution(c),
                transforms.AffineCouplingTransform(
                    mask=create_mask(c),
                    transform_net_create_fn=create_convnet
                )
            ]
            )
            new_shape = mct.add_transform(composite_transform, (c, h, w))
            if new_shape:
                c, h, w = new_shape
        preprocess_transform = transforms.CompositeTransform([
            # Map to [0,1]
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            # Map into unconstrained space as done in alpha
            transforms.AffineScalarTransform(shift=alpha,
                                                scale=(1 - alpha)),
            transforms.Logit()
        ])
        distribution = distributions.StandardNormal((np.prod(img_shape),))
        transform = transforms.CompositeTransform([preprocess_transform, mct])
        super().__init__(transform, distribution)

if __name__ == '__main__':
    img_shape = (3, 32, 32)
    levels = 3
    hidden_channels = 128
    num_bits = 8
    alpha = 0.05
    model = Glow(img_shape, levels, hidden_channels, num_bits, alpha)
    log_prob = model.transform_to_noise(torch.randn((64, 3, 32, 32)))
    print(log_prob.shape)