import numpy as np

import torch
import torchvision
import torch.nn as nn

from nflows.nn import nets 
from nflows import transforms

from nflows import distributions
from nflows import flows

from nflows.utils import torchutils



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
    
    def expand_noise(self, noise, context=None):
        embedded_context = self._embedding_net(context)

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=noise.shape[0]
            )

        samples, _ = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, noise.shape[0]])

        return samples

class GlowAdapted(Glow):
    def __init__(self, img_shape, levels, hidden_channels, num_bits, alpha):
        super().__init__(img_shape, levels, hidden_channels, num_bits, alpha)
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(img_shape), 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, np.prod(img_shape)),
            nn.Sigmoid()
        )

    def transform_to_noise(self, inputs, context=None):
        inputs = super().transform_to_noise(inputs, context=context)
        return self.encoder(inputs)

    def expand_noise(self, noise, context=None):
        noise = self.decoder(noise)
        return super().expand_noise(noise, context=context)
        


if __name__ == '__main__':
    img_shape = (3, 32, 32)
    levels = 3
    hidden_channels = 128
    num_bits = 8
    alpha = 0.05
    model = Glow(img_shape, levels, hidden_channels, num_bits, alpha)
    log_prob = model.transform_to_noise(torch.randn((64, 3, 32, 32)))
    print(log_prob.shape)
    noise = model._distribution.sample(64)
    sample = model.expand_noise(noise)
    torchvision.utils.make_grid(sample)
