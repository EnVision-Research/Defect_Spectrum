import math
import random
from turtle import forward

import torch
from torch import nn
from torch.nn import functional as F

from models.stylegan.modules import *


class StyleGANv2Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim=512,
        n_mlp=8,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        input_is_stylespace=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent and not input_is_stylespace:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1 and not input_is_stylespace:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if input_is_stylespace:
            latent = styles[0]
        elif len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)


        style_vector = []

        if not input_is_stylespace:
            out = self.input(latent)
            out, out_style = self.conv1(out, latent[:, 0], noise=noise[0])
            style_vector.append(out_style)

            skip, out_style = self.to_rgb1(out, latent[:, 1])
            style_vector.append(out_style)

            i = 1
        else:
            out = self.input(latent[0])
            out, out_style = self.conv1(out, latent[0], noise=noise[0], input_is_stylespace=input_is_stylespace)
            style_vector.append(out_style)

            skip, out_style = self.to_rgb1(out, latent[1], input_is_stylespace=input_is_stylespace)
            style_vector.append(out_style)

            i = 2

        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            if not input_is_stylespace:
                out, out_style1 = conv1(out, latent[:, i], noise=noise1)
                out, out_style2 = conv2(out, latent[:, i + 1], noise=noise2)
                skip, rgb_style = to_rgb(out, latent[:, i + 2], skip)

                style_vector.extend([out_style1, out_style2, rgb_style])

                i += 2
            else:
                out, out_style1 = conv1(out, latent[i], noise=noise1, input_is_stylespace=input_is_stylespace)
                out, out_style2 = conv2(out, latent[i + 1], noise=noise2, input_is_stylespace=input_is_stylespace)
                skip, rgb_style = to_rgb(out, latent[i + 2], skip, input_is_stylespace=input_is_stylespace)

                style_vector.extend([out_style1, out_style2, rgb_style])

                i += 3

        image = skip

        if return_latents:
            return image, latent, style_vector

        else:
            return image, None


class StyleGANv2Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            # EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4] * 4 * 8, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

class DiffusionStyle(nn.Module):
    def __init__(
        self,
        n_mlp,
        style_dim,
        lr_mlp=0.01,
        use_scale_shift_norm=False
    ):
        super().__init__()
        self.style_dim = style_dim        
        layers = [PixelNorm()]
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        self.use_scale_shift_norm = use_scale_shift_norm

        self.time_emb_layer = nn.Sequential(
            nn.Linear(
                style_dim,
                style_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                style_dim,
                2 * style_dim if use_scale_shift_norm else style_dim,
            ),
        )

        self.cond_emb_layer = nn.Sequential(
            nn.Linear(
                style_dim,
                2 * style_dim if use_scale_shift_norm else style_dim,
            ),
            nn.SiLU(),
        )

    def forward(self, x, timesteps, y=None):
        """
        :param x: [N x 1 x ...]
        :param timesteps: a 1-D batch of timesteps
        :param y: an [N] Tensor of labels
        """
        emb = self.time_emb_layer(timestep_embedding(timesteps, self.style_dim))

        if y is not None:
            assert y.shape == x.shape
            emb = emb + self.cond_emb_layer(y)
    
        skip = x
        
        for layer in self.style:
            output = layer(x)

            if self.use_scale_shift_norm:
                scale, shift = torch.chunk(emb, 2, dim=1)
                output = scale * output + shift
            else:
                output = output + emb

            x = output

        return x + skip





def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding