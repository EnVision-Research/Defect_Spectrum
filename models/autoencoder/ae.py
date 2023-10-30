from base64 import encode
from mimetypes import init
import torch
from torch import nn
from utils.dist_util import *
import torch.nn.functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            all_reduce(embed_onehot_sum)
            all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class AE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        stride=4,
        ckpt=None
    ):
        super().__init__()

        self.enc = Encoder(
            in_channel=in_channel, 
            channel=channel, 
            n_res_block=n_res_block, 
            n_res_channel=n_res_channel, 
            stride=stride
        )

        self.dec = Decoder(
            in_channel=channel,
            out_channel=out_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            stride=stride
        )
        if ckpt is not None:
            self.load_state_dict(torch.load(ckpt, map_location='cpu')["model"])

    def forward(self, input):
        x = self.encode(input)
        y = self.decode(x)

        return y

    def encode(self, input):
        return self.enc(input)

    def decode(self, input):
        return self.dec(input)



class VAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        stride=4,
        ckpt=None,
    ):
        super().__init__()

        self.enc = Encoder(
            in_channel=in_channel, 
            channel=channel, 
            n_res_block=n_res_block, 
            n_res_channel=n_res_channel, 
            stride=stride
        )

        self.dec = Decoder(
            in_channel=channel,
            out_channel=out_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            stride=stride
        )
        
        if ckpt is not None:
            self.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])

    def forward(self, x):
        mu = self.encode(x)
        z = self.reparameterize(mu)
        y = self.decode(z)
        return mu, y
    
    def reparameterize(self, mu):
        eps = torch.randn_like(mu)
        return mu + eps

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        out_channel=3,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        ckpt_path=None
    ):
        super().__init__()

        self.enc = Encoder(
            in_channel=in_channel, 
            channel=channel, 
            n_res_block=n_res_block, 
            n_res_channel=n_res_channel, 
            stride=2
        )

        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.quantize = Quantize(embed_dim, n_embed)

        self.dec = Decoder(
            in_channel=embed_dim, 
            out_channel=out_channel, 
            channel=channel, 
            n_res_block=n_res_block, 
            n_res_channel=n_res_channel, 
            stride=2
        )
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)

        return dec, diff

    def encode(self, input):
        enc = self.enc(input)
        quant = self.quantize_conv(enc).permute(0, 2, 3, 1)
        quant, diff, id = self.quantize(quant)
        quant = quant.permute(0, 3, 1, 2)

        return quant, diff, id

    def decode(self, quant):
        dec = self.dec(quant)
        return dec
    
    def decode_code(self, code):
        quant = self.quantize.embed_code(code)
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant)

        return dec

class VQInterface(VQVAE):
    def __init__(self, in_channel=3, channel=128, out_channel=3, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99, ckpt_path=None):
        super().__init__(in_channel, channel, out_channel, n_res_block, n_res_channel, embed_dim, n_embed, decay, ckpt_path)
    
    def encode(self, input):
        enc = self.enc(input)
        quant = self.quantize_conv(enc)
        return quant
    
    def decode(self, quant):
        quant, diff, id = self.quantize(quant.permute(0, 2, 3, 1))
        quant = quant.permute(0, 3, 1, 2)

        dec = self.dec(quant)
        return dec


class VQVAE2(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        out_channel=3,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        ckpt_path=None,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            out_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


class VQInterface2(VQVAE2):
    def __init__(self, in_channel=3, channel=128, out_channel=3, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, ckpt_path=None):
        super().__init__(in_channel, channel, out_channel, n_res_block, n_res_channel, embed_dim, n_embed, ckpt_path)
    
    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)

        return quant

    def decode(self, quant):
        return self.dec(quant)