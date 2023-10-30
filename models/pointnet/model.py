import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
from torch.nn import Module, Linear
import numpy as np
import math


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class PointwiseNet(Module):

    def __init__(self, point_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, 3),
            ConcatSquashLinear(128, 256, 3),
            ConcatSquashLinear(256, 512, 3),
            ConcatSquashLinear(512, 256, 3),
            ConcatSquashLinear(256, 128, 3),
            ConcatSquashLinear(128, point_dim, 3)
        ])

    def forward(self, x, timestep):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            timestep:     Time. (B, ).
        """
        b = x.shape[0]

        timestep = timestep.view(b, 1, 1)          # (B, 1, 1)
        time_emb = torch.cat([timestep, torch.sin(timestep), torch.cos(timestep)], dim=-1) # (B, 1, 3)
        # ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=time_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


# class DiffusionPoint(Module):

#     def __init__(self, net, var_sched:VarianceSchedule):
#         super().__init__()
#         self.net = net
#         self.var_sched = var_sched

#     def get_loss(self, x_0, context, t=None):
#         """
#         Args:
#             x_0:  Input point cloud, (B, N, d).
#             context:  Shape latent, (B, F).
#         """
#         batch_size, _, point_dim = x_0.size()
#         if t == None:
#             t = self.var_sched.uniform_sample_t(batch_size)
#         alpha_bar = self.var_sched.alpha_bars[t]
#         beta = self.var_sched.betas[t]

#         c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
#         c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

#         e_rand = torch.randn_like(x_0)  # (B, N, d)
#         e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

#         loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
#         return loss

#     def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
#         batch_size = context.size(0)
#         x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
#         traj = {self.var_sched.num_steps: x_T}
#         for t in range(self.var_sched.num_steps, 0, -1):
#             z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
#             alpha = self.var_sched.alphas[t]
#             alpha_bar = self.var_sched.alpha_bars[t]
#             sigma = self.var_sched.get_sigmas(t, flexibility)

#             c0 = 1.0 / torch.sqrt(alpha)
#             c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

#             x_t = traj[t]
#             beta = self.var_sched.betas[[t]*batch_size]
#             e_theta = self.net(x_t, beta=beta, context=context)
#             x_next = c0 * (x_t - c1 * e_theta) + sigma * z
#             traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
#             traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
#             if not ret_traj:
#                 del traj[t]
        
#         if ret_traj:
#             return traj
#         else:
#             return traj[0]