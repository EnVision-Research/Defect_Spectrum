import torch
import torch.nn as nn
from utils.dist_util import get_rank
from diffusion.base_diffusion import ColdDiffusion


class SymmetricalColdDiffusion(ColdDiffusion):
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised: bool):
        ret_dict = {}
        if get_rank() == 0 and self.reversed_sample:
            print('*****WARNING! You are using reversed_sample and symmetrical t at the same time! *****')
        if self.reversed_sample:
            t = (self.num_timesteps / 2 - torch.abs(self.num_timesteps / 2 - t)).long()
            model_out = model(x_t, self.num_timesteps - t)
        else:
            t = (self.num_timesteps / 2 - torch.abs(self.num_timesteps / 2 - t)).long()
            model_out = model(x_t, t)

        if self.parameterization == "eps":
            eps = model_out
            pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps)

        elif self.parameterization == "x0":
            pred_xstart = model_out
            eps = self._predict_eps_from_xstart(x_t=x_t, t=t, pred_xstart=pred_xstart)
        else:
            raise NotImplementedError

        ret_dict['xT_hat'] = eps
        ret_dict['x0_hat'] = pred_xstart

        if clip_denoised:
            ret_dict['x0_hat'].clamp_(-1., 1.)

        x_hat = ret_dict['x0_hat']
        if not torch.equal(t, torch.ones_like(t)):
            x_hat = self.q_sample(x_start=x_hat, t=t, noise=eps)

        x_prev_hat = ret_dict['x0_hat']
        if not torch.equal(t, torch.zeros_like(t)):
            x_prev_hat = self.q_sample(x_start=x_prev_hat, t=t - 1, noise=eps)

        ret_dict['img'] = x_t - x_hat + x_prev_hat
        return ret_dict

    def training_losses(self, model, x_start, t, device=None, noise=None):

        if device is None:
            device = next(model.parameters()).device

        if noise is None:
            noise = torch.randn(*x_start.shape, device=device)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        t = (self.num_timesteps / 2 - torch.abs(self.num_timesteps / 2 - t)).long()
        model_out = model(x_t, t)

        if self.parameterization == "eps":
            target = noise
        else:
            target = x_start

        # loss = nn.functional.mse_loss(target, model_out)
        loss = nn.functional.l1_loss(target, model_out)

        loss_dict = {}
        loss_dict.update({f'loss': loss})
        return loss, loss_dict
