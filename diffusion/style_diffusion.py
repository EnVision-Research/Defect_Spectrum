
from diffusion.base_diffusion import *
from utils.util import *

class StyleDiffusion(ColdDiffusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def p_sample(self, model, x_t, t, clip_denoised: bool, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        ret_dict = {}
        if self.reversed_sample:
            model_out = model(x_t, self.num_timesteps - t, **model_kwargs)
        else:
            model_out = model(x_t, beta=extract(self.betas, t, x_t.shape), **model_kwargs)

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
            x_prev_hat = self.q_sample(x_start=x_prev_hat, t=t-1, noise=eps)

        ret_dict['img'] = x_t - x_hat + x_prev_hat
        return ret_dict
    
    
    def training_losses(self, model, x_start, t, device=None, noise=None, model_kwargs=None):
        if device is None:
            device = next(model.parameters()).device

        if noise is None:
            noise = torch.randn(*x_start.shape, device=device)
        
        if model_kwargs is None:
            model_kwargs = {}

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        beta = extract(self.betas, t, x_t.shape)

        model_out = model(x_t, beta, **model_kwargs)

        if self.parameterization == "eps":
            target = noise
        else:
            target = x_start
        
        loss = nn.functional.mse_loss(target, model_out)

        loss_dict = {}
        loss_dict.update({f'loss':loss})
        return loss, loss_dict
    


