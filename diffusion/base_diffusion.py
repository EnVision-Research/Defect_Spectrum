from torch import nn
from tqdm import tqdm

from utils_new.util import *


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        parameterization,
        schedule,
        num_timestep,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3
    ):
        super().__init__()
        self.parameterization = parameterization
        betas = self.make_beta_schedule(
                schedule=schedule, num_timestep=num_timestep,
                linear_start=linear_start, linear_end=linear_end,
                cosine_s=cosine_s
            )
        self.register_schedule(betas)
        self.schedule = schedule
        self.reversed_sample = False


    def make_beta_schedule(self, schedule, num_timestep, linear_start, linear_end, cosine_s):
        if schedule == "linear":
            betas = (
                    torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timestep, dtype=torch.float64) ** 2
            )
        elif schedule == "cosine":
            timesteps = (
                torch.arange(num_timestep + 1, dtype=torch.float64) / num_timestep + cosine_s
            )
            alphas_cumprod = torch.cos(timesteps / (1 + cosine_s) * np.pi / 2).pow(2)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = np.clip(betas, a_min=0, a_max=0.999)
        elif schedule == "sqrt_linear":
            betas = torch.linspace(linear_start, linear_end, num_timestep, dtype=torch.float64).numpy()
        elif schedule == "sqrt":
            betas = (torch.linspace(linear_start, linear_end, num_timestep, dtype=torch.float64) ** 0.5).numpy()
        elif schedule == "inverse_proportion":
            betas = np.array([1 / (num_timestep - t + 1) for t in range(0, num_timestep)])
        else:
            raise ValueError(f"schedule '{schedule}' unknown.")

        return betas


    def register_schedule(self, betas):
        betas = torch.tensor(betas, dtype=torch.float32)
        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float32), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)),
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, var, log_var_clipped


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """

        mean = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance


    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn(*x_start.shape,device=x_start.device)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
  

    def p_mean_variance(self, model, x_t, t, clip_denoised: bool, model_kwargs=None):
        ret_dict = {}
        if self.reversed_sample:
            model_out = model(x_t, self.num_timesteps - t, **model_kwargs)
        else:
            model_out = model(x_t, t, **model_kwargs)
        if self.parameterization == "eps":
            pred_xstart = self._predict_xstart_from_eps(x_t, t=t, eps=model_out)
            ret_dict['xT_hat'] = model_out
            ret_dict['x0_hat'] = pred_xstart

        elif self.parameterization == "x0":
            pred_xstart = model_out
            pred_eps = self._predict_eps_from_xstart(x_t, t, pred_xstart)
            ret_dict['xT_hat'] = pred_eps
            ret_dict['x0_hat'] = pred_xstart
        else:
            raise NotImplementedError

        if clip_denoised:
            ret_dict['x0_hat'].clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_start=ret_dict['x0_hat'], x_t=x_t, t=t)
        ret_dict['model_mean'] = model_mean
        ret_dict['posterior_variance'] = posterior_variance
        ret_dict['posterior_log_variance'] = posterior_log_variance
        return ret_dict

    @torch.no_grad()
    def p_sample(self, x_t, model, t, clip_denoised=True, model_kwargs=None):
        ret_dict = {}
        b, *_, device = *x_t.shape, x_t.device
        p_mean_variance_ret = self.p_mean_variance(model=model, x_t=x_t, t=t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        model_mean = p_mean_variance_ret['model_mean']
        model_log_variance = p_mean_variance_ret['posterior_log_variance']
        ret_dict['x0_hat'] = p_mean_variance_ret['x0_hat']
        ret_dict['xT_hat'] = p_mean_variance_ret['xT_hat']

        noise = torch.randn(*x_t.shape, device=device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
        ret_dict['img'] = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return ret_dict
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, noise=None, device=None, return_intermediates=False,
                      log_interval=100, clip_denoised=True, progress=False, model_kwargs=None):
        if device is None:
            device = next(model.parameters()).device

        if model_kwargs is None:
            model_kwargs = {}

        b = shape[0]
        
        if noise is None:
            img = torch.randn(*shape, device=device, dtype=torch.float32)
        else:
            img = noise
        intermediates = {
            'img': [],
            'x0_hat': [],
            'xT_hat': [],
        }
        #print(model_kwargs["num_timesteps"])
        if model_kwargs["num_timesteps"] is not None: 
            start, end = model_kwargs["num_timesteps"]
            self.num_timesteps = start - end

        if progress:
            indices = tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps)
        else:
            indices = reversed(range(0, self.num_timesteps))
        for i in indices:
            p_sample_ret = self.p_sample(
                x_t=img,
                t=torch.full((b,), i, device=device, dtype=torch.long),
                model=model,
                clip_denoised=clip_denoised,
                model_kwargs=None
            )
            img = p_sample_ret['img']
            if i % log_interval == 0 or i == self.num_timesteps - 1:
                intermediates['img'].append(p_sample_ret['img'])
                intermediates['x0_hat'].append(p_sample_ret['x0_hat'])
                intermediates['xT_hat'].append(p_sample_ret['xT_hat'])
        if return_intermediates:
            return img, intermediates
        return img


    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )


    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


    def training_losses(self, model, x_start, t, device=None, noise=None, model_kwargs=None, seperate_channel_loss=False):

        if device is None:
            device = next(model.parameters()).device

        if noise is None:
            noise = torch.randn(*x_start.shape, device=device)
        
        if model_kwargs is None:
            model_kwargs = {}

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = model(x_t, t, **model_kwargs)

        if self.parameterization == "eps":
            target = noise
        else:
            target = x_start

        loss_dict = {}
        
        if seperate_channel_loss:
            rgb_part, mask_part = model_out[:, :3, :, :], model_out[:, 3:, :, :]
            rgb_target_part, mask_target_part = target[:, :3, :, :], target[:, 3:, :, :]
            
            
            loss_rgb = nn.functional.mse_loss(rgb_part, rgb_target_part)
            loss_mask = nn.functional.cross_entropy(mask_part, mask_target_part)
            loss = loss_rgb + 0.2 * loss_mask
            loss_dict.update({f'loss':loss, f'loss_rgb':loss_rgb, f'loss_mask':loss_mask})
        else:

            loss = nn.functional.mse_loss(model_out, target)
            loss_dict.update({f'loss':loss})


        return loss, loss_dict


class ColdDiffusion(GaussianDiffusion):
    def __init__(
        self,
        **kwargs,
    ):  
        super().__init__(**kwargs)

    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised: bool, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        ret_dict = {}
        if self.reversed_sample:
            model_out = model(x_t, self.num_timesteps - t, **model_kwargs)
        else:
            model_out = model(x_t, t, **model_kwargs)

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
