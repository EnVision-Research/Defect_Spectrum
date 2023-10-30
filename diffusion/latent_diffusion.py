from diffusion.base_diffusion import *
from utils.util import *

class LatentColdDiffusion(ColdDiffusion):
    def __init__(
        self,
        autoencoder,
        **kwargs,
    ):  
        super().__init__(**kwargs)
        self.auto_encoder = instantiate_from_config(autoencoder).eval()
        for param in self.auto_encoder.parameters():
            param.requires_grad = False
    
    def training_losses(self, model, x_start, t, device=None, noise=None):
        x_start = self.auto_encoder.encode(x_start)

        if noise is not None:
            noise = self.auto_encoder.encode(noise)
        else:
            noise = torch.randn(x_start.shape).to(x_start.device)

        return super().training_losses(model, x_start, t, device, noise)
        
    def p_sample_loop(self, model, shape, noise=None, device=None, return_intermediates=False, log_interval=100, clip_denoised=True, progress=False):
        if device is None:
            device = next(model.parameters()).device

        img_shape = shape

        b = img_shape[0]

        if noise is None:
            latent_shape = self.auto_encoder.encode(torch.randn(*img_shape, device=device)).shape
            latent = torch.randn(
                *latent_shape,
                device=device)
        else:
            src_img = noise
            latent = self.auto_encoder.encode(src_img)

        intermediates = {
            'img': [],
            'x0_hat': [],
            'xT_hat': [],
        }

        if progress:
            indices = tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps)
        else:
            indices = reversed(range(0, self.num_timesteps))

 
        for i in indices:
            p_sample_ret = self.p_sample(
                                x_t=latent, 
                                t=torch.full((b,), i, device=device, dtype=torch.long),
                                model=model,
                                clip_denoised=clip_denoised
                                )
            latent = p_sample_ret['img']
            if i % log_interval == 0 or i == self.num_timesteps - 1:
                intermediates['img'].append(self.auto_encoder.decode(latent))
                intermediates['x0_hat'].append(self.auto_encoder.decode(p_sample_ret['x0_hat']))
                intermediates['xT_hat'].append(self.auto_encoder.decode(p_sample_ret['xT_hat']))
        if return_intermediates:
            return self.auto_encoder.decode(latent), intermediates
        return self.auto_encoder.decode(latent)



class LatentGaussianDiffusion(GaussianDiffusion):
    def __init__(
        self,
        autoencoder,
        **kwargs,
    ):  
        super().__init__(**kwargs)
        self.auto_encoder = instantiate_from_config(autoencoder).eval()
        for param in self.auto_encoder.parameters():
            param.requires_grad = False
    
    def training_losses(self, model, x_start, t, device=None, noise=None):
        x_start = self.auto_encoder.encode(x_start)

        if noise is not None:
            noise = self.auto_encoder.encode(noise)
        else:
            noise = torch.randn(x_start.shape).to(x_start.device)

        return super().training_losses(model, x_start, t, device, noise)
        
    def p_sample_loop(self, model, shape, noise=None, device=None, return_intermediates=False, log_interval=100, clip_denoised=True, progress=False):
        if device is None:
            device = next(model.parameters()).device

        img_shape = shape

        b = img_shape[0]

        if noise is None:
            latent_shape = self.auto_encoder.encode(torch.randn(*img_shape, device=device)).shape
            latent = torch.randn(
                *latent_shape,
                device=device)
        else:
            src_img = noise
            latent = self.auto_encoder.encode(src_img)
        
        intermediates = []

        if progress:
            indices = tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps)
        else:
            indices = reversed(range(0, self.num_timesteps))

 
        for i in indices:
            latent = self.p_sample(
                                x_t=latent, 
                                t=torch.full((b,), i, device=device, dtype=torch.long),
                                model=model,
                                clip_denoised=clip_denoised
                                )
            if i % log_interval == 0 or i == self.num_timesteps - 1:
                intermediates.append(self.auto_encoder.decode(latent))
        if return_intermediates:
            return self.auto_encoder.decode(latent), intermediates
        return self.auto_encoder.decode(latent)