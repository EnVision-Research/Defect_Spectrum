import argparse
import io
import pathlib
import os
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
from tqdm.auto import tqdm
from PIL import Image
from eval.fid_inception import InceptionV3
from utils.util import sample_data
from utils.eval_util import ImagePathDataset, IMAGE_EXTENSIONS


INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"



class FIDEvaluator:
    def __init__(
            self,
            device,
            ref_dataset_path,
            diffusion,
            noise_loader=None,
            num_samples=5000,
            dims=2048
    ):
        self.metrics = 'fid'
        self.dims = dims
        self.device = device
        self.diffusion = diffusion
        if noise_loader is None:
            self.noise_iterator = None
        else:
            self.noise_iterator = sample_data(noise_loader)
        self.num_samples = num_samples
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()
        self.ref_dataset_path = ref_dataset_path
        assert os.path.exists(ref_dataset_path)
        self.m, self.s = self.compute_statistics_from_path(ref_dataset_path)

    def compute_statistics_from_path(self, path, batch_size=50, num_workers=8):
        if path.endswith('.npz'):
            print('load statistics from {}'.format(path))
            with np.load(path) as f:
                m, s = f['mu'][:], f['sigma'][:]
        else:
            print('compute statistics from {}'.format(path))
            path = pathlib.Path(path)
            files = sorted([file for ext in IMAGE_EXTENSIONS
                           for file in path.glob('*.{}'.format(ext))])
            dataset = ImagePathDataset(files, transforms=transforms.ToTensor())
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
            pred_arr = np.empty((len(files), self.dims))

            start_idx = 0

            for batch in tqdm(dataloader):
                batch = batch.to(self.device)

                with torch.no_grad():
                    pred = self.model(batch)[0]

                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                pred = pred.squeeze(3).squeeze(2).cpu().numpy()

                pred_arr[start_idx:start_idx + pred.shape[0]] = pred

                start_idx = start_idx + pred.shape[0]

            m = np.mean(pred_arr, axis=0)
            s = np.cov(pred_arr, rowvar=False)

            np.savez('{}/ref_dataset_stats.npz'.format(self.ref_dataset_path), mu=m, sigma=s)

        return m, s

    def compute_statistics_from_diffusion(self, u_net, shape):

        if self.noise_iterator is None:
            bs, c, h, w = shape
            # change to bs*10 to accelerate
            bs *= 10
            noise = None
        else:
            noise = next(self.noise_iterator)
            noise = noise.to(self.device)
            bs, c, h, w = noise.shape

        process = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])

        pred_arr = np.empty((self.num_samples, self.dims))

        start_idx = 0

        for i in tqdm(range(self.num_samples // bs)):
            with torch.no_grad():
                samples = self.diffusion.p_sample_loop(u_net, (bs, c, h, w),
                                                          progress=False,
                                                          device=self.device,
                                                          noise=noise)
                samples = process(samples)
                pred = self.model(samples)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

            if self.noise_iterator is not None:
                noise = next(self.noise_iterator)
                noise = noise.to(self.device)

        m = np.mean(pred_arr, axis=0)
        s = np.cov(pred_arr, rowvar=False)

        return m, s

    def calculate_frechet_distance(self, mu1, sigma1, mu2=None, sigma2=None, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """
        if mu2 is None and sigma2 is None:
            mu2 = self.m
            sigma2 = self.s

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def eval(self, model, shape):
        generated_dataset_mu, generated_dataset_sigma = self.compute_statistics_from_diffusion(model, shape)
        return self.calculate_frechet_distance(generated_dataset_mu, generated_dataset_sigma, self.m, self.s)

