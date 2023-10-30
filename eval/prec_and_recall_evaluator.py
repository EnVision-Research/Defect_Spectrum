import os
from collections import namedtuple
from functools import partial
from glob import glob

import numpy as np
import pathlib
from utils.eval_util import IMAGE_EXTENSIONS, ImagePathDataset
from utils.util import sample_data

try:
    from tqdm import tqdm, trange
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return x


    def trange(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return range(x)

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PrecisionAndRecall = namedtuple('PrecisinoAndRecall', ['precision', 'recall'])


class PrecAndRecallEvaluator:
    def __init__(self,
                 device,
                 ref_dataset_path,
                 diffusion,
                 noise_loader=None,
                 num_samples=5,
                 k=3,
                 distance_metric='euclidean',
                 vgg16=None):
        self.metrics = 'P&R'
        # computer_ref_dataset_features
        if noise_loader is None:
            self.noise_iterator = None
        else:
            self.noise_iterator = sample_data(noise_loader)
        self.device = device
        self.diffusion = diffusion
        self.num_samples = num_samples
        self.k = k
        self.distance_metric = distance_metric
        if vgg16 is None:
            print('loading vgg16 for improved precision and recall...', end='', flush=True)
            self.vgg16 = models.vgg16(pretrained=True).cuda().eval()
            print('done')
        else:
            self.vgg16 = vgg16

        self.ref_stats = dict()
        feats, radii = self.computer_ref_features(ref_dataset_path)
        self.ref_stats['feats'] = feats
        self.ref_stats['radii'] = radii


    def computer_ref_features(self, path, batch_size=1, num_workers=8):
        if path.endswith('.npz'):
            print('load ref_features from {}'.format(path))
            with np.load(path) as f:
                feats, radii = f['feats'][:], f['radii'][:]
        else:
            print('compute ref_features from {}'.format(path))
            path = pathlib.Path(path)
            files = sorted([file for ext in IMAGE_EXTENSIONS
                            for file in path.glob('*.{}'.format(ext))])
            num_found_images = len(files)
            # TODO: remove the 224 224 resize
            dataset = ImagePathDataset(files, transforms=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)

            desc = 'extracting features of %d images' % num_found_images
            if num_found_images < self.num_samples:
                print('WARNING: num_found_images(%d) < num_samples(%d)' % (num_found_images, self.num_samples))

            features = []
            for idx, batch in enumerate(tqdm(dataloader, desc=desc)):
                with torch.no_grad():
                    data_ = batch
                    feature = self.vgg16.features(data_.cuda())
                    feature = feature.view(-1, 7 * 7 * 512)
                    feature = self.vgg16.classifier[:4](feature)
                    features.append(feature.cpu().data.numpy())
                    if idx >= self.num_samples:
                        break

            feats = np.concatenate(features, axis=0)
            distances = self.compute_pairwise_distances(feats, distance_metric=self.distance_metric)
            radii = self.distances2radii(distances, k=self.k)

        return feats, radii

    def compute_statistics_from_diffusion(self, u_net, shape):
        if self.noise_iterator is None:
            bs, c, h, w = shape
            # change to bs*10 to accelerate
            bs *= 1
            noise = None
        else:
            noise = next(self.noise_iterator)
            noise = noise.to(self.device)
            bs, c, h, w = noise.shape

        process = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])

        features = []
        for i in tqdm(range(self.num_samples // bs)):
            with torch.no_grad():
                samples = self.diffusion.p_sample_loop(u_net, (bs, c, h, w),
                                                       progress=False,
                                                       device=self.device,
                                                       noise=noise)
                samples = process(samples)

                before_fc = self.vgg16.features(samples.cuda())
                before_fc = before_fc.view(-1, 7 * 7 * 512)
                feature = self.vgg16.classifier[:4](before_fc)
                features.append(feature.cpu().data.numpy())

            if self.noise_iterator is not None:
                noise = next(self.noise_iterator)
                noise = noise.to(self.device)

        feats = np.concatenate(features, axis=0)
        distances = self.compute_pairwise_distances(feats, distance_metric=self.distance_metric)
        radii = self.distances2radii(distances, k=self.k)

        return feats, radii

    def compute_metric(self, manifold_ref, feats_subject, desc='', distance_metric='euclidean', max_radii=None):
        num_subjects = feats_subject.shape[0]
        count = 0
        dist = self.compute_pairwise_distances(manifold_ref['feats'], feats_subject, distance_metric=distance_metric)

        radii = manifold_ref['radii']
        if max_radii is not None:
            idx = radii > max_radii
            radii[idx] = max_radii

        for i in trange(num_subjects, desc=desc):
            count += (dist[:, i] < radii).any()

        return count / num_subjects

    def compute_pairwise_distances(self, X, Y=None, distance_metric='euclidean'):
        '''
        args:
            X: np.array of shape N x dim
            Y: np.array of shape N x dim
        returns:
            N x N symmetric np.array
        '''
        num_X = X.shape[0]
        if Y is None:
            num_Y = num_X
        else:
            num_Y = Y.shape[0]
        X = X.astype(np.float64)  # to prevent underflow
        X_norm_square = np.sum(X ** 2, axis=1, keepdims=True)
        if Y is None:
            Y_norm_square = X_norm_square
        else:
            Y_norm_square = np.sum(Y ** 2, axis=1, keepdims=True)

        if Y is None:
            Y = X
        XY = np.dot(X, Y.T)

        if distance_metric == 'euclidean':
            X_square = np.repeat(X_norm_square, num_Y, axis=1)
            Y_square = np.repeat(Y_norm_square.T, num_X, axis=0)
            diff_square = X_square - 2 * XY + Y_square

            # check negative distance
            min_diff_square = diff_square.min()
            if min_diff_square < 0:
                idx = diff_square < 0
                diff_square[idx] = 0
                print('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
                      min_diff_square)

            distances = np.sqrt(diff_square)
        elif distance_metric == 'cosine':
            X_norm = np.sqrt(X_norm_square)
            Y_norm = np.sqrt(Y_norm_square)
            cos_distance = XY / np.dot(X_norm, Y_norm.T)
            distances = 1 - cos_distance
        else:
            raise NotImplementedError

        return distances

    def distances2radii(self, distances, k=3):
        def get_kth_value(np_array, k):
            kprime = k + 1  # kth NN should be (k+1)th because closest one is itself
            idx = np.argpartition(np_array, kprime)
            k_smallests = np_array[idx[:kprime]]
            kth_value = k_smallests.max()
            return kth_value

        num_features = distances.shape[0]
        radii = np.zeros(num_features)
        for i in range(num_features):
            radii[i] = get_kth_value(distances[i], k=k)
        return radii

    def compute_precision_and_recall(self, stats_dict):
        precision = self.compute_metric(self.ref_stats, stats_dict['feats'], 'computing precision...', distance_metric=self.distance_metric)
        recall = self.compute_metric(stats_dict, self.ref_stats['feats'], 'computing recall...', distance_metric=self.distance_metric)

        return PrecisionAndRecall(precision, recall)._asdict()

    def eval(self, model, shape):
        generated_dataset_stats_dict = {'feats': (self.compute_statistics_from_diffusion(model, shape))[0],
                                        'radii': (self.compute_statistics_from_diffusion(model, shape))[1]}
        return self.compute_precision_and_recall(generated_dataset_stats_dict)


