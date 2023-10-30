import os
import torch
from torch import nn
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T, utils


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
    return images


class ImageDataset(Dataset):
    def __init__(self, dir, resolution, h_flip):

        self.image_paths = sorted(make_dataset(dir))
        self.transform = T.Compose([
            T.Resize(resolution),
            T.RandomHorizontalFlip() if h_flip else nn.Identity(),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        fname, fpath = self.image_paths[index]
        im = Image.open(fpath).convert('RGB')
        im = self.transform(im)
        return im

class PairImageDataset(Dataset):
    def __init__(self, src_dir, tgt_dir, resolution, h_flip, pair):

        self.src = sorted(make_dataset(src_dir))
        self.tgt = sorted(make_dataset(tgt_dir))

        if not pair:
            random.shuffle(self.src)
        else: 
            assert len(self.src) == len(self.tgt), f"pair dataset should have same number of images, {len(self.src)}, {len(self.tgt)}"

        self.preprocess = T.Compose([
            T.Resize(resolution),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.augmentation = T.Compose([
            T.RandomHorizontalFlip() if h_flip else nn.Identity(),
            T.CenterCrop(resolution),
        ])

    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):
        _, src_fpath = self.src[index]
        src_im = Image.open(src_fpath).convert('RGB')
        src_im = self.preprocess(src_im)

        _, tgt_fpath = self.tgt[index]
        tgt_im = Image.open(tgt_fpath).convert('RGB')
        tgt_im = self.preprocess(tgt_im)

        src_im, tgt_im = self.augmentation(
            torch.cat([src_im.unsqueeze(0), tgt_im.unsqueeze(0)], 0))

        return src_im, tgt_im 



