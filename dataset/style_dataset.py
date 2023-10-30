import torch
from torch.utils.data import Dataset

class LatentsDataset(Dataset):

    def __init__(self, opts):
        self.latents = torch.load(opts['latents_w_path'],map_location=torch.device('cpu'))

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):

        return self.latents[index]


class W2CLIPDataset(Dataset):

    def __init__(self, style_dir, clip_dir):
        self.latents_w = torch.load(style_dir, map_location=torch.device('cpu'))
        self.clip_features = torch.load(clip_dir, map_location=torch.device('cpu'))

    def __len__(self):
        return self.latents_w.shape[0]

    def __getitem__(self, index):

        return self.latents_w[index], self.clip_features[index]

