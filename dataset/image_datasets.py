import math
import random
import os
from PIL import Image
# import blobfile as bf
# from mpi4py import MPI
import numpy as np
# import cv2
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
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

def get_first_five_images(root_dir, num_images):
    # Supported image file extensions
    img_extensions = ['.jpg', '.jpeg', '.png']

    all_images = []

    for foldername, subfolders, filenames in os.walk(root_dir):
        image_count = 0
        folder_images = []
        detect_good_folder = foldername.split('/')[-1] == "good"
        #print(detect_good_folder)
        if not detect_good_folder:
            for filename in sorted(filenames):  # Sorting ensures a consistent order
                if any(filename.lower().endswith(ext) for ext in img_extensions):
                    #print(os.path.join(foldername, filename))
                    folder_images.append(os.path.join(foldername, filename))
                    image_count += 1
                    if image_count >= num_images:
                        break

            all_images.extend(folder_images)

    return all_images


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    num_images, 
    deterministic=False,

):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset = ImageDataset(
        data_dir,
        image_size,
        num_images,
        
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True,
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def convert_to_onehot(mask_path):
    # Read the semantic mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    unique_values = np.unique(mask)
    # Remove 0 from unique values as we don't want a separate channel for it
    unique_values = unique_values[unique_values != 0]

    # Number of unique classes (excluding 0)
    num_classes = len(unique_values)

    # Create an empty multi-channel image
    h, w = mask.shape
    onehot_mask = np.zeros((h, w, num_classes), dtype=np.uint8)

    for i, val in enumerate(unique_values):
        onehot_mask[:, :, i][mask == val] = 255

    # Rearrange dimensions to have channels first
    onehot_mask = np.transpose(onehot_mask, (2, 0, 1))

    return onehot_mask
class ToTensorNoNorm(object):
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if not isinstance(pic, Image.Image):
            raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        img = img.permute(2, 0, 1).contiguous()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
class ImageDataset(Dataset):
    def __init__(self, dir, resolution, num_images):

        self.image_paths = sorted(get_first_five_images(dir+'/test', num_images))
        self.mask_paths = sorted(get_first_five_images(dir+'/converted_ground_truth', num_images))
        to_tensor_no_norm = ToTensorNoNorm()
        #print(self.image_paths)
        #print(self.mask_paths)
        self.transform = T.Compose([
            T.Resize((resolution,resolution)),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.transform_mask = T.Compose([
            #T.ToPILImage(),
            T.Resize((resolution,resolution), interpolation=Image.NEAREST),
            to_tensor_no_norm,
            #T.Normalize(mean=(0.0, 0.0, 0.0), std=(0.5, 0.5, 0.5)) #
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        f_imgpath = self.image_paths[index]
        f_maskpath = self.mask_paths[index]
        im = Image.open(f_imgpath).convert('RGB')
        mask = Image.open(f_maskpath)
        np.set_printoptions(threshold=np.inf)
        #print(np.asanyarray(mask))
        im = self.transform(im)
        mask = self.transform_mask(mask)

        return im, mask


def center_crop_arr(pil_image, image_size, crop_size):
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - crop_size) // 2
    crop_x = (arr.shape[1] - crop_size) // 2
    arr = arr[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
    arr = cv2.resize(arr, (image_size, image_size))
    return arr, (crop_y, crop_x)


def random_crop_arr(pil_image, image_size, crop_size, xy=None):
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - crop_size + 1) if xy is None else xy[0]
    crop_x = random.randrange(arr.shape[1] - crop_size + 1) if xy is None else xy[1]
    arr = arr[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
    arr = cv2.resize(arr, (image_size, image_size))
    return arr, (crop_y, crop_x)
