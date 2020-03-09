import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms

from utils.loaders import Compose, RandomHorizontallyFlip, GTScaleDown, LabelNormalize


class GTDataset(Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files)
        self.main_transform=main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, den = self.main_transform(img,den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples



LABEL_FACTOR = 1
def random_crop_GT(img,den,dst_size):
    # dst_size: ht, wd

    _,ts_hd,ts_wd = img.shape

    x1 = random.randint(0, ts_wd - dst_size[1])//LABEL_FACTOR*LABEL_FACTOR
    y1 = random.randint(0, ts_hd - dst_size[0])//LABEL_FACTOR*LABEL_FACTOR
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]

    label_x1 = x1//LABEL_FACTOR
    label_y1 = y1//LABEL_FACTOR
    label_x2 = x2//LABEL_FACTOR
    label_y2 = y2//LABEL_FACTOR

    return img[:,y1:y2,x1:x2], den[label_y1:label_y2,label_x1:label_x2]



def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out

crop_size = 256

def GT_collate(batch):
    # @GJY
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch)) # imgs and dens
    imgs, dens = [transposed[0],transposed[1]]


    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):

        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = random_crop_GT(imgs[i_sample],dens[i_sample],[crop_size,crop_size])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)


        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))

        return [cropped_imgs,cropped_dens]

    raise TypeError((error_msg.format(type(batch[0]))))


def loading_data_GT(batch_size=5, num_workers=8):
    mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
    log_para = 100.
    factor = 1
    DATA_PATH = "data/gt"


    train_main_transform = Compose([
        RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        GTScaleDown(factor),
        LabelNormalize(log_para)
    ])

    train_set = GTDataset(DATA_PATH+'/train', 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader =None
    if batch_size == 1:
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, drop_last=True)
    elif batch_size > 1:
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, collate_fn=GT_collate, shuffle=True, drop_last=True)

    val_set = GTDataset(DATA_PATH+'/val', 'val', main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=num_workers, shuffle=True, drop_last=False)

    test_set = GTDataset(DATA_PATH+'/test', 'test', main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers, shuffle=True, drop_last=False)

    return train_loader, val_loader, test_loader
