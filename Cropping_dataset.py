import json
import os
import random

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from config_cropping import cfg

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def rescale_bbox(bbox, ratio_w, ratio_h):
    bbox = np.array(bbox).reshape(-1, 4)
    bbox[:, 0] = np.floor(bbox[:, 0] * ratio_w)
    bbox[:, 1] = np.floor(bbox[:, 1] * ratio_h)
    bbox[:, 2] = np.ceil(bbox[:, 2] * ratio_w)
    bbox[:, 3] = np.ceil(bbox[:, 3] * ratio_h)
    return bbox.astype(np.float32)


class FCDBDataset(Dataset):
    def __init__(self, split, keep_aspect_ratio=False):
        self.split = split
        self.keep_aspect = keep_aspect_ratio
        self.data_dir = cfg.FCDB_dir
        assert os.path.exists(self.data_dir), self.data_dir
        self.image_dir = os.path.join(self.data_dir, f'{split}ing')
        assert os.path.exists(self.image_dir), self.image_dir
        self.annos = self.parse_annotations(split)
        self.image_list = list(self.annos.keys())
        self.data_augment = (cfg.data_augmentation and self.split == 'train')
        self.PhotometricDistort = T.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)
        self.image_transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def parse_annotations(self, split):
        if split == 'train':
            split_file = os.path.join(self.data_dir, 'FCDB-training.json')
        else:
            split_file = os.path.join(self.data_dir, 'FCDB-testing.json')
        assert os.path.exists(split_file), split_file
        origin_data = json.loads(open(split_file, 'r').read())
        annos = dict()
        for item in origin_data:
            url = item['url']
            image_name = os.path.split(url)[-1]
            if os.path.exists(os.path.join(self.image_dir, image_name)):
                x, y, w, h = item['crop']
                crop = [x, y, x + w, y + h]
                annos[image_name] = crop
        print('{} set, {} images'.format(split, len(annos)))
        return annos

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        if self.keep_aspect:
            scale = float(cfg.image_size[0]) / min(im_height, im_width)
            h = round(im_height * scale / 32.0) * 32
            w = round(im_width * scale / 32.0) * 32
        else:
            h = cfg.image_size[1]
            w = cfg.image_size[0]

        crop = self.annos[image_name]
        crop = np.array(crop).reshape(-1, 4)

        if self.split == 'train' and cfg.GPCA:
            if random.uniform(0, 1) > cfg.GPCA:
                g_x1, g_y1, g_x2, g_y2 = crop[0]
                x1 = random.randint(0, g_x1)
                y1 = random.randint(0, g_y1)
                x2 = random.randint(g_x2, im_width)
                y2 = random.randint(g_y2, im_height)

                image = image.crop((x1, y1, x2, y2))

                crop[:, 0::2] -= x1
                crop[:, 1::2] -= y1

        crop = crop.astype(np.float32)
        resized_image = image.resize((w, h), Image.Resampling.LANCZOS)

        if self.data_augment:
            if random.uniform(0, 1) > 0.5:
                resized_image = ImageOps.mirror(resized_image)
                temp_x1 = crop[:, 0].copy()
                crop[:, 0] = im_width - crop[:, 2]
                crop[:, 2] = im_width - temp_x1
            resized_image = self.PhotometricDistort(resized_image)
        im = self.image_transformer(resized_image)
        return im, crop, im_width, im_height, image_file


class FLMSDataset(Dataset):
    def __init__(self, split='test', keep_aspect_ratio=False, transforms_v1=False):
        self.keep_aspect = keep_aspect_ratio
        self.data_dir = cfg.FLMS_dir
        assert os.path.exists(self.data_dir), self.data_dir
        self.image_dir = os.path.join(self.data_dir, 'image')
        assert os.path.exists(self.image_dir), self.image_dir
        self.annos = self.parse_annotations()
        self.image_list = list(self.annos.keys())
        self.transforms_pil = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ])
        self.transforms_v2 = T.Compose([
            T.Resize((224, 224)),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ])
        self.transforms_v1 = transforms_v1

    def parse_annotations(self):
        image_crops_file = os.path.join(self.data_dir, '500_image_dataset.mat')
        assert os.path.exists(image_crops_file), image_crops_file
        import scipy.io as scio
        image_crops = dict()
        anno = scio.loadmat(image_crops_file)
        for i in range(anno['img_gt'].shape[0]):
            image_name = anno['img_gt'][i, 0][0][0]
            gt_crops = anno['img_gt'][i, 0][1]
            gt_crops = gt_crops[:, [1, 0, 3, 2]]
            keep_index = np.where((gt_crops < 0).sum(1) == 0)
            gt_crops = gt_crops[keep_index].tolist()
            image_crops[image_name] = gt_crops
        print('{} images'.format(len(image_crops)))
        return image_crops

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)

        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        if self.transforms_v1:
            if self.keep_aspect:
                scale = float(cfg.image_size[0]) / min(im_height, im_width)
                h = round(im_height * scale / 32.0) * 32
                w = round(im_width * scale / 32.0) * 32
            else:
                h = cfg.image_size[1]
                w = cfg.image_size[0]
            resized_image = image.resize((w, h), Image.Resampling.LANCZOS)
            im = self.transforms_pil(resized_image)
        else:
            image = read_image(image_file)
            im = self.transforms_v2(image)

        crop = self.annos[image_name]
        crop = np.array(crop).reshape(-1, 4).astype(np.float32)
        return im, crop, im_width, im_height, image_file


if __name__ == '__main__':
    fcdb_testset = FCDBDataset(split='train')
    dataloader = DataLoader(fcdb_testset, batch_size=4, num_workers=1)
    for batch_idx, data in enumerate(dataloader):
        im, crop, im_width, im_height, image_file = data
        print(crop.reshape(-1, 4), im_width, im_height)
