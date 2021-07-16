#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import albumentations as A
from .data_aug import *
from .bbox_util import *


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):

        transform = A.Compose([
        #A.RandomCrop(width=450, height=450),
        #A.HorizontalFlip(p=0.5),
        #A.RandomBrightnessContrast(p=0.2),

        #A.RandomSizedCrop(min_max_height=(200,200), height=360, width=640, p=0.5),
        #A.RandomSizedCropV2(height=512, width=512, scale=(0.08, 1.0), ratio=(0.75, 1.33333)),
        A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=512, width=512, p=1),
        A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5)
        ], 
        bbox_params=A.BboxParams(format='pascal_voc'))

        labels = target["labels"].reshape(-1,1)
        boxes = torch.cat((target["boxes"],labels),dim=1)

        transformed = transform(image=image, bboxes=np.array(boxes.tolist()))
        image = transformed['image'].copy()
        target["boxes"] = torch.as_tensor(transformed['bboxes'].copy(), dtype=torch.float32)[:,:-1]
        #target["labels"] = torch.as_tensor(transformed['class_labels'].copy(), dtype=torch.int64)

        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlips(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        #seq = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomShear()])
        #img_, bboxes_ = seq(image, np.array(target["boxes"].tolist()))
        #target["boxes"] = torch.as_tensor(bboxes_, dtype=torch.float32)
        #image = F.to_tensor(img_)

        image = F.to_tensor(image)
        return image, target

def get_transform(train):

    transforms = []
    
    if train:
        pass

    transforms.append(ToTensor())
    return Compose(transforms)