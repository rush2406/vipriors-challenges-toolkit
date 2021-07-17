#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A


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

    def __call__(self, image, target,mode):

        img_size_training = 448

        if mode=='train':

            transform = A.Compose(
        [   A.OneOf([
                A.HueSaturationValue(hue_shift_limit = 0.2, 
                                     sat_shift_limit = 0.2,
                                     val_shift_limit = 0.2,
                                     p = 0.3), 
            
                A.RandomBrightnessContrast(brightness_limit = 0.2,                                             
                                           contrast_limit = 0.2,
                                           p = 0.3),
                # RGB shift normally expects not-normalized images, so make sure to normalize the RGB shift!
                A.RGBShift(r_shift_limit = 20/255, 
                           g_shift_limit = 20/255, 
                           b_shift_limit = 10/255,
                           p = 0.3)
            ], 
            p = 0.2),
         
            A.OneOf([
                A.RandomGamma(gamma_limit = (80, 120),
                              p = 0.3),
                A.Blur(p = 0.6),
                A.GaussNoise(var_limit = (0.01, 0.05), mean = 0, p = 0.05),
                A.ToGray(p = 0.05)
                ],
                p = 0.1),

            A.OneOf([
                A.HorizontalFlip(p = 1), 
                A.VerticalFlip(p = 1),  
                A.Transpose(p = 1),                
                A.RandomRotate90(p = 1)
                ], 
                p = 0.7), 
         
            A.RandomSizedBBoxSafeCrop(img_size_training, 
                                      img_size_training, 
                                      p = 0.05),         
            A.Resize(height = img_size_training, 
                     width = img_size_training, 
                     p = 1),
         
            A.Cutout(num_holes = random.randint(1, 6),
                     max_h_size = 32, 
                     max_w_size = 32,
                     fill_value = 0, 
                     p = 0.15),
        ], p = 1.0,bbox_params=A.BboxParams(format='pascal_voc'))

        else:
            transform = A.Compose([
        A.Resize(height = img_size_training,
                 width = img_size_training,
                 p = 1),
        ], bbox_params = A.BboxParams(format='pascal_voc'))

        labels = target["labels"].reshape(-1,1)
        boxes = torch.cat((target["boxes"],labels),dim=1)
        transformed = transform(image=image, bboxes=np.array(boxes.tolist()))
        image = transformed['image'].copy()
        target["boxes"] = torch.as_tensor(transformed['bboxes'].copy(), dtype=torch.float32)[:,:-1]

        for t in self.transforms:
            image, target = t(image, target)

        #After converting to tensor, add mixup
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

        image = F.to_tensor(image.copy())
        return image, target

def get_transform(train):

    transforms = []
    
    #if train:
        #pass

    transforms.append(ToTensor())
    return Compose(transforms)