#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Containing the building of COCO Objects Dataset.
"""

import json, os
import numpy as np
import pandas as pd
import h5py
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import PIL
from random import random
import torch.nn.functional as F
try:
    from .utils import imagenet_preprocess
except:
    from utils import imagenet_preprocess
import torchvision.transforms as T

try:
    from .vg_objects_dataset import object_item, Cropped_VG_Dataset
except:
    from vg_objects_dataset import object_item, Cropped_VG_Dataset


from .augmentation import AllAugmentationTransform_image
class Cropped_COCO_Dataset(Cropped_VG_Dataset):
    def __init__(self,
                 objects_csv,
                 objects_pickle,
                 mem_bank_path,
                 top10_crop_ids=None,
                 output_size=None,
                 error_handling="None",
                 retrieve_sampling="ratio",
                 candidate_num=100,
                 normalize_method='imagenet',
        ):
        # Inherit the Cropped_VG_Dataset
        super(Cropped_COCO_Dataset, self).__init__(objects_csv,
                                                   objects_pickle,
                                                   mem_bank_path,
                                                   top10_crop_ids,
                                                   output_size,
                                                   error_handling,
                                                   retrieve_sampling,
                                                   candidate_num,
                                                   normalize_method,)
        self.objects_pickle = objects_pickle

        if self.objects_pickle is not None:
            with open(self.objects_pickle, 'rb') as f:
                self.objects = pickle.load(f)

        # self.transform = torch.Tensor
        image_size = (64, 64)
        transform = [T.Resize(image_size), T.ToTensor(), imagenet_preprocess()]
        transform2 = [T.Resize(image_size), T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                      T.ToTensor(), imagenet_preprocess()]

        self.transform = T.Compose(transform)
        self.transform2 = T.Compose(transform2)

        flip_param = {}
        flip_param['horizontal_flip'] = False
        flip_param['time_flip'] = False
        aug_param = {}

        if 'shift_param' not in aug_param:
            shift_param = {'bias': [20, 20]}
            aug_param['shift_param'] = shift_param

        self.augmentation = AllAugmentationTransform_image(resize_param=None,
                                                           rotation_param=None,
                                                           flip_param=flip_param,
                                                           crop_param=None,
                                                           trans_param=None,
                                                           shift_param=aug_param['shift_param'])

    def __getitem__(self, idx):
        """
        Args:
            idx: object idx as in the new indexing system of objects_df
        """
        key = str(idx)
        if self.objects_pickle:
            np_array = np.array(self.objects[key][0]).astype(np.float)
            seg_map = self.objects[key][2]
            seg_map = np.expand_dims(seg_map, axis=-1).astype(np.float)
            mask_map = self.objects[key][3]
            mask_map = np.expand_dims(mask_map, axis=-1).astype(np.float)

            all_array = np.concatenate([np_array, seg_map, mask_map], axis=2)
            all_array = np.expand_dims(all_array, axis=0)
            all_array = self.augmentation(all_array)
            all_array = np.squeeze(all_array, axis=0)
            np_array = all_array[:, :, 0:3]
            seg_map = all_array[:, :, 3]
            mask_map = all_array[:, :, 4]

            np_array = PIL.Image.fromarray(np_array.astype(np.uint8))
            tensor = self.transform(np_array)

            label_nc = 184
            seg_map = torch.Tensor(seg_map).long()
            seg_map[seg_map == 255] = label_nc - 1

            seg_map_clone = seg_map.clone().unsqueeze(dim=0).unsqueeze(dim=0).float()
            seg_map_clone = F.interpolate(seg_map_clone, size=(64, 64), mode='nearest')
            seg_map_clone = seg_map_clone.long()
            size = seg_map_clone.size()
            oneHot_size = (size[0], label_nc, size[2], size[3])
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, seg_map_clone.data.long(), 1.0)
            input_label = input_label[0]

            mask_map = torch.Tensor(mask_map).long()
            mask_map_clone = mask_map.clone().unsqueeze(dim=0).unsqueeze(dim=0).float()
            mask_map_clone = F.interpolate(mask_map_clone, size=(64, 64), mode='nearest')
            mask_map_clone = mask_map_clone.float()
            mask_map_clone = mask_map_clone[0]

            tensor = torch.cat([tensor, input_label, mask_map_clone], dim=0)
            assert tensor.size(0) != 1
            if self.random_flip and random() > .5:
                tensor = torch.flip(tensor, dims=[2,])
            return tensor

    def __getitem2__(self, idx):
        """
        Args:
            idx: object idx as in the new indexing system of objects_df
        """
        key = str(idx)
        if self.objects_pickle:
            np_array = np.array(self.objects[key][0])
            np_array = PIL.Image.fromarray(np_array)
            tensor = self.transform(np_array)

            label_nc = 184
            seg_map = self.objects[key][2]
            seg_map = torch.Tensor(seg_map).long()
            seg_map[seg_map == 255] = label_nc - 1

            seg_map_clone = seg_map.clone().unsqueeze(dim=0).unsqueeze(dim=0).long()
            # 1 * 1 * H * W
            size = seg_map_clone.size()
            oneHot_size = (size[0], label_nc, size[2], size[3])
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, seg_map_clone.data.long(), 1.0)
            input_label = input_label[0]

            mask_map = self.objects[key][3]
            mask_map = torch.Tensor(mask_map).long()
            mask_map_clone = mask_map.clone().unsqueeze(dim=0).unsqueeze(dim=0).float()
            mask_map_clone = F.interpolate(mask_map_clone, size=(64, 64), mode='nearest')
            mask_map_clone = mask_map_clone.float()
            mask_map_clone = mask_map_clone[0]

            tensor = torch.cat([tensor, input_label, mask_map_clone], dim=0)

            assert tensor.size(0) != 1
            if self.random_flip and random() > .5:
                tensor = torch.flip(tensor, dims=[2,])
            return tensor

    def get_mask(self, idx):
        """
        Args:
            idx: object idx as in the new indexing system of objects_df
        """
        assert not self.features
        np_array = self.masks[idx].astype(np.int32)
        tensor = self.transform(np_array)
        return tensor
