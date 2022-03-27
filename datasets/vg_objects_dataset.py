#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Containing the building of VGO dataset.
"""
import json, os, pickle
import os.path as osp
import numpy as np
import pandas as pd
import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from random import random, randint
try:
    from .utils import imagenet_preprocess
except:
    from utils import imagenet_preprocess
import torchvision.transforms as T
import PIL

class object_item(object):
    """
    This is the object item used for object crop retrival
    object_id: The only index linking to object in VG dataset
    object_index: The only index linking to object in COCO dataset
    """
    def __init__(self, object_category=-1, num_objects=None,
                 boxes=None, object_id=None, object_index=None):
        if isinstance(object_category, torch.Tensor):
            object_category = object_category.item()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        self.category = object_category
        x, y, w, h = boxes
        self.x = float(x + w / 2.)
        self.y = float(y + h / 2.)
        self.width = float(w)
        self.height = float(h)
        self.object_id = object_id
        self.object_index = object_index


class Cropped_VG_Dataset(Dataset):
    """Dataset of cropped VG objects."""
    def __init__(self,
                 objects_csv,
                 objects_pickle,
                 mem_bank_path,
                 top10_crop_ids=None,
                 output_size=None,
                 error_handling="None",
                 retrieve_sampling="random",
                 candidate_num=100,
                 normalize_method='imagenet',
        ):
        """
        Return a tensor standardized and reshaped

        Args:
            objects_pickle (string): pickle file with all cropped objects.
            objects_csv: csv containing object information
            output_size: the new width and height of reshaping, it not set, output the original size
            error_handling: what to do with retrieval failure,
                choice 1 -- "None": return None
                choice 2 -- "Zero": return Zero Tensor of (3, ) + output_size specified
        """
        print("Loading objects.csv.")
        self.candidate_num = candidate_num
        self.objects_csv = objects_csv

        self.error_handling = error_handling
        self.output_size = output_size
        self.retrieve_sampling = retrieve_sampling
        self.random_flip = False

        image_size = (64, 64)
        transform = [T.Resize(image_size), T.ToTensor(), imagenet_preprocess()]
        self.transform = T.Compose(transform)

    def __len__(self):
        return len(self.objects.keys())

    def __getitem__(self, idx):
        """
        Args:
            idx: object idx as in the new indexing system of objects_df
        """
        key = str(idx)
        if self.objects_pickle:
            np_array = np.array(self.objects[key][0])
            np_array = PIL.Image.fromarray(np_array)
            tensor = self.transform(np_array)
            assert tensor.size(0) != 1
            if self.random_flip and random() > .5:
                tensor = torch.flip(tensor, dims=[2,])
            return tensor

    def retrieve(self, object_item, image_id=None, num_crop=5):
        """
        Given the [object_item] containing the basic information
        about the objects to retrive the corresponding object crops
        """

        object_id = object_item.object_id
        if not(self.objects.get(str(int(object_id)))):
            print('not searched!', str(int(object_id)))
            keys_list = list(self.objects.keys())
            random_index = np.random.randint(0, high=len(keys_list)-1)
            keys_this = int(keys_list[random_index])
            original_crops = self.__getitem2__[keys_this]
            selected_crops = []
            for mm in range(num_crop):
                selected_crops_this = self.__getitem2__[keys_this]
                # 1 * c * h * w
                selected_crops.append(selected_crops_this.unsqueeze(dim=0))
            selected_crops = torch.cat(selected_crops, dim=0)

        else:
            selected_ids = self.objects[str(int(object_id))][1]
            original_crop = int(selected_ids[0])

            selected_crop_list = [int(selected_ids[1]), int(selected_ids[2]),
                                  int(selected_ids[3]), int(selected_ids[4]), int(selected_ids[5])]

            keys_list = list(self.objects.keys())
            key2 = str(original_crop)
            if not(key2 in keys_list):
                random_index = np.random.randint(0, high=len(keys_list) - 1)
                original_crop = int(keys_list[random_index])
            original_crops = self.__getitem2__(original_crop)

            selected_crops = []
            for mm in range(len(selected_crop_list)):
                key1 = str(selected_crop_list[mm])
                selected_crop = selected_crop_list[mm]
                if not (key1 in keys_list):
                    random_index = np.random.randint(0, high=len(keys_list) - 1)
                    selected_crop = int(keys_list[random_index])
                selected_crops_this = self.__getitem2__(selected_crop)
                selected_crops.append(selected_crops_this.unsqueeze(dim=0))
            selected_crops = torch.cat(selected_crops, dim=0)

        return selected_crops, original_crops
