#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to sample many images from a model for evaluation.
"""


import argparse, json
import os
import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from scipy.misc import imsave, imresize

# from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.data.utils import split_graph_batch
from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, bool_flag
from sg2im.vis import draw_scene_graph
import json
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='sg2im-models/vg64.pt')
parser.add_argument('--checkpoint_list', default=None)
parser.add_argument('--model_mode', default='eval', choices=['train', 'eval'])

# Shared dataset options
parser.add_argument('--dataset', default='vg', choices=['coco', 'vg'])
parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
##parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--save_gt_imgs', default=False, type=bool_flag)
parser.add_argument('--save_graphs', default=False, type=bool_flag)
parser.add_argument('--use_gt_boxes', default=False, type=bool_flag)
parser.add_argument('--use_gt_masks', default=False, type=bool_flag)
parser.add_argument('--save_layout', default=True, type=bool_flag)

parser.add_argument('--output_dir', default='output')

# For VG
VG_DIR = os.path.expanduser('/home/ubuntu/xgxu/PasteGAN/data/visual_genome')
# parser.add_argument('--vg_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vg_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--vg_image_dir',
        default=os.path.join(VG_DIR, 'images'))

# For COCO
COCO_DIR = os.path.expanduser('/mnt/proj3/xgxu/PasteGAN-master/data/coco')
parser.add_argument('--coco_image_dir',
        default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--instances_json',
        default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--stuff_json',
        default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))


def build_coco_dset(args, checkpoint):
  checkpoint_args = checkpoint['args']
  print('include other: ', checkpoint_args.get('coco_include_other'))
  print(checkpoint_args['coco_stuff_only'], 'bbbb', checkpoint_args['min_objects_per_image'], checkpoint_args.get('coco_include_other', True))
  # checkpoint_args['coco_stuff_only'] = False
  dset_kwargs = {
    'image_dir': args.coco_image_dir,
    'instances_json': args.instances_json,
    'stuff_json': args.stuff_json,
    'stuff_only': checkpoint_args['coco_stuff_only'],
    'image_size': args.image_size,
    'mask_size': checkpoint_args['mask_size'],
    'max_samples': None,
    'min_object_size': checkpoint_args['min_object_size'],
    'min_objects_per_image': checkpoint_args['min_objects_per_image'],
    'instance_whitelist': checkpoint_args['instance_whitelist'],
    'stuff_whitelist': checkpoint_args['stuff_whitelist'],
    'include_other': checkpoint_args.get('coco_include_other', True),
  }
  # min_objects_per_image=3, max_objects_per_image=8,
  dset = CocoSceneGraphDataset(**dset_kwargs)
  return dset


def build_vg_dset(args, checkpoint):
  vocab = checkpoint['model_kwargs']['vocab']
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.vg_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': None,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
  }
  dset = VgSceneGraphDataset(**dset_kwargs)
  return dset


def build_loader(args, checkpoint):
  if args.dataset == 'coco':
    dset = build_coco_dset(args, checkpoint)
    collate_fn = coco_collate_fn
  elif args.dataset == 'vg':
    dset = build_vg_dset(args, checkpoint)
    collate_fn = vg_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': collate_fn,
  }
  loader = DataLoader(dset, **loader_kwargs)

  print(len(loader.dataset), 'aaaa')
  return loader


def build_model(args, checkpoint):
  kwargs = checkpoint['model_kwargs']
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  if args.model_mode == 'eval':
    model.eval()
  elif args.model_mode == 'train':
    model.train()
  model.image_size = args.image_size
  model.cuda()
  return model


def makedir(base, name, flag=True):
  dir_name = None
  if flag:
    dir_name = os.path.join(base, name)
    if not os.path.isdir(dir_name):
      os.makedirs(dir_name)
  return dir_name

####
def run_model(args, checkpoint, output_dir, loader=None):
  vocab = checkpoint['model_kwargs']['vocab']
  model = build_model(args, checkpoint)
  if loader is None:
    loader = build_loader(args, checkpoint)

  args.coco_image_dir = os.path.join(COCO_DIR, 'images/val2017')
  args.instances_json = os.path.join(COCO_DIR, 'annotations/instances_val2017.json')
  args.stuff_json = os.path.join(COCO_DIR, 'annotations/stuff_val2017.json')
  loader_val = build_loader(args, checkpoint)

  import torch.nn.functional as F

  img_idx = 0
  object_rgb_list_all = []
  object_seg_list_all = []
  object_mask_list_all = []
  obj_pred_all = []
  obj_id_all = []
  obj_id_global = 0

  image_to_obj_index = {}
  for batch in loader:
    masks = None
    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, img_id, seg_map = [x.cuda() for x in batch]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    mean_value = torch.Tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std_value = torch.Tensor(IMAGENET_STD).view(1, 3, 1, 1)
    batch_size = imgs.shape[0]
    height = imgs.shape[2]
    width = imgs.shape[3]
    mean_value = mean_value.repeat(batch_size, 1, height, width).cuda()
    std_value = std_value.repeat(batch_size, 1, height, width).cuda()
    imgs_unnorm = imgs * std_value
    imgs_unnorm = imgs_unnorm + mean_value

    boxes_gt = None
    masks_gt = None
    if args.use_gt_boxes:
      boxes_gt = boxes
    if args.use_gt_masks:
      masks_gt = masks

    model_out = model(objs, triples, obj_to_img,
                      boxes_gt=boxes_gt, masks_gt=masks_gt, inference=1)
    imgs_pred, boxes_pred, masks_pred, _, obj_pred = model_out
    obj_pred = obj_pred.detach().cpu().numpy()

    obj_pred_list = []
    obj_id_list = []
    object_rgb_list = []
    object_seg_list = []
    object_mask_list = []
    for mm in range(boxes.shape[0]):
      bbox_this = boxes[mm]

      x0, y0, x1, y1 = bbox_this
      x0 = int(x0 * width)
      x1 = int(x1 * width)
      y0 = int(y0 * height)
      y1 = int(y1 * height)
      x0 = max(0, x0)
      y0 = max(0, y0)
      x1 = min(width - 1, x1)
      y1 = min(height - 1, y1)

      if (x1 - x0) < 2:
        x0 = x1 - 2
      if (y1 - y0) < 2:
        y0 = y1 - 2

      img_index = obj_to_img[mm]
      image_this = imgs_unnorm.clone()[img_index][:, y0:y1, x0:x1]
      image_this = image_this.permute(1, 2, 0)
      image_this = image_this * 255.0
      image_this = image_this.cpu().numpy().astype(np.uint8)
      image_this = cv2.resize(image_this, (64, 64))

      seg_this = seg_map.clone()[img_index][y0:y1, x0:x1]
      seg_this = seg_this.unsqueeze(dim=0).unsqueeze(dim=0)
      seg_this = F.interpolate(seg_this.float(), size=(64, 64), mode='nearest')
      seg_this = seg_this.squeeze(0).squeeze(0).cpu().numpy()

      mask_this = masks[mm].clone()
      mask_this = mask_this.unsqueeze(dim=0).unsqueeze(dim=0)
      mask_this = F.interpolate(mask_this.float(), size=(64, 64), mode='nearest')
      mask_this = mask_this.squeeze(0).squeeze(0).cpu().numpy()

      image_id = img_id[img_index].item()
      obj_id_this = obj_id_global
      if image_to_obj_index.get(image_id):
        dic_this = image_to_obj_index[image_id]
        keys_list = list(dic_this.keys())
        length = len(keys_list)
        image_to_obj_index[image_id][length] = obj_id_this
      else:
        image_to_obj_index[image_id] = {}
        image_to_obj_index[image_id][0] = obj_id_this

      object_rgb_list.append(image_this)
      object_seg_list.append(seg_this)
      object_mask_list.append(mask_this)
      obj_pred_list.append(obj_pred[mm])
      obj_id_list.append(obj_id_this)
      obj_id_global += 1

    object_rgb_list = np.array(object_rgb_list)
    object_seg_list = np.array(object_seg_list)
    object_mask_list = np.array(object_mask_list)
    obj_pred_list = np.array(obj_pred_list)
    obj_id_list = np.array(obj_id_list)
    print(object_rgb_list.shape, obj_pred_list.shape, obj_id_list.shape, len(np.unique(obj_id_list)), np.sum(obj_id_list==-1))
    object_rgb_list_all.append(object_rgb_list)
    object_seg_list_all.append(object_seg_list)
    object_mask_list_all.append(object_mask_list)
    obj_pred_all.append(obj_pred_list)
    obj_id_all.append(obj_id_list)
    img_idx += 1

  object_rgb_list_all = np.concatenate(object_rgb_list_all, axis=0)
  object_seg_list_all = np.concatenate(object_seg_list_all, axis=0)
  object_mask_list_all = np.concatenate(object_mask_list_all, axis=0)
  obj_id_all = np.concatenate(obj_id_all, axis=0)
  obj_pred_all = np.concatenate(obj_pred_all, axis=0)
  # N * L
  print(object_rgb_list_all.shape)
  print(object_seg_list_all.shape)
  print(object_mask_list_all.shape)
  print(obj_pred_all.shape)
  print(obj_id_all.shape)

  image_content = {}
  obj_pred_all_t = obj_pred_all.transpose()
  norm = np.linalg.norm(obj_pred_all, axis=1)
  norm = np.reshape(norm, (1, norm.shape[0]))
  for mm in range(len(obj_id_all)):
    if obj_id_all[mm] == -1:
      continue
    obj_feature = obj_pred_all[mm]
    obj_rgb_image = object_rgb_list_all[mm]
    obj_seg_image = object_seg_list_all[mm]
    obj_mask_image = object_mask_list_all[mm]
    obj_id = obj_id_all[mm]
    image_content[str(obj_id)] = [obj_rgb_image]

    obj_feature = np.expand_dims(obj_feature, axis=0)
    similarity = obj_feature.dot(obj_pred_all_t)

    similarity = np.divide(similarity, norm)
    similarity = similarity[0]
    max_list = np.argsort(-similarity)

    choose_list = []
    for nn in range(10):
      number = max_list[nn]
      object_id = obj_id_all[number]
      choose_list.append(object_id)
    if not (choose_list[0] == obj_id):
      choose_list[1:] = choose_list[0:-1]
      choose_list[0] = obj_id
    image_content[str(obj_id)].append(choose_list)
    image_content[str(obj_id)].append(obj_seg_image)
    image_content[str(obj_id)].append(obj_mask_image)

  output = open('processed_objects_np16_32_coco3.pickle', 'wb')
  pickle.dump(image_content, output)
  output.close()

  output = open('object_index_mapping3.pickle', 'wb')
  pickle.dump(image_to_obj_index, output)
  output.close()

  #############################################################################
  object_rgb_list_all_val = []
  object_seg_list_all_val = []
  object_mask_list_all_val = []
  obj_pred_all_val = []
  obj_id_all_val = []
  obj_id_global_val = obj_id_global
  image_to_obj_index_val = {}
  for batch in loader_val:
    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, img_id, seg_map = [x.cuda() for x in batch]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    mean_value = torch.Tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std_value = torch.Tensor(IMAGENET_STD).view(1, 3, 1, 1)
    batch_size = imgs.shape[0]
    height = imgs.shape[2]
    width = imgs.shape[3]
    mean_value = mean_value.repeat(batch_size, 1, height, width).cuda()
    std_value = std_value.repeat(batch_size, 1, height, width).cuda()
    imgs_unnorm = imgs * std_value
    imgs_unnorm = imgs_unnorm + mean_value

    boxes_gt = None
    masks_gt = None
    if args.use_gt_boxes:
      boxes_gt = boxes
    if args.use_gt_masks:
      masks_gt = masks

    model_out = model(objs, triples, obj_to_img,
                      boxes_gt=boxes_gt, masks_gt=masks_gt, inference=1)
    imgs_pred, boxes_pred, masks_pred, _, obj_pred = model_out
    obj_pred = obj_pred.detach().cpu().numpy()

    obj_pred_list = []
    obj_id_list = []
    object_rgb_list = []
    object_seg_list = []
    object_mask_list = []
    for mm in range(boxes.shape[0]):
      bbox_this = boxes[mm]

      x0, y0, x1, y1 = bbox_this
      x0 = int(x0 * width)
      x1 = int(x1 * width)
      y0 = int(y0 * height)
      y1 = int(y1 * height)
      x0 = max(0, x0)
      y0 = max(0, y0)
      x1 = min(width - 1, x1)
      y1 = min(height - 1, y1)

      if (x1 - x0) < 2:
        x0 = x1 - 2
      if (y1 - y0) < 2:
        y0 = y1 - 2

      img_index = obj_to_img[mm]
      image_this = imgs_unnorm.clone()[img_index][:, y0:y1, x0:x1]
      image_this = image_this.permute(1, 2, 0)
      image_this = image_this * 255.0
      image_this = image_this.cpu().numpy().astype(np.uint8)
      image_this = cv2.resize(image_this, (64, 64))

      seg_this = seg_map.clone()[img_index][y0:y1, x0:x1]
      seg_this = seg_this.unsqueeze(dim=0).unsqueeze(dim=0)
      seg_this = F.interpolate(seg_this.float(), size=(64, 64), mode='nearest')
      seg_this = seg_this.squeeze(0).squeeze(0).cpu().numpy()

      mask_this = masks[mm].clone()
      mask_this = mask_this.unsqueeze(dim=0).unsqueeze(dim=0)
      mask_this = F.interpolate(mask_this.float(), size=(64, 64), mode='nearest')
      mask_this = mask_this.squeeze(0).squeeze(0).cpu().numpy()

      image_id = img_id[img_index].item()
      obj_id_this = obj_id_global_val
      if image_to_obj_index_val.get(image_id):
        dic_this = image_to_obj_index_val[image_id]
        keys_list = list(dic_this.keys())
        length = len(keys_list)
        image_to_obj_index_val[image_id][length] = obj_id_this
      else:
        image_to_obj_index_val[image_id] = {}
        image_to_obj_index_val[image_id][0] = obj_id_this

      object_rgb_list.append(image_this)
      object_seg_list.append(seg_this)
      object_mask_list.append(mask_this)
      obj_pred_list.append(obj_pred[mm])
      obj_id_list.append(obj_id_this)
      obj_id_global_val += 1

    object_rgb_list = np.array(object_rgb_list)
    object_seg_list = np.array(object_seg_list)
    object_mask_list = np.array(object_mask_list)
    obj_pred_list = np.array(obj_pred_list)
    obj_id_list = np.array(obj_id_list)
    print(object_rgb_list.shape, obj_pred_list.shape, obj_id_list.shape, len(np.unique(obj_id_list)),
          np.sum(obj_id_list == -1))
    object_rgb_list_all_val.append(object_rgb_list)
    object_seg_list_all_val.append(object_seg_list)
    object_mask_list_all_val.append(object_mask_list)
    obj_pred_all_val.append(obj_pred_list)
    obj_id_all_val.append(obj_id_list)
    img_idx += 1

  object_rgb_list_all_val = np.concatenate(object_rgb_list_all_val, axis=0)
  object_seg_list_all_val = np.concatenate(object_seg_list_all_val, axis=0)
  object_mask_list_all_val = np.concatenate(object_mask_list_all_val, axis=0)
  obj_id_all_val = np.concatenate(obj_id_all_val, axis=0)
  obj_pred_all_val = np.concatenate(obj_pred_all_val, axis=0)
  print(object_rgb_list_all_val.shape)
  print(object_seg_list_all_val.shape)
  print(object_mask_list_all_val.shape)
  print(obj_pred_all_val.shape)
  print(obj_id_all_val.shape)

  obj_pred_all = np.concatenate([obj_pred_all, obj_pred_all_val], axis=0)
  object_rgb_list_all = np.concatenate([object_rgb_list_all, object_rgb_list_all_val], axis=0)
  object_seg_list_all = np.concatenate([object_seg_list_all, object_seg_list_all_val], axis=0)
  object_mask_list_all = np.concatenate([object_mask_list_all, object_mask_list_all_val], axis=0)
  obj_id_all = np.concatenate([obj_id_all, obj_id_all_val], axis=0)
  print(object_rgb_list_all.shape)
  print(object_seg_list_all.shape)
  print(object_mask_list_all.shape)
  print(obj_pred_all.shape)
  print(obj_id_all.shape)

  image_content = {}
  obj_pred_all_t = obj_pred_all.transpose()
  norm = np.linalg.norm(obj_pred_all, axis=1)
  norm = np.reshape(norm, (1, norm.shape[0]))
  for mm in range(len(obj_id_all)):
    if obj_id_all[mm] == -1:
      continue
    obj_feature = obj_pred_all[mm]
    obj_rgb_image = object_rgb_list_all[mm]
    obj_seg_image = object_seg_list_all[mm]
    obj_mask_image = object_mask_list_all[mm]
    obj_id = obj_id_all[mm]
    image_content[str(obj_id)] = [obj_rgb_image]

    obj_feature = np.expand_dims(obj_feature, axis=0)
    similarity = obj_feature.dot(obj_pred_all_t)

    similarity = np.divide(similarity, norm)
    similarity = similarity[0]
    max_list = np.argsort(-similarity)

    choose_list = []
    for nn in range(10):
      number = max_list[nn]
      object_id = obj_id_all[number]
      choose_list.append(object_id)

    if not (choose_list[0] == obj_id):
      choose_list[1:] = choose_list[0:-1]
      choose_list[0] = obj_id
    image_content[str(obj_id)].append(choose_list)
    image_content[str(obj_id)].append(obj_seg_image)
    image_content[str(obj_id)].append(obj_mask_image)

  output = open('processed_objects_np16_32_coco_val3.pickle', 'wb')
  pickle.dump(image_content, output)
  output.close()

  output = open('object_index_mapping_val3.pickle', 'wb')
  pickle.dump(image_to_obj_index_val, output)
  output.close()



def main(args):
  got_checkpoint = args.checkpoint is not None
  got_checkpoint_list = args.checkpoint_list is not None
  if got_checkpoint == got_checkpoint_list:
    raise ValueError('Must specify exactly one of --checkpoint and --checkpoint_list')

  if got_checkpoint:
    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint, args.output_dir)
  elif got_checkpoint_list:
    # For efficiency, use the same loader for all checkpoints
    loader = None
    with open(args.checkpoint_list, 'r') as f:
      checkpoint_list = [line.strip() for line in f]
    for i, path in enumerate(checkpoint_list):
      if os.path.isfile(path):
        print('Loading model from ', path)
        checkpoint = torch.load(path)
        if loader is None:
          loader = build_loader(args, checkpoint)
        output_dir = os.path.join(args.output_dir, 'result%03d' % (i + 1))
        run_model(args, checkpoint, output_dir, loader)
      elif os.path.isdir(path):
        # Look for snapshots in this dir
        for fn in sorted(os.listdir(path)):
          if 'snapshot' not in fn:
            continue
          checkpoint_path = os.path.join(path, fn)
          print('Loading model from ', checkpoint_path)
          checkpoint = torch.load(checkpoint_path)
          if loader is None:
            loader = build_loader(args, checkpoint)

          # Snapshots have names like "snapshot_00100K.pt'; we want to
          # extract the "00100K" part
          snapshot_name = os.path.splitext(fn)[0].split('_')[1]
          output_dir = 'result%03d_%s' % (i, snapshot_name)
          output_dir = os.path.join(args.output_dir, output_dir)

          run_model(args, checkpoint, output_dir, loader)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)


