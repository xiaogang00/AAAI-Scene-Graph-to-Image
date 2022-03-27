import functools
import os
import json
import math
from collections import defaultdict
import random
import time
import pyprind
import glog as log
from shutil import copyfile

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from datasets import imagenet_deprocess_batch
import datasets
import models
import models.perceptual
from utils.losses import get_gan_losses
from utils import timeit, LossManager
from options.opts import args, options
from utils.logger import Logger
from utils import tensor2im
from utils.training_utils4 import add_loss, check_model, calculate_model_losses, calculate_model_losses_seg_img
from models.utils import DiscriminatorDataParallel, GeneratorDataParallel
from utils.training_utils4_search2 import visualize_sample, unpack_batch, visualize_sample2, visualize_sample3_img
from utils.evaluate import evaluate
from utils.bilinear import crop_bbox_batch, uncrop_bbox
torch.backends.cudnn.benchmark = True
import cv2
from utils.canvas import make_canvas_baseline


from utils.visualization import draw_scene_graph2, draw_scene_graph

from models.transformer.Models import Encoder_patch

def weights_init(m, init_type='normal', gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            # m.weight.data *= scale
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)



import fid_score
from torch.autograd import Variable

def toVariable(obj, requires_grad=False):
    if isinstance(obj, Variable):
        y = Variable(obj.data, requires_grad=requires_grad)
    elif type(obj) == np.ndarray:
        y = torch.from_numpy(obj.astype(np.float32))
        y = Variable(y, requires_grad=requires_grad)
    elif isinstance(obj, torch.FloatTensor) or isinstance(obj,torch.cuda.FloatTensor):
        y = Variable(obj, requires_grad=requires_grad)
    elif type(obj) == list or type(obj) == tuple:
        y = []
        for item in obj:
            y += [toVariable(item, requires_grad=requires_grad)]
    else:
        assert 0, 'type: %s is not supported yet' % type(obj)
    return y


def untransformVariable(vggImageVariable):
    mean = torch.Tensor((0.5, 0.5, 0.5))
    stdv = torch.Tensor((0.5, 0.5, 0.5))
    mean = toVariable(mean).cuda()
    stdv = toVariable(stdv).cuda()
    vggImageVariable *= stdv.view(1, 3, 1, 1).expand_as(vggImageVariable)
    vggImageVariable += mean.view(1, 3, 1, 1).expand_as(vggImageVariable)
    vggImageVariable[vggImageVariable.data > 1.] = 1.
    vggImageVariable[vggImageVariable.data < 0.] = 0.
    return vggImageVariable



def main():
    global args, options
    print(args)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    log.info("Building loader...")
    vocab, val_loader = datasets.build_loaders5_val(options["data"])

    log.info("Building Generative Model...")
    model, model_kwargs = models.build_model(
        options["generator"],
        vocab,
        image_size=options["data"]["image_size"],
        checkpoint_start_from=args.checkpoint_start_from)
    model.type(float_dtype)

    patch_discriminator = Encoder_patch()
    patch_discriminator.type(float_dtype)
    ### the path of SCSM
    restore_path = 'output/crop_selection_module/checkpoint_with_model.pt'
    log.info('Restoring from checkpoint: {}'.format(restore_path))
    checkpoint = torch.load(restore_path)
    patch_discriminator.load_state_dict(checkpoint['d_seg2_state'])
    patch_discriminator.eval()

    ### the path of PSGIM
    restore_path = 'output/generator/checkpoint_with_model.pt'
    log.info('Restoring from checkpoint: {}'.format(restore_path))
    checkpoint = torch.load(restore_path)
    model.crop_encoder.load_state_dict(checkpoint['model_state_crop_encoder'])
    model.obj_embeddings.load_state_dict(checkpoint['model_state_obj_embeddings'])
    model.pred_embeddings.load_state_dict(checkpoint['model_state_pred_embeddings'])
    model.gconv.load_state_dict(checkpoint['model_state_gconv'])
    model.gconv_net.load_state_dict(checkpoint['model_state_gconv_net'])
    model.box_net.load_state_dict(checkpoint['model_state_box_net'])
    model.img_decoder.load_state_dict(checkpoint['model_state_img_decoder'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GeneratorDataParallel(model.to(device))
    patch_discriminator.to(device)

    isBest = True
    pred_crops = None
    others = None
    epoch = 0
    model.eval()

    result_dir = './results/ours_coco'
    if not (os.path.exists(result_dir)):
        os.mkdir(result_dir)
    count = 0

    for iter, batch in enumerate(pyprind.prog_bar(val_loader, title="[Epoch {}/{}]".format(epoch, args.epochs), width=50)):
        ######### unpack the data #########
        batch = unpack_batch(batch, options)
        (imgs, canvases_sel, canvases_ori, seg_map, label_map,
         objs, boxes, selected_crops_all,
         original_crops, triples, predicates,
         obj_to_img, triple_to_img,
         scatter_size_obj, scatter_size_triple) = batch

        image_id_list = []
        for mm in range(obj_to_img.shape[0]):
            if not (obj_to_img[mm] in image_id_list):
                image_id_list.append(obj_to_img[mm])

        selected_crops_all = selected_crops_all.to(device)
        boxes = boxes.to(device)
        retrieved_patchs = []
        for mm in range(len(image_id_list)):
            index_this = (obj_to_img == image_id_list[mm])
            selected_crops_this = selected_crops_all[index_this]
            # b *k * c* h* w
            bbox_this = boxes[index_this]
            batch_size = selected_crops_this.shape[0]
            if batch_size <= 2:
                retrieved_patchs.append(selected_crops_this[:, 0, 0:3, :, :])
                continue
            selected_crops_this_final = selected_crops_this[-1:, 0, 0:3, :, :]

            objs_this = objs[index_this]
            obj_mask = objs_this.nonzero().view(-1)
            selected_crops_this = selected_crops_this[obj_mask]
            bbox_this = bbox_this[obj_mask]
            retrieved_patchs.append(selected_crops_this[0:1, 0, 0:3, :, :])

            previous_patch = selected_crops_this[0:1, 0, 0:3, :, :]
            previous_bbox = bbox_this[0:1]
            for kkk in range(1, selected_crops_this.shape[0], 1):
                patch_this_list = selected_crops_this[kkk]
                bbox_this_one = bbox_this[kkk:kkk + 1]
                # kk * 3 * h *w
                value_list = []
                for kk in range(patch_this_list.shape[0]):
                    patch_this = patch_this_list[kk:kk + 1, 0:3, :, :]
                    input_this = torch.cat([previous_patch, patch_this], dim=0)
                    input_this_bbox = torch.cat([previous_bbox, bbox_this_one], dim=0)
                    with torch.no_grad():
                        score = patch_discriminator(input_this, input_this_bbox)
                    value_list.append(score)
                value_list = torch.cat(value_list, dim=0).view(5, )
                index = torch.argmax(value_list)
                patch_this = patch_this_list[index:index + 1, 0:3, :, :]
                retrieved_patchs.append(patch_this)
                previous_patch = torch.cat([previous_patch, patch_this], dim=0)
                previous_bbox = torch.cat([previous_bbox, bbox_this_one], dim=0)
            retrieved_patchs.append(selected_crops_this_final)
        selected_crops = torch.cat(retrieved_patchs, dim=0)
        for kkkk in range(selected_crops.shape[0]):
            selected_crops[kkkk].to(original_crops[kkkk].device)
            boxes[kkkk].to(original_crops[kkkk].device)
            selected_crops_all[kkkk].to(original_crops[kkkk].device)

        ###################################
        with timeit('forward', args.timing):
            with torch.no_grad():
                model_boxes = boxes
                model_out = model(objs, triples, obj_to_img, triple_to_img,
                                  selected_crops=selected_crops,
                                  original_crops=original_crops,
                                  scatter_size_obj=scatter_size_obj,
                                  scatter_size_triple=scatter_size_triple)
                ## if use the GT bounding box
                '''
                model_out = model(objs, triples, obj_to_img, triple_to_img,
                                  boxes_gt=model_boxes,
                                  selected_crops=selected_crops,
                                  original_crops=original_crops,
                                  scatter_size_obj=scatter_size_obj,
                                  scatter_size_triple=scatter_size_triple)
                '''
                boxes_pred, seg_pred, seg_rcst, others = model_out
                graph_feature = others['graph_feature']

        im = seg_pred[0].permute(1, 2, 0)
        im = (im + 1) * 255.0 / 2
        im = im[:, :, [2, 1, 0]]
        im = im.detach().cpu().numpy()
        height = im.shape[0]
        width = im.shape[1]

        im2 = imgs[0].permute(1, 2, 0)
        im2 = (im2 + 1) * 255.0 / 2
        im2 = im2.detach().cpu().numpy()
        im2 = im2[:, :, [2, 1, 0]]
        im = np.concatenate([im, im2], axis=1)

        sg_array = draw_scene_graph(objs[obj_to_img == 0],
                                    triples[triple_to_img == 0], vocab=vocab)
        sg_array = rgba2rgb(sg_array)
        print(seg_pred.shape, objs.shape, sg_array.shape, selected_crops.shape, original_crops.shape, type(sg_array))
        sg_array = cv2.resize(sg_array, (width, height))
        im = np.concatenate([im, sg_array], axis=1)
        output_name = os.path.join(result_dir, '%04d.jpg' % count)
        cv2.imwrite(output_name, im)

        count += 1


def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


if __name__ == '__main__':
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True
    seed = 1234
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)

    ### the generation from scene graph to image with predicted bounding box
    main()
