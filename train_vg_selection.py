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
from utils.training_utils4_search import add_loss, check_model, calculate_model_losses, calculate_model_losses_seg_img
from models.utils import DiscriminatorDataParallel, GeneratorDataParallel
from utils.training_utils4_search import visualize_sample, unpack_batch3, visualize_sample2, visualize_sample3_img_vg
from utils.evaluate import evaluate
from utils.bilinear import crop_bbox_batch, uncrop_bbox
torch.backends.cudnn.benchmark = True
import cv2
from utils.canvas import make_canvas_baseline


from models.transformer.Models import Encoder_patch

def info_nce_loss(features, batch_size, n_views, temperature):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0]).to(features.device)

    logits = logits / temperature
    return logits, labels

def weights_init(m, init_type='kaiming', gain=0.02):
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

    if isinstance(m, nn.Conv2d):
        nn.init.normal(m.weight, mean=0.0, std=gain)
        nn.init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal(m.weight, mean=0.0, std=gain)
        # nn.init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal(m.weight, mean=1.0, std=gain)
        nn.init.constant(m.bias, 0)


def main():
    global args, options
    print(args)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    log.info("Building loader...")
    vocab, train_loader = datasets.build_loaders(options["data"])

    patch_discriminator = Encoder_patch()
    patch_discriminator.type(float_dtype)
    # patch_discriminator.apply(weights_init)
    patch_discriminator.train()
    ## patch_discriminator is SCSM

    optimizer_d_seg2 = torch.optim.Adam(
        filter(lambda x: x.requires_grad, patch_discriminator.parameters()),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999), )
    restore_path = None
    if args.resume is not None:
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(
            options["logs"]["output_dir"], args.resume, restore_path)

    if restore_path is not None and os.path.isfile(restore_path):
        print('loading *******************')
        log.info('Restoring from checkpoint: {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        patch_discriminator.load_state_dict(checkpoint['d_seg2_state'])
        optimizer_d_seg2.load_state_dict(checkpoint['d_seg2_optim_state'])

        t = checkpoint['counters']['t'] + 1
        start_epoch = checkpoint['counters']['epoch'] + 1
        log_path = os.path.join(options["logs"]["output_dir"], args.resume,)
        lr = checkpoint.get('learning_rate', args.learning_rate)
        best_inception = checkpoint["counters"].get("best_inception", (0., 0.))
        options = checkpoint.get("options", options)

        del checkpoint
        checkpoint = {
            'args': args.__dict__,
            'options': options,
            'vocab': vocab,
            'train_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'train_inception': [],
            'lr': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'val_inception': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
                'best_inception': None,
            },
            'model_best_state': None, 'optim_state': None,
            'model_state_crop_encoder': None,
            'model_state_obj_embeddings': None,
            'model_state_pred_embeddings': None,
            'model_state_gconv': None, 'model_state_gconv_net': None,
            'model_state_img_decoder': None,
            'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
            'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
            'd_seg_state': None, 'd_seg_best_state': None, 'd_seg_optim_state': None,
            'd_seg2_state': None, 'd_seg2_best_state': None, 'd_seg2_optim_state': None,
            'd_seg3_state': None, 'd_seg3_best_state': None, 'd_seg3_optim_state': None,
            'd_seg_state128': None, 'd_seg_best_state128': None, 'd_seg_optim_state128': None,
            'd_seg_state64': None, 'd_seg_best_state64': None, 'd_seg_optim_state64': None,
        }

    else:
        t, start_epoch, best_inception = 0, 0, (0., 0.)
        lr = args.learning_rate
        checkpoint = {
            'args': args.__dict__,
            'options': options,
            'vocab': vocab,
            'train_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'train_inception': [],
            'lr': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'val_inception': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
                'best_inception': None,
            },
            'model_best_state': None, 'optim_state': None,
            'model_state_crop_encoder': None,
            'model_state_obj_embeddings': None,
            'model_state_pred_embeddings': None,
            'model_state_gconv': None, 'model_state_gconv_net': None,
            'model_state_img_decoder': None,
            'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
            'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
            'd_seg_state': None, 'd_seg_best_state': None, 'd_seg_optim_state': None,
            'd_seg2_state': None, 'd_seg2_best_state': None, 'd_seg2_optim_state': None,
            'd_seg3_state': None, 'd_seg3_best_state': None, 'd_seg3_optim_state': None,
            'd_seg_state128': None, 'd_seg_best_state128': None, 'd_seg_optim_state128': None,
            'd_seg_state64': None, 'd_seg_best_state64': None, 'd_seg_optim_state64': None,
        }

        log_path = os.path.join(
            options["logs"]["output_dir"],
            options["logs"]["name"] + "-" + time.strftime("%Y%m%d-%H%M%S")
        )
    logger = Logger(log_path)
    log.info("Logging to: {}".format(log_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patch_discriminator = nn.DataParallel(patch_discriminator.to(device)) if patch_discriminator else None

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        for iter, batch in enumerate(pyprind.prog_bar(train_loader, title="[Epoch {}/{}]".format(epoch, args.epochs), width=50)):
            if args.timing:
                print("Loading Time: {} ms".format((time.time() - start_time) * 1000))
            t += 1
            ######### unpack the data #########
            batch = unpack_batch3(batch, options)
            (imgs, canvases_sel, canvases_ori,
             objs, boxes, selected_crops,
             original_crops, triples, predicates,
             obj_to_img, triple_to_img,
             scatter_size_obj, scatter_size_triple) = batch

            image_id_list = []
            for mm in range(obj_to_img.shape[0]):
                if not (obj_to_img[mm] in image_id_list):
                    image_id_list.append(obj_to_img[mm])

            if (iter + 1) % args.visualize_every == 0:
                patch_discriminator.eval()
                samples = visualize_sample3_img_vg(patch_discriminator, batch, vocab)
                patch_discriminator.train()
                logger.image_summary(samples, t, tag="vis")

            d_patch_gan_loss = 0
            optimizer_d_seg2.zero_grad()
            for mm in range(len(image_id_list)):
                index_this = (obj_to_img == image_id_list[mm])
                original_crops_this = original_crops[index_this][:, 0:3, :, :]
                selected_crops_this = selected_crops[index_this][:, :, 0:3, :, :]
                objs_this = objs[index_this]
                bbox_this = boxes[index_this]
                obj_mask = objs_this.nonzero().view(-1)
                original_crops_this = original_crops_this[obj_mask]
                selected_crops_this = selected_crops_this[obj_mask]
                bbox_this = bbox_this[obj_mask]

                batch_size = original_crops_this.shape[0]
                if batch_size < 2:
                    continue
                random_choose = random.randint(0, batch_size - 1)
                patch_this_positive = original_crops_this[random_choose:random_choose + 1, :, :, :]
                num_negative = 5
                patch_this_negative = []
                for kkk in range(num_negative):
                    random_choose2 = random.randint(0, selected_crops.shape[0] - 1)
                    patch_this_negative_this = selected_crops[random_choose2, :, :, :, :]
                    random_choose3 = random.randint(0, patch_this_negative_this.shape[0]-1)
                    patch_this_negative_this = patch_this_negative_this[random_choose3:random_choose3+1, 0:3, :, :]
                    patch_this_negative.append(patch_this_negative_this)
                patch_this_negative = torch.cat(patch_this_negative, dim=0)
                bbox_this_one = bbox_this[random_choose:random_choose + 1]

                num_paste = random.randint(1, batch_size - 1)
                index_list = []
                for kkk in range(num_paste):
                    random_choose1 = random.randint(0, batch_size - 1)
                    while (random_choose1 in index_list) or (random_choose1 == random_choose):
                        random_choose1 = random.randint(0, batch_size - 1)
                    index_list.append(random_choose1)

                patch_this_choose = original_crops_this[index_list, 0:3, :, :]
                bbox_this_choose = bbox_this[index_list]
                bbox_this_choose = torch.cat([bbox_this_choose, bbox_this_one], dim=0)
                patch_this_positive = torch.cat([patch_this_choose, patch_this_positive], dim=0)
                output1 = patch_discriminator(patch_this_positive, bbox_this_choose)

                output_list = [output1]
                k_num = patch_this_negative.shape[0]
                for kkk in range(k_num):
                    patch_this_negative_input = torch.cat([patch_this_choose, patch_this_negative[kkk:kkk + 1]], dim=0)
                    output2 = patch_discriminator(patch_this_negative_input, bbox_this_choose)
                    output_list.append(output2)

                final_score1 = torch.cat(output_list, dim=0).view(1, k_num + 1)
                temperature = 0.02
                final_score1 = final_score1 * 1.0 / temperature
                final_score = torch.sum(torch.exp(output1 * 1.0 / temperature)) * 1.0 / torch.sum(torch.exp(final_score1))
                print(final_score, final_score1)
                d_patch_gan_loss_this = -torch.log(final_score)
                d_patch_gan_loss_this = d_patch_gan_loss_this * 1.0 / (len(image_id_list))

                d_patch_gan_loss_this.backward()
                d_patch_gan_loss = d_patch_gan_loss + d_patch_gan_loss_this

            optimizer_d_seg2.step()
            d_patch_losses = LossManager()
            d_patch_losses.add_loss(d_patch_gan_loss, 'd_patch_gan_loss')

            del boxes
            if patch_discriminator is not None:
                for name, val in d_patch_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
            start_time = time.time()

            if (iter + 1) % 1000 == 0:
                log.info('[Epoch {}/{}] checking on val'.format(epoch, args.epochs))
                isBest = True

                checkpoint['d_seg2_state'] = patch_discriminator.module.state_dict()
                checkpoint['d_seg2_optim_state'] = optimizer_d_seg2.state_dict()

                checkpoint['counters']['epoch'] = epoch
                checkpoint['counters']['t'] = t
                checkpoint['lr'] = lr
                checkpoint_path = os.path.join(log_path, '%s_with_model.pt' % args.checkpoint_name)
                log.info('[Epoch {}/{}] Saving checkpoint: {}'.format(epoch, args.epochs, checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                if isBest:
                    copyfile(checkpoint_path, os.path.join(log_path, 'best_with_model.pt'))

        if epoch % args.eval_epochs == 0:

            log.info('[Epoch {}/{}] checking on val'.format(epoch, args.epochs))
            isBest = True

            checkpoint['d_seg2_state'] = patch_discriminator.module.state_dict()
            checkpoint['d_seg2_optim_state'] = optimizer_d_seg2.state_dict()

            checkpoint['counters']['epoch'] = epoch
            checkpoint['counters']['t'] = t
            checkpoint['lr'] = lr
            checkpoint_path = os.path.join(log_path, '%s_with_model.pt' % args.checkpoint_name)
            log.info('[Epoch {}/{}] Saving checkpoint: {}'.format(epoch, args.epochs, checkpoint_path))
            torch.save(checkpoint, checkpoint_path)
            if isBest:
                copyfile(checkpoint_path, os.path.join(log_path, 'best_with_model.pt'))

        if epoch >= args.decay_lr_epochs:
            lr_end = args.learning_rate * 1e-3
            decay_frac = (epoch - args.decay_lr_epochs + 1) / (args.epochs - args.decay_lr_epochs + 1e-5)
            lr = args.learning_rate - decay_frac * (args.learning_rate - lr_end)
            for param_group in optimizer_d_seg2.param_groups:
                param_group["lr"] = lr
            log.info('[Epoch {}/{}] learning rate: {}'.format(epoch + 1, args.epochs, lr))
        logger.scalar_summary("ckpt/learning_rate", lr, epoch)


if __name__ == '__main__':
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True
    seed = 1234
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)

    main()
