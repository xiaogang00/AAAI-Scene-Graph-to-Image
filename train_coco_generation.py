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


def main():
    global args, options
    print(args)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    log.info("Building loader...")
    vocab, train_loader = datasets.build_loaders5(options["data"])

    log.info("Building Generative Model...")
    model, model_kwargs = models.build_model(
        options["generator"],
        vocab,
        image_size=options["data"]["image_size"],
        checkpoint_start_from=args.checkpoint_start_from)
    model.type(float_dtype)
    print(model)

    optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.learning_rate,
            betas=(args.beta1, 0.999),)

    seg_discriminator, d_seg_kwargs = models.build_img_discriminator_feature(
        options["discriminator"], vocab
    )
    log.info("Done Building Segmentation Discriminator.")
    seg_discriminator.type(float_dtype)
    seg_discriminator.apply(weights_init)
    seg_discriminator.train()

    seg_discriminator128, d_seg_kwargs128 = models.build_img_discriminator_feature(
        options["discriminator128"], vocab
    )
    log.info("Done Building Segmentation Discriminator.")
    seg_discriminator128.type(float_dtype)
    seg_discriminator128.apply(weights_init)
    seg_discriminator128.train()

    seg_discriminator64, d_seg_kwargs64 = models.build_img_discriminator_feature(
        options["discriminator64"], vocab
    )
    log.info("Done Building Segmentation Discriminator.")
    seg_discriminator64.type(float_dtype)
    seg_discriminator64.apply(weights_init)
    seg_discriminator64.train()

    patch_discriminator = Encoder_patch()
    patch_discriminator.type(float_dtype)
    restore_path = 'path_to_trained_scsm/checkpoint_with_model.pt'
    log.info('Restoring from checkpoint: {}'.format(restore_path))
    checkpoint = torch.load(restore_path)
    patch_discriminator.load_state_dict(checkpoint['d_seg2_state'])
    patch_discriminator.eval()

    gan_g_loss, gan_d_loss = get_gan_losses(options["optim"]["gan_loss_type"])

    optimizer_d_seg = torch.optim.Adam(
        filter(lambda x: x.requires_grad, seg_discriminator.parameters()),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999), )

    optimizer_d_seg128 = torch.optim.Adam(
        filter(lambda x: x.requires_grad, seg_discriminator128.parameters()),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999), )

    optimizer_d_seg64 = torch.optim.Adam(
        filter(lambda x: x.requires_grad, seg_discriminator64.parameters()),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999), )

    perceptual_module = None
    if options["optim"].get("perceptual_loss_weight", -1) > 0 or \
                    options["optim"].get("obj_perceptual_loss_weight", -1) > 0:
        perceptual_module = getattr(
            models.perceptual,
            options.get("perceptual", {}).get("arch", "VGGLoss"))()

    restore_path = 'path_to_resume'
    if restore_path is not None and os.path.isfile(restore_path):
        log.info('Restoring from checkpoint: {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        model.crop_encoder.load_state_dict(checkpoint['model_state_crop_encoder'])
        model.obj_embeddings.load_state_dict(checkpoint['model_state_obj_embeddings'])
        model.pred_embeddings.load_state_dict(checkpoint['model_state_pred_embeddings'])
        model.gconv.load_state_dict(checkpoint['model_state_gconv'])
        model.gconv_net.load_state_dict(checkpoint['model_state_gconv_net'])
        model.box_net.load_state_dict(checkpoint['model_state_box_net'])
        model.img_decoder.load_state_dict(checkpoint['model_state_img_decoder'])


        if seg_discriminator is not None:
            seg_discriminator.load_state_dict(checkpoint['d_seg_state'])
            seg_discriminator128.load_state_dict(checkpoint['d_seg_state128'])
            seg_discriminator64.load_state_dict(checkpoint['d_seg_state64'])

        t = checkpoint['counters']['t'] + 1
        if 0 <= args.eval_mode_after <= t:
            model.eval()
        else:
            model.train()
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
            'model_kwargs': model_kwargs,
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
            'model_kwargs': model_kwargs,
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
    model = GeneratorDataParallel(model.to(device))
    seg_discriminator = nn.DataParallel(seg_discriminator.to(device)) if seg_discriminator else None
    seg_discriminator128 = nn.DataParallel(seg_discriminator128.to(device)) if seg_discriminator128 else None
    seg_discriminator64 = nn.DataParallel(seg_discriminator64.to(device)) if seg_discriminator64 else None
    perceptual_module = nn.DataParallel(perceptual_module.to(device)) if perceptual_module else None
    patch_discriminator.to(device)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        for iter, batch in enumerate(pyprind.prog_bar(train_loader, title="[Epoch {}/{}]".format(epoch, args.epochs), width=50)):

            if args.timing:
                print("Loading Time: {} ms".format((time.time() - start_time) * 1000))
            t += 1
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
                model_boxes = boxes

                model_out = model(objs, triples, obj_to_img, triple_to_img,
                                  boxes_gt=model_boxes,
                                  selected_crops=selected_crops,
                                  original_crops=original_crops,
                                  scatter_size_obj=scatter_size_obj,
                                  scatter_size_triple=scatter_size_triple)

                boxes_pred, seg_pred, seg_rcst, others = model_out
                graph_feature = others['graph_feature']

            if (iter+1) % args.visualize_every == 0:
                training_status = model.training
                model.eval()
                samples = visualize_sample3_img(model, patch_discriminator, batch, vocab)
                model.train(mode=training_status)
                logger.image_summary(samples, t, tag="vis")

            with timeit('G_loss', args.timing):
                # calculate L1 loss between imgs and imgs_self

                imgs = imgs.to(device)
                boxes = boxes.to(device)
                total_loss, losses = calculate_model_losses_seg_img(options["optim"],  boxes, boxes_pred, seg_rcst, imgs)

                h_target = 128
                w_target = 128
                imgs_128 = F.interpolate(imgs, size=[h_target, w_target], mode='bilinear')
                size128_loss = torch.mean(torch.abs(others['seg_rcst_128']-imgs_128))
                total_loss = add_loss(total_loss, size128_loss, losses, "size128_loss", 1.0)

                h_target = 64
                w_target = 64
                imgs_64 = F.interpolate(imgs, size=[h_target, w_target], mode='bilinear')
                size64_loss = torch.mean(torch.abs(others['seg_rcst_64'] - imgs_64))
                total_loss = add_loss(total_loss, size64_loss, losses, "size64_loss", 1.0)

                if seg_discriminator is not None:
                    weight = options["optim"]["d_loss_weight"] * options["optim"]["d_img_weight"]
                    scores_fake, feature_fake = seg_discriminator(seg_pred)
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses, 'g_gan_seg_loss', weight)

                    scores_fake_rcst, feature_fake_rcst = seg_discriminator(seg_rcst)
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake_rcst), losses, 'g_gan_seg_rcst_loss', weight)

                    with torch.no_grad():
                        _, feature_real_rcst = seg_discriminator(imgs)
                    loss_gan_feature = 0
                    weight = 0.1 / (len(feature_real_rcst))
                    for mm in range(len(feature_real_rcst)):
                        loss_gan_feature += torch.mean(torch.abs(feature_fake_rcst[mm] - feature_real_rcst[mm].detach())) * weight
                    total_loss = add_loss(total_loss, loss_gan_feature, losses, 'g_gan_img_feature_loss', 1.0)

                if seg_discriminator128 is not None:
                    weight = options["optim"]["d_loss_weight"] * options["optim"]["d_img_weight"]
                    scores_fake, _ = seg_discriminator128(others['seg_pred_128'])
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses, 'g_gan_seg128_loss', weight)

                    scores_fake_rcst, _ = seg_discriminator128(others['seg_rcst_128'])
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake_rcst), losses, 'g_gan_seg128_rcst_loss', weight)

                if seg_discriminator64 is not None:
                    weight = options["optim"]["d_loss_weight"] * options["optim"]["d_img_weight"]
                    scores_fake, _ = seg_discriminator64(others['seg_pred_64'])
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses, 'g_gan_seg64_loss', weight)

                    scores_fake_rcst, _ = seg_discriminator64(others['seg_rcst_64'])
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake_rcst), losses, 'g_gan_seg64_rcst_loss', weight)

                perceptual_loss = perceptual_module(seg_rcst, imgs)
                perceptual_loss = perceptual_loss.mean()
                weight = options["optim"]["perceptual_loss_weight"]
                total_loss = add_loss(total_loss, perceptual_loss, losses, "img_rcst_perceptual_loss", weight)

                pred256_downsample = F.interpolate(seg_pred, size=[64, 64], mode='bilinear')
                rcst256_downsample = F.interpolate(seg_rcst, size=[64, 64], mode='bilinear')
                pred128_downsample = F.interpolate(others['seg_pred_128'], size=[64, 64], mode='bilinear')
                rcst128_downsample = F.interpolate(others['seg_rcst_128'], size=[64, 64], mode='bilinear')
                loss_invariant = torch.mean(torch.abs(pred256_downsample-others['seg_pred_64']))
                loss_invariant += torch.mean(torch.abs(pred128_downsample - others['seg_pred_64']))
                loss_invariant += torch.mean(torch.abs(rcst256_downsample - others['seg_rcst_64']))
                loss_invariant += torch.mean(torch.abs(rcst128_downsample - others['seg_rcst_64']))
                total_loss = add_loss(total_loss, loss_invariant, losses, 'invariant_loss', weight)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                log.warn('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            with timeit('backward', args.timing):
                total_loss.backward()
            optimizer.step()

            with timeit('D_loss', args.timing):
                if seg_discriminator is not None:
                    d_seg_losses = LossManager()
                    seg_pred_fake = seg_pred.detach()
                    seg_rcst_fake = seg_rcst.detach()

                    scores_real, _ = seg_discriminator(imgs)
                    scores_fake_pred, _ = seg_discriminator(seg_pred_fake)
                    scores_fake_rcst, _ = seg_discriminator(seg_rcst_fake)

                    d_seg_gan_loss = gan_d_loss(scores_real, scores_fake_pred)
                    d_seg_losses.add_loss(d_seg_gan_loss, 'd_seg_gan_loss')

                    d_seg_gan_rcst_loss = gan_d_loss(scores_real, scores_fake_rcst)
                    d_seg_losses.add_loss(d_seg_gan_rcst_loss, 'd_seg_gan_rcst_loss')

                    optimizer_d_seg.zero_grad()
                    d_seg_losses.total_loss.backward()
                    optimizer_d_seg.step()

                if seg_discriminator128 is not None:
                    d_seg128_losses = LossManager()
                    seg_pred_fake = others['seg_pred_128'].detach()
                    seg_rcst_fake = others['seg_rcst_128'].detach()

                    scores_real, _ = seg_discriminator128(imgs_128)
                    scores_fake_pred, _ = seg_discriminator128(seg_pred_fake)
                    scores_fake_rcst, _ = seg_discriminator128(seg_rcst_fake)

                    d_seg_gan128_loss = gan_d_loss(scores_real, scores_fake_pred)
                    d_seg128_losses.add_loss(d_seg_gan128_loss, 'd_seg128_gan_loss')

                    d_seg_gan128_rcst_loss = gan_d_loss(scores_real, scores_fake_rcst)
                    d_seg128_losses.add_loss(d_seg_gan128_rcst_loss, 'd_seg128_gan_rcst_loss')

                    optimizer_d_seg128.zero_grad()
                    d_seg128_losses.total_loss.backward()
                    optimizer_d_seg128.step()

                if seg_discriminator64 is not None:
                    d_seg64_losses = LossManager()
                    seg_pred_fake = others['seg_pred_64'].detach()
                    seg_rcst_fake = others['seg_rcst_64'].detach()

                    scores_real, _ = seg_discriminator64(imgs_64)
                    scores_fake_pred, _ = seg_discriminator64(seg_pred_fake)
                    scores_fake_rcst, _ = seg_discriminator64(seg_rcst_fake)

                    d_seg_gan64_loss = gan_d_loss(scores_real, scores_fake_pred)
                    d_seg64_losses.add_loss(d_seg_gan64_loss, 'd_seg64_gan_loss')

                    d_seg_gan64_rcst_loss = gan_d_loss(scores_real, scores_fake_rcst)
                    d_seg64_losses.add_loss(d_seg_gan64_rcst_loss, 'd_seg64_gan_rcst_loss')

                    optimizer_d_seg64.zero_grad()
                    d_seg64_losses.total_loss.backward()
                    optimizer_d_seg64.step()

            del seg_map
            del boxes
            # Logging generative model losses
            for name, val in losses.items():
                logger.scalar_summary("loss/{}".format(name), val, t)
            # Logging discriminative model losses
            if seg_discriminator is not None:
                for name, val in d_seg_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
            if seg_discriminator128 is not None:
                for name, val in d_seg128_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
            if seg_discriminator64 is not None:
                for name, val in d_seg64_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
            start_time = time.time()

        if epoch % args.eval_epochs == 0:

            log.info('[Epoch {}/{}] checking on val'.format(epoch, args.epochs))
            isBest = True

            checkpoint['model_state_crop_encoder'] = model.module.crop_encoder.state_dict()
            checkpoint['model_state_obj_embeddings'] = model.module.obj_embeddings.state_dict()
            checkpoint['model_state_pred_embeddings'] = model.module.pred_embeddings.state_dict()
            checkpoint['model_state_gconv'] = model.module.gconv.state_dict()
            checkpoint['model_state_gconv_net'] = model.module.gconv_net.state_dict()
            checkpoint['model_state_box_net'] = model.module.box_net.state_dict()
            checkpoint['model_state_img_decoder'] = model.module.img_decoder.state_dict()

            if seg_discriminator is not None:
                checkpoint['d_seg_state'] = seg_discriminator.module.state_dict()
                checkpoint['d_seg_optim_state'] = optimizer_d_seg.state_dict()
                checkpoint['d_seg_state128'] = seg_discriminator128.module.state_dict()
                checkpoint['d_seg_optim_state128'] = optimizer_d_seg128.state_dict()
                checkpoint['d_seg_state64'] = seg_discriminator64.module.state_dict()
                checkpoint['d_seg_optim_state64'] = optimizer_d_seg64.state_dict()

            checkpoint['optim_state'] = optimizer.state_dict()
            checkpoint['counters']['epoch'] = epoch
            checkpoint['counters']['t'] = t
            checkpoint['lr'] = lr
            checkpoint_path = os.path.join(log_path, '%s_with_model_%d.pt' % (args.checkpoint_name, epoch))
            log.info('[Epoch {}/{}] Saving checkpoint: {}'.format(epoch, args.epochs, checkpoint_path))
            torch.save(checkpoint, checkpoint_path)
            if isBest:
                copyfile(checkpoint_path, os.path.join(log_path, 'best_with_model.pt'))

        if epoch >= args.decay_lr_epochs:
            lr_end = args.learning_rate * 1e-3
            decay_frac = (epoch - args.decay_lr_epochs + 1) / (args.epochs - args.decay_lr_epochs + 1e-5)
            lr = args.learning_rate - decay_frac * (args.learning_rate - lr_end)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            if seg_discriminator is not None:
                for param_group in optimizer_d_seg.param_groups:
                    param_group["lr"] = lr
            if seg_discriminator128 is not None:
                for param_group in optimizer_d_seg128.param_groups:
                    param_group["lr"] = lr
            if seg_discriminator64 is not None:
                for param_group in optimizer_d_seg64.param_groups:
                    param_group["lr"] = lr
            log.info('[Epoch {}/{}] learning rate: {}'.format(epoch+1, args.epochs, lr))
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
