import math
import copy
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from random import random
import random
import glog as log


import utils.box_utils as box_utils
from .graph import GraphTripleConv, GraphTripleConvNet
from .graph2d import GraphTripleConv2d, GraphTripleConv2dNet
from utils.layout import boxes_to_layout, masks_to_layout, boxes_to_layouts
from .layers import build_mlp, ResidualBlock
from .paste_gan_base import PasteGAN_Base
from .crop_encoder import CropEncoder
from utils.bilinear import crop_bbox_batch

import cv2

class PSGIM(PasteGAN_Base):
    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 gconv_valid_edge_only=False,
                 refinement_dims=(1024, 512, 256, 128, 64),
                 normalization='batch', activation='leakyrelu-0.2',
                 mlp_normalization='none', canvas_noise_dim=0,
                 crop_encoder=None, generator_kwargs=None,
                 transform_residual=False, gconv2d_num_layers=4,
                 crop_matching_loss=False, class_related_bbox=False,
                 use_flow_net=False, use_mask_net=False,
                 mask_size=None, use_canvas_res=True,
                 **kwargs):

        super(PSGIM, self).__init__(vocab, image_size, )
        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)
        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)
        self.crop_embedding_dim = crop_encoder["crop_embedding_dim"]
        self.canvas_noise_dim = canvas_noise_dim
        self.use_canvas_res = use_canvas_res
        self.class_related_bbox = class_related_bbox
        self.crop_matching_loss = crop_matching_loss
        self.gconv2d_num_layers = gconv2d_num_layers

        self.crop_encoder = CropEncoder(
            output_D=self.crop_embedding_dim,
            num_categories=num_objs if crop_encoder["category_aware_encoder"] else 1,
            cropEncoderArgs=crop_encoder["crop_encoder_kwargs"],
            decoder_dims = crop_encoder.get('decoder_dims', None),
            pooling=crop_encoder["pooling"],
        )


        if self.class_related_bbox:
            box_net_dim = 4 * num_objs
        else:
            box_net_dim = 4
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

        assert gconv_num_layers > 2
        gconv_kwargs = {
            'input_dim': embedding_dim,
            'output_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'mlp_normalization': mlp_normalization,
        }
        self.gconv = GraphTripleConv(**gconv_kwargs)
        gconv_kwargs = {
            'input_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers - 2,
            'mlp_normalization': mlp_normalization,
        }
        self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        from .generation.generator2 import PSGIM_network
        self.label_nc = 184
        self.max_objects_per_image = 10
        self.canvas_noise_dim = 0
        input_dim = (128 + self.canvas_noise_dim + 256) * 2
        self.img_decoder = PSGIM_network(input_dim, label_nc=self.label_nc, nf=32, style_dim=256)

    def forward(self, objs, triples, obj_to_img=None, triple_to_img=None,
                boxes_gt=None, selected_crops=None, original_crops=None,
                **kwargs):
        """
        Required Inputs:
        - objs: LongTensor of shape (O,) giving categories for all objects
        - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])
        - selected_crops: LongTensor of shape (O, 3, H/2, W/2), giving the selected
          object crops as the source materials of generation.
        - original_crops: LongTensor of shape (O, 3, H/2, W/2), giving the original
          object crops as the source materials of generation.
          (If you don't want to specify the object crops in inference, the Selector
           will select the most-matching crops for the generation.)

        Optional Inputs:
        - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
          means that objects[o] is an object in image i. If not given then
          all objects are assumed to belong to the same image.
        - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
          the spatial layout; if not given then use predicted boxes.
        """
        O, T = objs.size(0), triples.size(0)
        HH_o, WW_o = selected_crops.size(2), selected_crops.size(3)
        s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o, p], dim=1)          # Shape is (T, 2)
        others = {}

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)

        obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        #####################################################################
        others['graph_feature'] = obj_vecs.clone().detach()
        #####################################################################

        boxes_pred = self.box_net(obj_vecs)
        if self.class_related_bbox:
            obj_cat = objs.view(-1, 1) * 4 + torch.arange(end=4, device=objs.device).view(1, -1)
            boxes_pred = boxes_pred.gather(1, obj_cat)
        layout_boxes = boxes_pred if boxes_gt is None else boxes_gt


        selected_crops_seg = selected_crops[:, 0:3, :, :]
        original_crops_seg = original_crops[:, 0:3, :, :]
        sel_crops_feat = self.crop_encoder(selected_crops_seg, objs)
        ori_crops_feat = self.crop_encoder(original_crops_seg, objs)

        H = 256
        W = 256
        H2 = 64
        W2 = 64

        f_h, f_w = sel_crops_feat.shape[2], sel_crops_feat.shape[3]
        obj_vecs = obj_vecs.view(O, -1, 1, 1)
        obj_vecs = obj_vecs.repeat(1, 1, f_h, f_w)
        input_feature_ori = torch.cat([ori_crops_feat, obj_vecs], dim=1)
        input_feature_sel = torch.cat([sel_crops_feat, obj_vecs], dim=1)

        input_feature_ori = boxes_to_layouts(input_feature_ori, layout_boxes, H2, W2)
        input_feature_sel = boxes_to_layouts(input_feature_sel, layout_boxes, H2, W2)
        input_img_ori = boxes_to_layouts(original_crops[:, 0:3, :, :], layout_boxes, H, W)
        input_img_sel = boxes_to_layouts(selected_crops[:, 0:3, :, :], layout_boxes, H, W)

        image_id_list = []
        for mm in range(obj_to_img.shape[0]):
            if not (obj_to_img[mm] in image_id_list):
                image_id_list.append(obj_to_img[mm])

        image_list_ori = []
        image_list_sel = []
        image_list_ori_img = []
        image_list_sel_img = []


        style_feature_ori = []
        style_feature_sel = []

        degree_num = [0] * obj_to_img.shape[0]
        for mm in range(edges.shape[0]):
            p_this = edges[mm, 0].item()
            q_this = edges[mm, 1].item()
            degree_num[p_this] += 1
            degree_num[q_this] += 1
        degree_num = torch.Tensor(degree_num).float()

        for mm in range(len(image_id_list)):
            index_this = (obj_to_img == image_id_list[mm])

            style_feature_ori_list = original_crops[index_this][:, 0:3, :, :]
            style_feature_sel_list = selected_crops[index_this][:, 0:3, :, :]
            objs_this = objs[index_this]
            obj_mask = objs_this.nonzero().view(-1)
            style_feature_ori_list = style_feature_ori_list[obj_mask]
            style_feature_sel_list = style_feature_sel_list[obj_mask]
            style_feature_ori.append(style_feature_ori_list)
            style_feature_sel.append(style_feature_sel_list)

            feature_this_ori = input_feature_ori[index_this]
            feature_this_ori = torch.sum(feature_this_ori, dim=0).unsqueeze(dim=0)

            feature_this_sel = input_feature_sel[index_this]
            feature_this_sel = torch.sum(feature_this_sel, dim=0).unsqueeze(dim=0)

            ##########################################
            degree_list = degree_num[index_this]
            degree_list = degree_list[obj_mask]

            image_this_ori = input_img_ori[index_this]
            image_this_ori = image_this_ori[obj_mask]
            image_this_ori = image_this_ori[:, 0:1, :, :] * 0.299 + \
                             image_this_ori[:, 1:2, :, :] * 0.587 + \
                             image_this_ori[:, 2:3, :, :] * 0.114

            batch_size = image_this_ori.shape[0]
            image_this_ori_this = image_this_ori.view(batch_size, H * W).permute(1, 0)
            image_this_ori_this_index = (image_this_ori_this == 0)
            image_this_ori_this_index = image_this_ori_this_index.float()
            image_this_ori_this_index = 1-image_this_ori_this_index

            degree_list = degree_list.float().to(feature_this_ori.device)
            degree_list = degree_list.view(1, batch_size).repeat(H*W, 1)
            image_this_ori_this_index = image_this_ori_this_index * degree_list

            max_index = torch.argmax(image_this_ori_this_index, dim=1)
            index_range = np.linspace(1, H*W, H*W)-1
            index_range = index_range.astype(np.int)
            image_this_ori_that = image_this_ori_this[index_range, max_index]
            image_this_ori_that = image_this_ori_that.view(1, 1, H, W)

            mask_this = (image_this_ori_that == 0)
            mask_this = mask_this.float()
            mask_this = F.interpolate(mask_this, size=[64, 64], mode='nearest')
            channel_feature = feature_this_ori.shape[1]
            mask_this = mask_this.repeat(1, channel_feature, 1, 1)
            feature_this_ori = torch.cat([feature_this_ori, mask_this], dim=1)
            image_list_ori.append(feature_this_ori)
            image_list_ori_img.append(image_this_ori_that)

            ##########################################
            image_this_sel = input_img_sel[index_this]
            image_this_sel = image_this_sel[obj_mask]
            image_this_sel = image_this_sel[:, 0:1, :, :] * 0.299 + \
                             image_this_sel[:, 1:2, :, :] * 0.587 + \
                             image_this_sel[:, 2:3, :, :] * 0.114

            batch_size = image_this_sel.shape[0]
            image_this_sel_this = image_this_sel.view(batch_size, H * W).permute(1, 0)
            image_this_sel_this_index = (image_this_sel_this == 0)
            image_this_sel_this_index = image_this_sel_this_index.float()
            image_this_sel_this_index = 1-image_this_sel_this_index

            image_this_sel_this_index = image_this_sel_this_index * degree_list

            max_index = torch.argmax(image_this_sel_this_index, dim=1)
            index_range = np.linspace(1, H * W, H * W) - 1
            index_range = index_range.astype(np.int)
            image_this_sel_that = image_this_sel_this[index_range, max_index]
            image_this_sel_that = image_this_sel_that.view(1, 1, H, W)

            mask_this2 = (image_this_sel_that == 0)
            mask_this2 = mask_this2.float()
            mask_this2 = F.interpolate(mask_this2, size=[64, 64], mode='nearest')
            mask_this2 = mask_this2.repeat(1, channel_feature, 1, 1)
            feature_this_sel = torch.cat([feature_this_sel, mask_this2], dim=1)
            image_list_sel.append(feature_this_sel)
            image_list_sel_img.append(image_this_sel_that)

        image_list_ori = torch.cat(image_list_ori, dim=0)
        image_list_sel = torch.cat(image_list_sel, dim=0)
        image_list_ori_img = torch.cat(image_list_ori_img, dim=0).detach()
        image_list_sel_img = torch.cat(image_list_sel_img, dim=0).detach()

        if not (self.canvas_noise_dim == 0):
            B, C, H2, W2 = image_list_ori.size()
            noise1 = torch.randn(B, self.canvas_noise_dim, H2, W2).to(image_list_ori.device)
            noise2 = torch.randn(B, self.canvas_noise_dim, H2, W2).to(image_list_ori.device)
            image_list_ori = torch.cat([image_list_ori, noise1], dim=1)
            image_list_sel = torch.cat([image_list_sel, noise2], dim=1)

        seg_pred, seg_pred_128, seg_pred_64 = self.img_decoder(image_list_sel, image_list_sel_img, style_feature_sel)
        seg_rcst, seg_rcst_128, seg_rcst_64 = self.img_decoder(image_list_ori, image_list_ori_img, style_feature_ori)
        others['seg_pred_128'] = seg_pred_128
        others['seg_pred_64'] = seg_pred_64
        others['seg_rcst_128'] = seg_rcst_128
        others['seg_rcst_64'] = seg_rcst_64

        return boxes_pred, seg_pred, seg_rcst, others
