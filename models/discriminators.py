import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bilinear import crop_bbox_batch, uncrop_bbox
from utils.box_utils import box_union, box_in_region
from .layers import GlobalAvgPool, Flatten, get_activation, build_cnn, build_cnn_feature
import models


class PatchDiscriminator(nn.Module):
    def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
                 padding='same', pooling='avg', input_size=(128, 128),
                 layout_dim=0):
        super(PatchDiscriminator, self).__init__()
        input_dim = 3 + layout_dim
        arch = 'I%d,%s' % (input_dim, arch)
        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        self.cnn, output_dim = build_cnn(**cnn_kwargs)
        self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

    def forward(self, x, layout=None):
        if layout is not None:
            x = torch.cat([x, layout], dim=1)
        return self.classifier(self.cnn(x))


####
class PatchDiscriminator_feature(nn.Module):
    def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
                 padding='same', pooling='avg', input_size=(128, 128),
                 layout_dim=0):
        super(PatchDiscriminator_feature, self).__init__()
        input_dim = 3 + layout_dim
        arch = 'I%d,%s' % (input_dim, arch)
        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        self.cnn, output_dim = build_cnn_feature(**cnn_kwargs)
        self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

    def forward(self, x, layout=None):
        if layout is not None:
            x = torch.cat([x, layout], dim=1)

        output_feature = []
        for layer in self.cnn.children():
            x = layer(x)
            output_feature.append(x)
        x = self.classifier(x)
        return x, output_feature


class AcDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 padding='same', pooling='avg'):
        super(AcDiscriminator, self).__init__()
        self.vocab = vocab

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, D = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
        num_objects = len(vocab['object_idx_to_name'])

        self.real_classifier = nn.Linear(1024, 1)
        self.obj_classifier = nn.Linear(1024, num_objects)

    def forward(self, x, y):
        if x.dim() == 3:
            x = x[:, None]

        vecs = self.cnn(x)
        real_scores = self.real_classifier(vecs)
        obj_scores = self.obj_classifier(vecs)
        ac_loss = F.cross_entropy(obj_scores, y)
        return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 object_size=64, padding='same', pooling='avg'):
        super(AcCropDiscriminator, self).__init__()
        self.vocab = vocab
        self.discriminator = AcDiscriminator(vocab, arch, normalization,
                                             activation, padding, pooling)
        self.object_size = object_size

    def forward(self, imgs, objs, boxes, obj_to_img,
                object_crops=None, **kwargs):
        if object_crops is None:
            object_crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
        real_scores, ac_loss = self.discriminator(object_crops, objs)
        return real_scores, ac_loss, object_crops


################
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, num_D=2, n_layers_D=4):
        super(MultiscaleDiscriminator, self).__init__()
        for i in range(num_D):
            subnetD = self.create_single_discriminator(n_layers_D)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, n_layers_D):
        netD = NLayerDiscriminator(n_layers_D)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input, layout):
        input = torch.cat([input, layout], dim=1)
        result = []
        out_list = []
        for name, D in self.named_children():
            feature_out, out = D(input)
            result.append(feature_out)
            out_list.append(out)
            input = self.downsample(input)

        return result, out_list

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, n_layers_D, ndf=64):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = int((kw - 1.0) / 2)
        nf = ndf
        input_nc = 3 + 184

        norm_layer = nn.InstanceNorm2d
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]]

        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)), nn.LeakyReLU(0.2, False)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for name, submodel in self.named_children():
            if 'model' not in name:
                continue
            else:
                x = results[-1]
            intermediate_output = submodel(x)
            results.append(intermediate_output)
        return results[1:-1], results[-1]
