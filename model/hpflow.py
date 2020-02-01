"""Implementation of Hyperpixel Flow: Semantic Correspondence with Multi-layer Nueral Features"""

from functools import reduce
from operator import add

from torchvision.models import resnet, vgg
import torch.nn.functional as F
import torch
import gluoncvth as gcv

from . import geometry
from . import util
from . import rhm


class HyperpixelFlow:
    r"""Hyperpixel Flow framework"""
    def __init__(self, backbone, hyperpixel_ids, benchmark, device):
        r"""Constructor for Hyperpixel Flow framework"""

        # Feature extraction network initialization.
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True).to(device)
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            nbottlenecks = [3, 4, 23, 3]
        elif backbone == 'fcn101':
            self.backbone = gcv.models.get_fcn_resnet101_voc(pretrained=True).to(device).pretrained
            nbottlenecks = [3, 4, 23, 3]
        elif backbone == 'vgg11':
            self.backbone = vgg.vgg11(pretrained=True).to(device)
            nbottlenecks = [8]
            self.maxpool_idx = [2,5,10,15,20]
        elif backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True).to(device)
            nbottlenecks = [13]
            self.maxpool_idx = [4,9,16,23,30]
        elif backbone == 'vgg19':
            self.backbone = vgg.vgg19(pretrained=True).to(device)
            nbottlenecks = [16]
            self.maxpool_idx = [4,9,18,27,36]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.backbone.eval()
        self.backbone_name = backbone

        # Hyperpixel id and pre-computed jump and receptive field size initialization
        # Reference: https://fomoro.com/research/article/receptive-field-calculator
        # (the jump and receptive field sizes for 'fcn101' are heuristic values)
        self.hyperpixel_ids = util.parse_hyperpixel(hyperpixel_ids)
        if 'vgg11' in self.backbone_name: ## total 13 backbone feature
            self.jsz = torch.tensor([1, 2, 4, 4, 8, 8, 16, 16]).to(device)  
            self.rfsz = torch.tensor([3,8,18,26,46,62,102,134]).to(device)
        if 'vgg16' in self.backbone_name: ## total 13 backbone feature
            self.jsz = torch.tensor([1, 1, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16]).to(device)  
            self.rfsz = torch.tensor([3, 5, 10, 14, 24, 32, 40, 60, 76, 92, 132, 164, 196]).to(device)
        if 'vgg19' in self.backbone_name: ## total 13 backbone feature
            self.jsz = torch.tensor([1, 1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16]).to(device)  
            self.rfsz = torch.tensor([3, 5, 10, 14, 24, 32, 40, 48, 68, 84, 100, 116, 156, 188, 220, 256, 288]).to(device)
        else:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)

        # Miscellaneous
        self.hsfilter = geometry.gaussian2d(7).to(device)
        self.device = device
        self.benchmark = benchmark

    def __call__(self, *args, **kwargs):
        r"""Forward pass"""
        src_hyperpixels = self.extract_hyperpixel(args[0])
        trg_hyperpixels = self.extract_hyperpixel(args[1])
        confidence_ts = rhm.rhm(src_hyperpixels, trg_hyperpixels, self.hsfilter)
        return confidence_ts, src_hyperpixels[0], trg_hyperpixels[0]

    def extract_hyperpixel(self, img):
        r"""Given image, extract desired list of hyperpixels"""
        if self.backbone_name == 'vgg16':
            hyperfeats, rfsz, jsz = self.extract_intermediate_feat_vgg16(img.unsqueeze(0))
        elif self.backbone_name == 'vgg19':
            hyperfeats, rfsz, jsz = self.extract_intermediate_feat_vgg19(img.unsqueeze(0))
        elif self.backbone_name == 'vgg11':
            hyperfeats, rfsz, jsz = self.extract_intermediate_feat_vgg11(img.unsqueeze(0))
        else:
            hyperfeats, rfsz, jsz = self.extract_intermediate_feat(img.unsqueeze(0))
        hpgeometry = geometry.receptive_fields(rfsz, jsz, hyperfeats.size()).to(self.device)
        hyperfeats = hyperfeats.view(hyperfeats.size()[0], -1).t()

        # Prune boxes on margins (causes error on Caltech benchmark)
        if self.benchmark != 'caltech':
            hpgeometry, valid_ids = geometry.prune_margin(hpgeometry, img.size()[1:], jsz.float())
            hyperfeats = hyperfeats[valid_ids, :]

        return hpgeometry, hyperfeats, img.size()[1:][::-1]

    def extract_intermediate_feat_vgg19(self, img):

        feats = []
        rfsz = self.rfsz[self.hyperpixel_ids[0]]
        jsz = self.jsz[self.hyperpixel_ids[0]]

        num_of_layers_vgg = len(self.backbone.features)
        # print(num_of_layers_vgg)

        # layer 0
        feat = self.backbone.features[0](img)
        feat = self.backbone.features[1](feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 1
        feat = self.backbone.features[2](feat)
        feat = self.backbone.features[3](feat)
        feat = self.backbone.features[4](feat)
        if 1 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 2
        feat = self.backbone.features[5](feat)
        feat = self.backbone.features[6](feat)
        if 2 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 3
        feat = self.backbone.features[7](feat)
        feat = self.backbone.features[8](feat)
        feat = self.backbone.features[9](feat)
        if 3 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 4
        feat = self.backbone.features[10](feat)
        feat = self.backbone.features[11](feat)
        if 4 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 5
        feat = self.backbone.features[12](feat)
        feat = self.backbone.features[13](feat)
        if 5 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 6
        feat = self.backbone.features[14](feat)
        feat = self.backbone.features[15](feat)
        if 6 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 7
        feat = self.backbone.features[16](feat)
        feat = self.backbone.features[17](feat)
        feat = self.backbone.features[18](feat)
        if 7 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 8
        feat = self.backbone.features[19](feat)
        feat = self.backbone.features[20](feat)
        if 8 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 9
        feat = self.backbone.features[21](feat)
        feat = self.backbone.features[22](feat)
        if 9 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 10
        feat = self.backbone.features[23](feat)
        feat = self.backbone.features[24](feat)
        if 10 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 11
        feat = self.backbone.features[25](feat)
        feat = self.backbone.features[26](feat)
        feat = self.backbone.features[27](feat)
        if 11 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 12
        feat = self.backbone.features[28](feat)
        feat = self.backbone.features[29](feat)
        if 12 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 13
        feat = self.backbone.features[30](feat)
        feat = self.backbone.features[31](feat)
        if 13 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 14
        feat = self.backbone.features[32](feat)
        feat = self.backbone.features[33](feat)
        if 14 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 15
        feat = self.backbone.features[34](feat)
        feat = self.backbone.features[35](feat)
        feat = self.backbone.features[36](feat)
        if 15 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            if idx == 0:
                continue
            feats[idx] = F.interpolate(feat, tuple(feats[0].size()[2:]), None, 'bilinear', True)
        feats = torch.cat(feats, dim=1)
        # print(feats.shape, rfsz, jsz)

        return feats[0], rfsz, jsz 

    def extract_intermediate_feat_vgg16(self, img):

        feats = []
        rfsz = self.rfsz[self.hyperpixel_ids[0]]
        jsz = self.jsz[self.hyperpixel_ids[0]]

        num_of_layers_vgg = len(self.backbone.features)
        # print(num_of_layers_vgg)

        # layer 0
        feat = self.backbone.features[0](img)
        feat = self.backbone.features[1](feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 1
        feat = self.backbone.features[2](feat)
        feat = self.backbone.features[3](feat)
        feat = self.backbone.features[4](feat)
        if 1 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 2
        feat = self.backbone.features[5](feat)
        feat = self.backbone.features[6](feat)
        if 2 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 3
        feat = self.backbone.features[7](feat)
        feat = self.backbone.features[8](feat)
        feat = self.backbone.features[9](feat)
        if 3 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 4
        feat = self.backbone.features[10](feat)
        feat = self.backbone.features[11](feat)
        if 4 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 5
        feat = self.backbone.features[12](feat)
        feat = self.backbone.features[13](feat)
        if 5 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 6
        feat = self.backbone.features[14](feat)
        feat = self.backbone.features[15](feat)
        feat = self.backbone.features[16](feat)
        if 6 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 7
        feat = self.backbone.features[17](feat)
        feat = self.backbone.features[18](feat)
        if 7 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 8
        feat = self.backbone.features[19](feat)
        feat = self.backbone.features[20](feat)
        if 8 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 9
        feat = self.backbone.features[21](feat)
        feat = self.backbone.features[22](feat)
        feat = self.backbone.features[23](feat)
        if 9 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 10
        feat = self.backbone.features[24](feat)
        feat = self.backbone.features[25](feat)
        if 10 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 11
        feat = self.backbone.features[26](feat)
        feat = self.backbone.features[27](feat)
        if 11 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 12
        feat = self.backbone.features[28](feat)
        feat = self.backbone.features[29](feat)
        feat = self.backbone.features[30](feat)
        if 12 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            if idx == 0:
                continue
            feats[idx] = F.interpolate(feat, tuple(feats[0].size()[2:]), None, 'bilinear', True)
        feats = torch.cat(feats, dim=1)
        # print(feats.shape, rfsz, jsz)

        return feats[0], rfsz, jsz 

    def extract_intermediate_feat_vgg11(self, img):
        feats = []
        rfsz = self.rfsz[self.hyperpixel_ids[0]]
        jsz = self.jsz[self.hyperpixel_ids[0]]

        num_of_layers_vgg = len(self.backbone.features)
        # print(num_of_layers_vgg)

        # layer 0
        feat = self.backbone.features[0](img)
        feat = self.backbone.features[1](feat)
        feat = self.backbone.features[2](feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 1
        feat = self.backbone.features[3](feat)
        feat = self.backbone.features[4](feat)
        feat = self.backbone.features[5](feat)
        if 1 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 2
        feat = self.backbone.features[6](feat)
        feat = self.backbone.features[7](feat)
        if 2 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 3
        feat = self.backbone.features[8](feat)
        feat = self.backbone.features[9](feat)
        feat = self.backbone.features[10](feat)
        if 3 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 4
        feat = self.backbone.features[11](feat)
        feat = self.backbone.features[12](feat)
        if 4 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 5
        feat = self.backbone.features[13](feat)
        feat = self.backbone.features[14](feat)
        feat = self.backbone.features[15](feat)
        if 5 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 6
        feat = self.backbone.features[16](feat)
        feat = self.backbone.features[17](feat)
        if 6 in self.hyperpixel_ids:
            feats.append(feat.clone())
        # layer 7
        feat = self.backbone.features[18](feat)
        feat = self.backbone.features[19](feat)
        feat = self.backbone.features[20](feat)
        if 7 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            if idx == 0:
                continue
            feats[idx] = F.interpolate(feat, tuple(feats[0].size()[2:]), None, 'bilinear', True)
        feats = torch.cat(feats, dim=1)
        # print(feats.shape, rfsz, jsz)

        return feats[0], rfsz, jsz 


    def extract_intermediate_feat(self, img):
        r"""Extract desired a list of intermediate features"""

        # print(self.bottleneck_ids)
        # print(self.layer_ids)
        # exit()

        feats = []
        rfsz = self.rfsz[self.hyperpixel_ids[0]]
        jsz = self.jsz[self.hyperpixel_ids[0]]

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                if hid + 1 == max(self.hyperpixel_ids):
                    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)


        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            if idx == 0:
                continue
            feats[idx] = F.interpolate(feat, tuple(feats[0].size()[2:]), None, 'bilinear', True)
        feats = torch.cat(feats, dim=1)

        print(feats.shape, rfsz, jsz)


        return feats[0], rfsz, jsz
