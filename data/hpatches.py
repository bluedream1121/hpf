import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image

import cv2
import torch.nn.functional as F 

import numpy as np
from model import geometry


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def mnn_matcher_hpf(sim):#(descriptors_a, descriptors_b):
    # device = descriptors_a.device
    # sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=nn12.device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])

    return matches.t().data.cpu().numpy()

class HPatchesDataset(Dataset):
    r"""HPatches evaluation class"""
    def __init__(self, dataset_path, device):
        r"""CorrespondenceDataset constructor"""
        super(HPatchesDataset, self).__init__()

        self.device = device
        self.n_i = 52
        self.n_v = 56
        self.dataset_path = dataset_path

        self.normalize = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])
        self.extension = 'png'

        self.side_thres = 300
        
    def __len__(self):
        return self.n_i + self.n_v
    
    def __getitem__(self, seq_name, idx):
        im_name = os.path.join(self.dataset_path, seq_name, '%d.%s' % (idx, self.extension))
        img = cv2.imread(im_name)
        # img = img / 255.0
        img = self.normalize(img).to(self.device).unsqueeze(0)
        # hpnet doesn't need downsampled image
        #img_downsample, inter_ratio = self.resize(img)
        return img.squeeze(0)
        # return img.squeeze(0), img_downsample, inter_ratio

    # def resize(self, img, side_thres=300):
    #     r"""Resize given image with imsize: (1, 3, H, W)"""
    #     imsize = torch.tensor(img.size()).float()
    #     side_max = torch.max(imsize)
    #     inter_ratio = 1.0
    #     if side_max > side_thres:
    #         inter_ratio = side_thres / side_max
    #         img = F.interpolate(img,
    #                             size=(int(imsize[2] * inter_ratio), int(imsize[3] * inter_ratio)),
    #                             mode='bilinear',
    #                             align_corners=False)
    #     return img.squeeze(0), inter_ratio

def summary_hpatches(stats, n_i, n_v):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5),
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5),
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )


def resize_img(img, side_thres=600):
    r"""Resize given image with imsize: (1, 3, H, W)"""
    imsize = torch.tensor(img.size()).float()
    side_max = torch.max(imsize)
    inter_ratio = 1.0
    if side_max > side_thres:
        inter_ratio = side_thres / side_max
        img = F.interpolate(img,
                            size=(int(imsize[2] * inter_ratio), int(imsize[3] * inter_ratio)),
                            mode='bilinear',
                            align_corners=False)
    return img.squeeze(0), inter_ratio



def compute_mnn_matches(img_a, img_b, model, topk=None):
    
    """_, H_img_b, W_img_b = img_b.shape
    upsample_size = (model.imside // 4, model.imside//4) 

    rsz_value =   model.imside / np.array(img_a.shape)[1:][::-1]  
    rsz_value_2 = model.imside / np.array(img_b.shape)[1:][::-1] 
    rfsz = 11; jsz = 4
    
    ## make image same size 
    img_a = F.interpolate(img_a.unsqueeze(0), size=(H_img_b, W_img_b), mode='bilinear', align_corners=True).squeeze(0)

    src_box = geometry.receptive_fields(rfsz, jsz, upsample_size).to(img_a.device)
    trg_box = geometry.receptive_fields(rfsz, jsz, upsample_size).to(img_a.device)
    # src_box, src_valid_ids = geometry.prune_margin(src_box, img_b.size()[1:], float(jsz))
    # trg_box, trg_valid_ids = geometry.prune_margin(trg_box, img_b.size()[1:], float(jsz))
    
    #TODO: every grid considered as keypoints
    # This can be refined by off-the-shelf keypoint detectors
    keypoints_a = geometry.center(src_box)
    keypoints_b = geometry.center(trg_box)

    keypoints_a = keypoints_a.cpu() / rsz_value
    keypoints_b = keypoints_b.cpu() / rsz_value_2

    keypoints_a = torch.cat((keypoints_a, torch.ones(keypoints_a.shape[0], dtype=torch.double).unsqueeze(1)), dim=1).numpy()
    keypoints_b = torch.cat((keypoints_b, torch.ones(keypoints_b.shape[0], dtype=torch.double).unsqueeze(1)), dim=1).numpy()
    """
    img_a, inter_ratio_a = resize_img(img_a.unsqueeze(0))
    img_b, inter_ratio_b = resize_img(img_b.unsqueeze(0))
    with torch.no_grad():
        correlation_ts, src_box, trg_box = model(img_a, img_b)
    ## keypoints resized
    keypoints_a = geometry.center(src_box)
    keypoints_b = geometry.center(trg_box)
    keypoints_a = keypoints_a.cpu() / inter_ratio_a
    keypoints_b = keypoints_b.cpu() / inter_ratio_b

    keypoints_a = torch.cat((keypoints_a, torch.ones(keypoints_a.shape[0], dtype=torch.float).unsqueeze(1)), dim=1).numpy()
    keypoints_b = torch.cat((keypoints_b, torch.ones(keypoints_b.shape[0], dtype=torch.float).unsqueeze(1)), dim=1).numpy()

    if topk is not None:
        matches = topk_mnn_matcher_for_hpf_vis(correlation_ts, topk)
    else:
        matches = mnn_matcher_hpf(correlation_ts.squeeze(0))

    return keypoints_a, keypoints_b, matches

def compute_patches_error_dist(keypoints_a, keypoints_b, matches, homography):
    pos_a = keypoints_a[matches[:, 0], : 2]
    pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
    pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
    pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2:]

    pos_b = keypoints_b[matches[:, 1], : 2]
    dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

    if dist.shape[0] == 0:
        dist = np.array([float("inf")])

    return dist

def print_results_by_thres(errors, method, n_i, n_v, rng):
    i_err, v_err, _ = errors[method]
    print('method: ', method)
    printer = {}
    for thr in rng:
        printer[thr] = round( (i_err[thr] +v_err[thr]) / ((n_i + n_v) * 5), 4)
    print('\n total thr: ', printer)
    for thr in rng:
        printer[thr] = round( i_err[thr] / (n_i * 5) , 4) 
    print('\n illum thr: ', printer)
    for thr in rng:
        printer[thr] =  round( v_err[thr] / (n_v * 5), 4) 
    print('\n vp thr: ', printer)

import matplotlib.pyplot as plt

def result_plotting(eval_method , res, top_k=None):
    methods = ['hesaff', 'hesaffnet', 'delf', 'delf-new', 'superpoint', 'lf-net', 'd2-net', 'd2-net-ms', 'd2-net-trained', 'd2-net-trained-ms']
    names = ['Hes. Aff. + Root-SIFT', 'HAN + HN++', 'DELF', 'DELF New', 'SuperPoint', 'LF-Net', 'D2-Net', 'D2-Net MS', 'D2-Net Trained', 'D2-Net Trained MS']
    colors = ['black', 'orange', 'red', 'red', 'blue', 'brown', 'purple', 'green', 'purple', 'green']
    linestyles = ['-', '-', '-', '--', '-', '-', '-', '-', '--', '--']
    # methods = ['hesaff', 'hesaffnet', 'delf', 'delf-new', 'superpoint', 'd2-net', 'd2-net-trained']
    # names = ['Hes. Aff. + Root-SIFT', 'HAN + HN++', 'DELF', 'DELF New', 'SuperPoint', 'D2-Net', 'D2-Net Trained']
    # colors = ['black', 'orange', 'red', 'red', 'blue', 'purple', 'purple']
    # linestyles = ['-', '-', '-', '--', '-', '-', '--']

    n_i = 52; n_v = 56
    cache_dir = 'cache'
    errors = {}
    ## Load the cached results. You can add the pre-computed results by your own hand.
    for method in methods:
        output_file = os.path.join(cache_dir, method + '.npy')
        if os.path.exists(output_file):
            print('Loading precomputed errors...')
            errors[method] = np.load(output_file, allow_pickle=True)
        else:
            print(output_file , ' does not exist.')
        # for check
        summary_hpatches(errors[method][-1],n_i ,n_v)

    ## add the results.
    for key in res:
        errors[key] = res[key]
        methods.append(key)
        names.append(key)
        colors.append('cyan')
        linestyles.append('-')

    ## plotting config
    plt_lim = [1, 15]
    plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)

    for method, name, color, ls in zip(methods, names, colors, linestyles):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
    plt.title('Overall')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylabel('MMA')
    plt.ylim([0, 1])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()

    plt.subplot(1, 3, 2)
    for method, name, color, ls in zip(methods, names, colors, linestyles):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
    plt.title('Illumination')
    plt.xlabel('threshold [px]')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.subplot(1, 3, 3)
    for method, name, color, ls in zip(methods, names, colors, linestyles):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
    plt.title('Viewpoint')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    if top_k is None:
        plt.savefig(eval_method +'_hseq.pdf', bbox_inches='tight', dpi=300)
    else:
        plt.savefig('hseq-top.pdf', bbox_inches='tight', dpi=300)