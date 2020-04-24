r"""Runs Hyperpixel Flow framework"""

import argparse
import datetime
import os

from torch.utils.data import DataLoader
import torch

from model import hpflow, geometry, evaluation, util
from data import download

from data.hpatches import HPatchesDataset, compute_mnn_matches, compute_patches_error_dist
from data.hpatches import summary_hpatches, print_results_by_thres, result_plotting

import numpy as np
from tqdm import tqdm


def run(datapath, benchmark, backbone, thres, alpha, hyperpixel,
        logpath, beamsearch, model=None, dataloader=None, visualize=False):
    r"""Runs Hyperpixel Flow framework"""

    # 1. Logging initialization
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    if not beamsearch:
        cur_datetime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logfile = os.path.join('logs', logpath + cur_datetime + '.log')
        util.init_logger(logfile)
        util.log_args(args)
        if visualize: os.mkdir(logfile + 'vis')


    # 2. Evaluation benchmark initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if benchmark == 'hpatches':
        method = 'hpf_'+ '_' + backbone + '_' + str(hyperpixel) #+ '_' + str(args.exp_id)
        if not beamsearch:
            method += '_' + str(args.exp_id)
        dataset_path = '/home/jongmin/datasets/hpatches-sequences/hpatches-sequences-release'
        if dataloader is None:
            # method = 'hpf_'+ '_' + args.backbone + '_' + str(args.hyperpixel) + '_' + str(args.exp_id)
            dset = HPatchesDataset(dataset_path, device)
            dataloader = dset
        
    else:
        if dataloader is None:
            download.download_dataset(os.path.abspath(datapath), benchmark)
            split = 'val' if beamsearch else 'test'
            dset = download.load_dataset(benchmark, datapath, thres, device, split)
            dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    # 3. Model initialization
    if model is None:
        model = hpflow.HyperpixelFlow(backbone, hyperpixel, benchmark, device)
    else:
        model.hyperpixel_ids = util.parse_hyperpixel(hyperpixel)

    if benchmark == 'hpatches':
        with torch.no_grad():
            errors = test_hpacthes(model, method, dataset_path, dataloader, device)
        if beamsearch:
            ## Beamsearched by pixel thres 5px
            thr = 5
            i_err, v_err, _ = errors[method]
            # printer[thr] = round( (i_err[thr] +v_err[thr]) / ((n_i + n_v) * 5), 4)
            #return (sum(evaluator.eval_buf['pck']) / len(evaluator.eval_buf['pck'])) * 100.
            return (i_err[thr] +v_err[thr]) / ((dataloader.n_i + dataloader.n_v) * 5)
        else:
            result_plotting(eval_method=method, res=errors)
    else:
        # 4. Evaluator initialization
        evaluator = evaluation.Evaluator(benchmark, device)
        for idx, data in enumerate(dataloader):
            # a) Retrieve images and adjust their sizes to avoid large numbers of hyperpixels
            data['src_img'], data['src_kps'], data['src_intratio'] = util.resize(data['src_img'], data['src_kps'][0])
            data['trg_img'], data['trg_kps'], data['trg_intratio'] = util.resize(data['trg_img'], data['trg_kps'][0])
            data['alpha'] = alpha
            # b) Feed a pair of images to Hyperpixel Flow model
            with torch.no_grad():
                confidence_ts, src_box, trg_box = model(data['src_img'], data['trg_img'])
            # c) Predict key-points & evaluate performance
            prd_kps = geometry.predict_kps(src_box, trg_box, data['src_kps'], confidence_ts)
            evaluator.evaluate(prd_kps, data)
            # d) Log results
            if not beamsearch:
                evaluator.log_result(idx, data=data)
            if visualize:
                vispath = os.path.join(logfile + 'vis', '%03d_%s_%s' % (idx, data['src_imname'][0], data['trg_imname'][0]))
                util.visualize_prediction(data['src_kps'].t().cpu(), prd_kps.t().cpu(),
                                        data['src_img'], data['trg_img'], vispath)


        if beamsearch:
            return (sum(evaluator.eval_buf['pck']) / len(evaluator.eval_buf['pck'])) * 100.
        else:
            evaluator.log_result(len(dset), data=None, average=True)

def test_hpacthes(model, method, dataset_path, dataset_cls,device):
    seq_names = sorted(os.listdir(dataset_path))  ## Get the sequence list.
    lim = [1, 15]; rng = np.arange(lim[0], lim[1] + 1)

    cache_dir = 'cache_hpnet'    
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    errors = {}
    output_file = os.path.join(cache_dir, method + '.npy')
    n_i = dataset_cls.n_i
    n_v = dataset_cls.n_v

    if os.path.exists(output_file):
        print('Loading precomputed errors...')
        errors[method] = np.load(output_file, allow_pickle=True)
    else:
        n_feats = []; n_matches = [];  seq_type = []
        i_err = {thr: 0 for thr in rng}; v_err = {thr: 0 for thr in rng}

        # print(seq_names)
        for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
            # img_a_origin, img_a, rsz_value = dataset_cls.__getitem__(seq_name, 1)
            img_a = dataset_cls.__getitem__(seq_name, 1)
            for im_idx in range(2,7):
                # img_b_origin, img_b, rsz_value_2 = dataset_cls.__getitem__(seq_name, im_idx)
                img_b = dataset_cls.__getitem__(seq_name, im_idx)
                # print("img_a.shape, img_b.shape : ", img_a.shape, img_b.shape)
                homography = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))

                keypoints_a, keypoints_b, matches = compute_mnn_matches(img_a, img_b, model)
                dist = compute_patches_error_dist(keypoints_a, keypoints_b, matches, homography)

                n_feats.append(keypoints_b.shape[0])
                n_matches.append(matches.shape[0])
                seq_type.append(seq_name[0])
                
                for thr in rng:
                    if seq_name[0] == 'i':
                        i_err[thr] += np.mean(dist <= thr)
                    else:
                        v_err[thr] += np.mean(dist <= thr)

                # del dist; del correlation_ts; del matches
        seq_type = np.array(seq_type)
        n_feats = np.array(n_feats)
        n_matches = np.array(n_matches)
        
        errors[method] = (i_err, v_err, [seq_type, n_feats, n_matches])
    
    print('Results is saved at ', output_file)
    np.save(output_file, errors[method])
    summary_hpatches(errors[method][-1], n_i, n_v)
    print_results_by_thres(errors, method, n_i, n_v, rng)

    return errors


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Hyperpixel Flow in pytorch')
    parser.add_argument('--datapath', type=str, default='../Datasets_HPF')
    parser.add_argument('--dataset', type=str, default='hpatches')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--hyperpixel', type=str, default='')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    run(datapath=args.datapath, benchmark=args.dataset, backbone=args.backbone, thres=args.thres, alpha=args.alpha,
        hyperpixel=args.hyperpixel, logpath=args.logpath, beamsearch=False, visualize=args.visualize)
