import argparse
import json
import os
import warnings

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from CACNet import CACNet
from Cropping_dataset import FCDBDataset, FLMSDataset
from config_cropping import cfg

warnings.filterwarnings("ignore")


def compute_iou_and_disp(gt_crop, pre_crop, im_w, im_h):
    ''''
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    '''
    gt_crop = gt_crop[gt_crop[:, 0] >= 0]
    zero_t = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.maximum(gt_crop[:, 0], pre_crop[:, 0])
    over_y1 = torch.maximum(gt_crop[:, 1], pre_crop[:, 1])
    over_x2 = torch.minimum(gt_crop[:, 2], pre_crop[:, 2])
    over_y2 = torch.minimum(gt_crop[:, 3], pre_crop[:, 3])
    over_w = torch.maximum(zero_t, over_x2 - over_x1)
    over_h = torch.maximum(zero_t, over_y2 - over_y1)
    inter = over_w * over_h
    area1 = (gt_crop[:, 2] - gt_crop[:, 0]) * (gt_crop[:, 3] - gt_crop[:, 1])
    area2 = (pre_crop[:, 2] - pre_crop[:, 0]) * (pre_crop[:, 3] - pre_crop[:, 1])
    union = area1 + area2 - inter
    iou = inter / union
    disp = (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])) / im_w + \
           (torch.abs(gt_crop[:, 1] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / im_h
    iou_idx = torch.argmax(iou, dim=-1)
    dis_idx = torch.argmin(disp, dim=-1)
    index = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
    return iou[index].item(), disp[index].item()


def evaluate_on_FCDB_and_FLMS(model, dataset, save_results=False, results_dir=''):
    model.eval()
    device = next(model.parameters()).device
    accum_disp = 0
    accum_iou = 0
    crop_cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0

    if save_results:
        save_file = os.path.join(results_dir, dataset + '.json')
        crop_dir = os.path.join(results_dir, dataset)
        os.makedirs(crop_dir, exist_ok=True)
        test_results = dict()

    print('=' * 5, f'Evaluating on {dataset}', '=' * 5)
    with torch.no_grad():
        if dataset == 'FCDB':
            test_set = [FCDBDataset]
        elif dataset == 'FLMS':
            test_set = [FLMSDataset]
        else:
            raise Exception('Undefined test set ', dataset)
        for dataset in test_set:
            test_dataset = dataset(split='test',
                                   keep_aspect_ratio=cfg.keep_aspect_ratio)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
            for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                im = batch_data[0].to(device)
                gt_crop = batch_data[1]  # x1,y1,x2,y2
                width = batch_data[2].item()
                height = batch_data[3].item()
                image_file = batch_data[4][0]
                image_name = os.path.basename(image_file)

                logits, kcm, crop = model(im, only_classify=False)
                crop[:, 0::2] = crop[:, 0::2] / im.shape[-1] * width
                crop[:, 1::2] = crop[:, 1::2] / im.shape[-2] * height
                pred_crop = crop.detach().cpu()
                gt_crop = gt_crop.reshape(-1, 4)
                pred_crop[:, 0::2] = torch.clip(pred_crop[:, 0::2], min=0, max=width)
                pred_crop[:, 1::2] = torch.clip(pred_crop[:, 1::2], min=0, max=height)

                iou, disp = compute_iou_and_disp(gt_crop, pred_crop, width, height)
                if iou >= alpha:
                    alpha_cnt += 1
                accum_iou += iou
                accum_disp += disp
                cnt += 1

                if save_results:
                    best_crop = pred_crop[0].numpy().tolist()
                    best_crop = [int(x) for x in best_crop]  # x1,y1,x2,y2
                    test_results[image_name] = best_crop

                    # save the best crop
                    source_img = cv2.imread(image_file)
                    croped_img = source_img[best_crop[1]: best_crop[3], best_crop[0]: best_crop[2]]
                    cv2.imwrite(os.path.join(crop_dir, image_name), croped_img)
    if save_results:
        with open(save_file, 'w') as f:
            json.dump(test_results, f)
    avg_iou = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--results')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    results_dir = args.results
    os.makedirs(results_dir, exist_ok=True)
    weight_file = args.weight
    model = CACNet(loadweights=False)
    model.load_state_dict(torch.load(weight_file, map_location='cpu'))
    model = model.to(device).eval()
    evaluate_on_FCDB_and_FLMS(model, dataset='FCDB', save_results=True, results_dir=results_dir)
    evaluate_on_FCDB_and_FLMS(model, dataset='FLMS', save_results=True, results_dir=results_dir)


if __name__ == '__main__':
    main()
