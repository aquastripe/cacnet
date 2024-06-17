import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from KUPCP_dataset import CompositionDataset, composition_cls
from config_cropping import cfg


def evaluate_composition_classification(model):
    model.eval()
    device = next(model.parameters()).device
    print('=' * 5, 'Evaluating on Composition Classification Dataset', '=' * 5)
    total = 0
    correct = 0
    cls_cnt = [0 for i in range(9)]
    cls_correct = [0 for i in range(9)]

    with torch.no_grad():
        test_dataset = CompositionDataset(split='test', keep_aspect_ratio=cfg.keep_aspect_ratio)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            im = batch_data[0].to(device)
            labels = batch_data[1]
            image_path = batch_data[2]

            logits, kcm = model(im, only_classify=True)
            logits = logits.cpu()
            _, predicted = torch.max(logits.data, 1)
            total += labels.shape[0]
            pr = predicted[0].item()
            gt = labels[0].numpy().tolist()

            if pr in gt:
                correct += 1
                cls_cnt[pr] += 1
                cls_correct[pr] += 1
            else:
                cls_cnt[gt[0]] += 1
    acc = float(correct) / total
    print('Test on {} images, {} Correct, Acc {:.2%}'.format(total, correct, acc))
    for i in range(len(cls_cnt)):
        print('{}: total {} images, {} correct, Acc {:.2%}'.format(
            composition_cls[i], cls_cnt[i], cls_correct[i], float(cls_correct[i]) / cls_cnt[i]))
    return acc


def visualize_com_prediction(image_path, logits, kcm, category, save_folder):
    _, predicted = torch.max(logits.data, 1)
    # print('Composition prediction', predicted)
    # print('Ground-truth composition', category)
    label = composition_cls[predicted[0].item()]
    gt_label = [composition_cls[c] for c in category[0].numpy().tolist()]
    im = cv2.imread(image_path[0])
    height, width, _ = im.shape
    dst = im.copy()
    gt_ss = 'gt:{}'.format(gt_label)
    dst = cv2.putText(dst, gt_ss, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    pr_ss = 'predict:{}'.format(label)
    dst = cv2.putText(dst, pr_ss, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    # h,w,1
    kcm = kcm.permute(0, 2, 3, 1)[0].detach().cpu().numpy().astype(np.float32)
    # norm_kcm = np.zeros((height, width, 1))
    norm_kcm = cv2.normalize(kcm, None, 0, 255, cv2.NORM_MINMAX)
    norm_kcm = np.asarray(norm_kcm, dtype=np.uint8)
    heat_im = cv2.applyColorMap(norm_kcm, cv2.COLORMAP_JET)
    # heat_im = cv2.cvtColor(heat_im, cv2.COLOR_BGR2RGB)
    heat_im = cv2.resize(heat_im, (width, height))
    fuse_im = cv2.addWeighted(im, 0.2, heat_im, 0.8, 0)
    fuse_im = np.concatenate([dst, fuse_im], axis=1)
    cv2.imwrite(os.path.join(save_folder, os.path.basename(image_path[0])), fuse_im)
