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
