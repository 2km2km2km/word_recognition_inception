from __future__ import division

from models import *
#from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model,path,img_size, batch_size, criterion):
    model.eval()

    # Get dataloader
    dataset = CASIA_Dataset(path, img_size=img_size)
    #print(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(dataloader), batch_time, losses, top1, top5)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    total_acc=0
    for batch_i, (_, imgs, y_reals) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        y_reals = Variable(y_reals.to("cuda"), requires_grad=False)
        # print("train y_real",y_reals)
        # y_reals=real2more(y_reals,10)
        with torch.no_grad():
            y_hats = model(imgs)
        loss = criterion(y_hats, y_reals)
        _, prediction = torch.max(y_hats.data, 1)
        batch_len=len(y_reals)
        correct = (prediction == y_reals).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
        # measure accuracy and record loss
        loss = criterion(y_hats, y_reals)
        acc1, acc5 = accuracy(y_hats, y_reals, topk=(1, 2))
        losses.update(loss.item(), imgs.size(0))
        top1.update(acc1[0], imgs.size(0))
        top5.update(acc5[0], imgs.size(0))
        #print("y_real",y_reals)
        #print("y_hat,",prediction)
        #sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    #true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    #precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    precision=1
    #return precision, recall, AP, f1, ap_class
    return losses,top1,top5


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'