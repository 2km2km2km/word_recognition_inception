#各种库的导入
from __future__ import division

from models import *
#from utils.utils import *
from utils.datasets import *
from utils.loss import *
from utils.parse_config import *
#from test import evaluate
from test import *


#from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter

w=SummaryWriter(comment="inception")

def train(epochs):
    #载入部分超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--learning_rate", default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=6, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/casia.data", help="path to data config file")
    #parser.add_argument("--pretrained_weights", type=str, default='./weights/darknet53.conv.74',help="if specified starts from checkpoint model")
    #parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    #parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    #print("type(opt) --> ",type(opt))
    #print(opt)

    #创建文件夹
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    #判断能否使用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    # 初始化网络模型
    #model = InceptionV2(opt.model_def, img_size=opt.img_size).to(device)
    model = InceptionV2().to(device)
    '''
    ＃先不用
    # 初始化权重　
    model.apply(weights_init_normal)
    
    # print(model)

    # 载入已经预训练的模型参数
    """
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    """
    # 对模型每一层的参数关闭自动求导？？
    for i, p in enumerate(model.named_parameters()):
        if i == 156:
            break
        p[1].requires_grad = False
    '''
    # Get data configuration
    # 获取数据配置
    data_config = parse_data_config(opt.data_config)

    train_path = data_config["train"]
    test_path = data_config["test"]


    # Get dataloader
    # 获取dataloader
    dataset = CASIA_Dataset(train_path,img_size=opt.img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        #num_workers=　opt.n_cpu,  # 实现多线程
        #pin_memory=True,  # 是否将数据集拷贝到显卡上
        collate_fn=dataset.collate_fn,  # 将数据整合成一个batch返回的方法
    )

    # writer = SummaryWriter('logs')

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    count_show = 30  # 每训练30步显示一次训练效果

    #for epoch in range(opt.epochs):
    precisions=[]
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        batch_time = AverageMeter('Time', ':6.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(dataloader), batch_time, losses, top1,top5)
        epoch_loss=0
        for batch_i, (_, imgs, y_reals) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i  #已训练的图片batch数
            imgs = Variable(imgs.to(device))
            y_reals = Variable(y_reals.to(device), requires_grad=False)
            #print("train y_real",y_reals)
            #y_reals=real2more(y_reals,10)
            y_hats = model(imgs)
            loss=loss_CEL(y_hats,y_reals)
            acc1, acc5 = accuracy(y_hats, y_reals, topk=(1, 2))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1[0], imgs.size(0))
            top5.update(acc5[0], imgs.size(0))
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            #metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            '''
            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]
            
                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                # for targe, value in tensorboard_log:
                # writer.add_scalar(targe, value, batches_done)
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)
            '''
            # log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            epoch_loss+=loss.item()
            #print(log_str)

            #model.seen += imgs.size(0)
        print("epoch",epoch)
        print("loss",epoch_loss)
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            val_losses,val_top1,val_top5=evaluate(
                model,
                test_path,
                opt.img_size,
                6,
                loss_CEL
            )
            #precisions.append(precision)
            #print(precision)
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/inception_ckpt_%d.pth" % epoch)
        w.add_scalars("loss",{"train":losses.avg,"val":val_losses.avg},epoch)
        w.add_scalars("top1",{"train":top1.avg,"val":val_top1.avg},epoch)
        w.add_scalars("top5",{"train":top5.avg,"val":val_top5.avg},epoch)
    w.close()
    return precisions

if __name__ == "__main__":
    train(20)