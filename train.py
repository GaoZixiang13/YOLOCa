from net import loss
from net import yolox
from net import preprocess
from utils import load_model, utils_fit

import pandas as pd
import torch, random, time, os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import glob, tqdm
from PIL import Image

# from torch.backends import cudnn
# cudnn.benchmark = False            # if benchmark=True, deterministic will be False
# cudnn.deterministic = True
'''
一般来说，神经网络不收敛的原因有以下 11 种原因：
忘记对你的数据进行归一化
忘记检查输出结果
没有对数据进行预处理
没有使用任何的正则化方法
使用了一个太大的 batch size
使用一个错误的学习率
在最后一层使用错误的激活函数
网络包含坏的梯度
网络权重没有正确的初始化
使用了一个太深的神经网络
隐藏层神经元数量设置不正确
# ---------------------------------------------------------------
yolox未实现的模块有：
1. 数据增强如 mosaic, cutmix
2. 
# ---------------------------------------------------------------
'''

x_train_path, y_train, x_val_path, y_val = [], [], [], []
with open('/home/b201/gzx/yolox_self/train.txt') as f:
    for line in f.readlines():
        line = line.strip('\n')
        data = line.split(' ')
        # print(data)
        x_train_path.append(data[0])
        # if len(data)%5 != 1:
        #     raise RuntimeError('stop')
        lists = []
        for i in range(1, len(data)-1, 5):
            # if data[i] == '' or data[i] == ' ' or data[i] == '\n':
            #     break
            list = [float(data[i]), float(data[i + 1]), float(data[i + 2]), float(data[i + 3]), int(data[i + 4])]
            lists.append(list)
        y_train.append(lists)

with open('/home/b201/gzx/yolox_self/val.txt') as f1:
    for line1 in f1.readlines():
        line1 = line1.strip('\n')
        data1 = line1.split(' ')
        x_val_path.append(data1[0])
        # if len(data)%5 != 1:
        #     raise RuntimeError('stop')
        lists1 = []
        for i in range(1, len(data1)-1, 5):
            # if data[i] == '' or data[i] == ' ' or data[i] == '\n':
            #     break
            list1 = [float(data1[i]), float(data1[i + 1]), float(data1[i + 2]), float(data1[i + 3]), int(data1[i + 4])]
            lists1.append(list1)
        y_val.append(lists1)

# ---------------------------------------------------------------
# YoloX Hyper Parameters
BATCH_SIZE = 16
# 初始学习率大小
warmup = False
warmup_lr = 1e-6
basic_lr_per_img = 0.01 / 64.0
LR = basic_lr_per_img * BATCH_SIZE
use_cosine = False
# 训练的世代数
warmup_epoch = 1
start_epoch = 0
EPOCH = 70
# 图片原来的size
pic_shape = 1024
# 网络输入图片size
RE_SIZE_shape = 640
# 总的类别数
num_classes = 1
# 标签平滑
label_smoothing = 0
# Cosine_lr = False
CUDA = True
# 是否载入预训练模型参数
use_pretrain = False
Parallel = False
# gpu
gpu_device_id = 1
# ---------------------------------------------------------------

device = torch.device("cuda:%d" % gpu_device_id if torch.cuda.is_available() else "cpu")

# 数据集读取
train_loader = DataLoader(
    dataset=preprocess.yolodataset(x_train_path, y_train, RE_SIZE_shape, num_classes, train=True),
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=BATCH_SIZE,
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    dataset=preprocess.yolodataset(x_val_path, y_val, RE_SIZE_shape, num_classes, train=False),
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=BATCH_SIZE,
    pin_memory=True,
    drop_last=True
)

model = yolox.yolox(num_classes)

if use_pretrain:
    LR = 7.61e-6
    model_path = '/home/b201/gzx/yolox_self/logs/' \
                 'val_loss3.507-size640-lr0.00000761-ep055-train_loss3.465.pth'
else:
    model_path = ''

if model_path != '':
    load_model.load_model(model, model_path)
else:
    yolox.weights_init(model)

# 冻结主干进行训练
# for param in model.backbone.parameters():
#     param.requires_grad = False
if Parallel:
    model = torch.nn.DataParallel(model)

if CUDA:
    model = model.to(device)

'''
Adam等自适应学习率算法对于稀疏数据具有优势，且收敛速度很快；
但精调参数的SGD（+Momentum）往往能够取得更好的最终结果。
'''
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
loss_func = loss.yolox_loss(input_shape=RE_SIZE_shape, num_classes=num_classes, label_smoothing=label_smoothing, device=device)
# loss_func = loss_baseBox.YOLOLoss(num_classes=num_classes)
if not use_cosine:
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
else:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=LR/100)


val_loss_save = 1e10
# time = time.asctime(time.localtime(time.time()))
# logs_save_path = '/home/b201/gzx/yolox_self/logs/' + str(time)
# os.mkdir(logs_save_path)
# 预热训练
if not use_pretrain and warmup:
    print('start warm up Training!')
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    optimizer_wp = torch.optim.Adam(model.parameters(), lr=warmup_lr, weight_decay=5e-4)
    lr_scheduler_wp = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_wp, gamma=(LR/warmup_lr)**(1/warmup_epoch))
    for epoch in range(warmup_epoch):
        val_loss_save = utils_fit.fit_one_epoch(model, optimizer_wp, loss_func, lr_scheduler_wp, warmup_epoch, epoch, train_loader, val_loader, RE_SIZE_shape, val_loss_save, time, CUDA, device, warmup=True)
    print('Finish warm up Training!')
# 正式训练
val_loss_save = 1e10
print('start Training!')
for epoch in range(start_epoch, EPOCH):
    val_loss_save = utils_fit.fit_one_epoch(model, optimizer, loss_func, lr_scheduler, EPOCH, epoch, train_loader, val_loader, RE_SIZE_shape, val_loss_save, time, CUDA, device)

