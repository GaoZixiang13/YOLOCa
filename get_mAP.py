from net import loss
from net import yolox
from net import preprocess
from utils import load_model

import pandas as pd
import torch, time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import glob, tqdm
from PIL import Image, ImageDraw
from torchvision.ops import nms

def cal_iou(boxes1, box2):
    '''
    :param boxes1:
    :param box2:
    :return:
    '''
    zsxy = torch.maximum(boxes1[..., 0:2] - boxes1[..., 2:4] / 2, box2[..., 0:2] - box2[..., 2:4] / 2)
    yxxy = torch.minimum(boxes1[..., 0:2] + boxes1[..., 2:4] / 2, box2[..., 0:2] + box2[..., 2:4] / 2)
    wh = torch.maximum(yxxy - zsxy, torch.zeros_like(yxxy))
    Intersection = wh[..., 0] * wh[..., 1]
    Union = boxes1[..., 2] * boxes1[..., 3] + box2[..., 2] * box2[..., 3] - Intersection

    return Intersection / torch.clamp(Union, 1e-6)

def decode(prediction, l_size, stride):
    '''
    :param x: tensor: [l_size, l_size]
    :param y:
    :param w:
    :param h:
    :return: [..., 4]
    '''
    x, y, w, h = prediction[..., 0], prediction[..., 1], prediction[..., 2], prediction[..., 3]
    ix = torch.arange(l_size).repeat(l_size, 1).to(device)
    iy = torch.arange(l_size).repeat(l_size, 1).t().to(device)
    x = (ix+x) * stride
    y = (iy+y) * stride
    w = torch.exp(w) * stride
    # w = w.squeeze(0)
    h = torch.exp(h) * stride

    prediction[..., 0], prediction[..., 1], prediction[..., 2], prediction[..., 3] = x, y, w, h

    return prediction

def get_AP(preds, targets, threshold):
    if targets.size(0) == 0:
        return 1 if preds.size(0) == 0 else 0
    preds_ex = preds.unsqueeze(-2).expand(preds.size(0), targets.size(0), 5 + num_classes)
    iou, iou_t_idx = torch.max(cal_iou(preds_ex, targets), dim=-1)
    iou, iou_t_idx = np.array(iou.cpu()), np.array(iou_t_idx.cpu())
    # print(iou_t_idx)
    hitted = torch.zeros(targets.size(0))
    n = targets.size(0)
    hit = 0
    points = [[0, 1]]
    for i, iou_s in enumerate(iou):
        if iou_s > threshold:
            if hitted[iou_t_idx[i]] == 0:
                hit += 1
                r = hit / n
                p = hit / (i+1)
                points.append([r, p])
                hitted[iou_t_idx[i]] = 1
                if r == 1:
                    break

    points.append([1, 0])
    if len(points) == 2:
        return 0
        # raise RuntimeError('no pred hit!')

    points = np.array(points).astype(np.float32)
    pointsR, pointsP = points[:, 0], points[:, 1]

    ret = 0.
    for i in range(1, len(pointsR)):
        ret += (pointsR[i] - pointsR[i-1]) * pointsP[i] + np.abs(pointsP[i-1] - pointsP[i]) * (pointsR[i] - pointsR[i-1]) * 0.5

    return ret

def get_TrueFalseNum(pred_box, target, threshold=0.5):
    '''
    根据预测框与真实框的交并比区分出tp,fp,fn,tn样本
    :param pred_box: [n, 4] [x,y,w,h]
    :param target: [n, 5 + num_classes]
    :return: [true_num, false_num]
    '''
    if target.size(0) == 0:
        return 0, 0
    ty_true = target[..., 0:4]
    preds = torch.unsqueeze(pred_box, dim=-2).expand(pred_box.size(0), ty_true.size(0), 4)
    # print(preds.size())
    iou = cal_iou(preds, ty_true)
    # print(iou.size())
    maxiou = torch.max(iou, dim=-1).values
    # print(maxiou.size())
    keep = maxiou >= threshold

    return torch.sum(keep == True).item(), torch.sum(keep == False).item()

def get_PR(preds, targets, threshold=0.5):
    '''
    :param preds: 必须是已解码的预测值 tensor [n, 5 + num_classes] [x,y,w,h,conf,cls]
    :param targets: tensor [n, 5 + num_classes] [x,y,w,h,conf,cls]
    :return: P, R 值
    TP : score > 0.5 & iou > 0.5
    FP : score > 0.5 & iou < 0.5
    FN : score < 0.5 & iou > 0.5
    TN : score < 0.5 & iou < 0.5
    P = TP/(TP + FP)
    R = TP/(TP + FN)
    '''
    # indices = preds[..., 4] > threshold
    # box_pred = box_pred[indices]
    pos_boxes = preds
    target = targets[targets[..., 4]==1].view(-1, 6)
    score, cnt = 0, 0
    for pd_gt_iou_hold in np.arange(0.5, 0.76, 0.05):
        tp, fp = get_TrueFalseNum(pos_boxes[..., 0:4], target, pd_gt_iou_hold)
        fn = torch.sum(targets[..., 4] == 1) - tp
        # fn, tn = get_TrueFalseNum(neg_boxes[..., 0:4], targets, iou_hold)
        score += tp/max(tp+fp+fn, 1e-6)
        cnt += 1
    score /= cnt

    tp, fp = get_TrueFalseNum(pos_boxes[..., 0:4], target, threshold)
    fn = torch.sum(targets[..., 4] == 1) - tp
    P = tp/max(tp+fp, 1e-6)
    R = tp/max(tp+fn, 1e-6)
    f1 = 2*P*R/max(P+R, 1e-6)

    return P, R, f1, score
    # print(f'精确度：{P}, 召回率：{R}, f1值：{f1}')

def Nms_yolox_self(bboxes, threshold=0.45):
    '''
    :param bboxes: [n, 6] [x,y,w,h,conf,cls]
    :param threshold:
    :return:
    '''
    # area = bboxes[..., 3] * bboxes[..., 4]
    # classes = torch.argmax(bboxes[..., 5:])
    bboxes = bboxes.view(-1, 6) # [n, 6]
    zs = bboxes[..., 0:2] - bboxes[..., 2:4]/2
    yx = bboxes[..., 0:2] + bboxes[..., 2:4]/2
    box4 = torch.cat([zs, yx], dim=-1)
    # bboxes[..., 0:2], bboxes[..., 2:4] = zs, yx
    # boxes = bboxes.view(-1, 6)
    # boxes = torch.cat((zs, yx, conf.unsqueeze(-1), cls.unsqueeze(-1)), dim=-1).view(-1, 6) # [n, 6]

    # if negetive:
    #     indice = nms(box4, 1 - bboxes[..., 5], threshold)
    # else:
    indice = nms(box4, bboxes[..., 4]*bboxes[..., 5], threshold)

    return bboxes[indice] # [m, 6]

# def decode_y(y):
#     '''
#     :param y: [bs, 3, l_size, l_size, 6]
#     :return: [n, 6]
#     '''
#     ty = y[y[..., 4]==1].view(-1, 6)
#     return ty

x_test_path, y_test = [], []
with open('/home/b201/gzx/yolox_self/val.txt') as f:
    for line in f.readlines():
        line = line.strip('\n')
        data = line.split(' ')
        # jpg_name = data[0].split('/')[-1]
        # x_test_path.append('C:/practice/global-wheat-detection/train/'+jpg_name)
        x_test_path.append(data[0])
        # if len(data)%5 != 1:
        #     raise RuntimeError('stop')
        lists = []
        for i in range(1, len(data) - 1, 5):
            if data[i] == '' or data[i] == ' ' or data[i] == '\n':
                break
            list = [float(data[i]), float(data[i + 1]), float(data[i + 2]), float(data[i + 3]), int(data[i + 4])]
            lists.append(list)
        y_test.append(lists)

CUDA = True
NMS_hold = 0.45
conf_hold = 0.5
pred_gt_cal_pr = 0.5
pic_shape = 1024
input_shape = 640
num_classes = 1
gpu_device_id = 0
Parallel = False
strides = (8, 16, 32)
get_ap = True

device = torch.device("cuda:%d" % gpu_device_id if torch.cuda.is_available() else "cpu")

test_loader = DataLoader(
    dataset=preprocess.yolodataset(x_test_path, y_test, input_shape, 1, train=False),
    shuffle=False,
    batch_size=1,
    num_workers=8,
    # drop_last=False
)
# img_path = x_test_path[0]
# y_test0 = y_test[0]

model = yolox.yolox(1)
model_path = '/home/b201/gzx/yolox_self/logs/' \
             'center_focal_loss2.057'
load_model.load_model(model, model_path)

# print('Load weights {}.'.format(model_path))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if Parallel:
    model = torch.nn.DataParallel(model)

if CUDA:
    model = model.to(device)
    # anchors = anchors.cuda()


model.eval()
# max_p, max_r, max_f1 = 0., 0., 0.
tol_p, tol_r, tol_f1, tol_s = 0., 0., 0., 0.
FPS = 0.
AP = torch.zeros(10)
with tqdm.tqdm(total=len(test_loader), desc=f'mAP Testing', postfix=dict) as pbar:
    with torch.no_grad():
        for _, (bx, by) in enumerate(test_loader):
            if CUDA:
                bx = bx.cuda()
                by = by.cuda()
            # print(img_path)
            # img = image_preprocess(img_path)
            st=time.time()
            outputs = model(bx)
            ed=time.time()
            FPS += 1/(ed-st)
            # target = by
            # P, R, f1 = .0, .0, .0
            predboxes = []
            for j, output in enumerate(outputs):
                l_size = input_shape // strides[j]
                output = output.view(1, 5 + num_classes, l_size, l_size).permute(0, 2, 3, 1).contiguous()
                output[..., :2] = torch.sigmoid(output[..., :2])
                output[..., 4:] = torch.sigmoid(output[..., 4:])
                output = decode(output, l_size, strides[j])
                predboxes.append(output.view(1, l_size * l_size, 5 + num_classes))
            prediction = torch.cat(predboxes, dim=1).view(-1, 5 + num_classes)

            # print(boxes_pred[torch.argsort(boxes_pred[..., 4], dim=-1, descending=True)])
            # print(target)
            gt_tures = by[by[..., 4] == 1].view(-1, 5+num_classes)

            if get_ap:
                mask = prediction[..., 4]*prediction[..., 5] > 0.001
                prediction_nmsed = Nms_yolox_self(prediction[mask], NMS_hold)
                prediction_t = prediction_nmsed[torch.argsort(prediction_nmsed[..., 4]*prediction_nmsed[..., 5], descending=True)]
                for num, hold in enumerate(torch.arange(0.5, 0.951, 0.05)):
                    ap = get_AP(prediction_t, gt_tures, hold)
                    AP[num] += ap

            pred_true_boxes = prediction[prediction[..., 4]*prediction[..., 5] >= conf_hold]
            pred_true_boxes = Nms_yolox_self(pred_true_boxes, NMS_hold)

            P, R, f1, score = get_PR(pred_true_boxes, gt_tures, pred_gt_cal_pr)

            tol_p += P
            tol_r += R
            tol_f1 += f1
            tol_s += score
            pbar.update(1)

n = len(test_loader)
# n = len(test_loader)
ever_p = tol_p / n
ever_r = tol_r / n
ever_f1 = tol_f1 / n
ever_s = tol_s / n

FPS /= len(x_test_path)
AP /= len(x_test_path)

AP50 = AP[0].item()
AP75 = AP[5].item()
AP = (AP.sum(-1)/10).item()

print('test 数据集上得到的P, R, f1, score值, FPS分别为：{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}'.format(ever_p, ever_r, ever_f1, ever_s, FPS))
print('AP为：{:.4f}'.format(AP))
print('AP50为：{:.4f}'.format(AP50))
print('AP75为：{:.4f}'.format(AP75))

