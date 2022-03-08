from net import loss
from net import yolox
from net import preprocess
from utils import load_model, utils_fit

import pandas as pd
import torch, time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import glob, tqdm
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

x_test_path = glob.glob('/home/b201/gzx/yolov4_self_0/test/*.jpg')
print(x_test_path)

CUDA = True
NMS_hold = 0.45
conf_hold = 0.5
pic_shape = 1024
input_shape = 640
num_classes = 1
gpu_device_id = 0
Parallel = False
strides = (8, 16, 32)
single_img = False

# font = ImageFont.truetype(r'/home/b201/gzx/yolox_self/font/STSONG.TTF', 12)

# anchors_path = '/home/b201/gzx/yolov3_self/utils/yolo_wheat_anchors.txt'
# # 先验框的大小
# # 输入为416，anchor大小为
# anchors = load_model.load_anchors(anchors_path)
# anchors = torch.tensor(anchors)/pic_shape

device = torch.device("cuda:%d" % gpu_device_id if torch.cuda.is_available() else "cpu")
model = yolox.yolox(1)
model_path = '/home/b201/gzx/yolox_self/logs/' \
             'center_stride_2_loss3.464'
load_model.load_model(model, model_path)

if Parallel:
    model = torch.nn.DataParallel(model)

if CUDA:
    model = model.to(device)
    # anchors = anchors.cuda()

def Nms_yolov4_self(bboxes, threshold=0.3):
    # area = bboxes[..., 3] * bboxes[..., 4]
    # classes = torch.argmax(bboxes[..., 5:])
    zs = bboxes[..., 0:2] - bboxes[..., 2:4]/2
    yx = bboxes[..., 0:2] + bboxes[..., 2:4]/2
    box4 = torch.cat([zs, yx], dim=-1)
    # bboxes[..., 0:2], bboxes[..., 2:4] = zs, yx
    # boxes = bboxes.view(-1, 6)
    # boxes = torch.cat((zs, yx, conf.unsqueeze(-1), cls.unsqueeze(-1)), dim=-1).view(-1, 6) # [n, 6]

    indice = nms(box4, bboxes[..., 4]*bboxes[..., 5], threshold)
    # boxes = torch.cat((bboxes[..., 5].unsqueeze(dim=-1), zs, bboxes[..., 2:4]), dim=-1)

    return bboxes[indice] # [m, 4]

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
    # h = h.squeeze(0)
    # print(x.size())
    # print(y.size())
    # print(w.size())
    # print(h.size())
    # box = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], dim=-1)
    return prediction

def image_preprocess_test(img_path):
    img = Image.open(img_path).convert('RGB')
    # img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.5)(img)
    img = img.resize((input_shape, input_shape), Image.BICUBIC)
    img = np.transpose(np.array(img) / 255., (2, 0, 1))
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # img, labels = self.horizontal_flip(img, labels)
    return img_tensor

FPS = 0
model.eval()
if single_img:
    img_path = '/home/b201/gzx/global-wheat-detection/train/f2bfe5abb.jpg'
    img = image_preprocess_test(img_path).type(torch.FloatTensor).unsqueeze(0)
    if CUDA:
        img = img.to(device)

    st = time.time()
    outputs = model(img)
    ed = time.time()
    FPS += 1 / (ed - st)

    predboxes = []
    for j, output in enumerate(outputs):
        l_size = input_shape // strides[j]
        output = output.view(1, 5 + num_classes, l_size, l_size).permute(0, 2, 3, 1).contiguous()
        output[..., :2] = torch.sigmoid(output[..., :2])
        output[..., 4:] = torch.sigmoid(output[..., 4:])
        output = decode(output, l_size, strides[j])
        predboxes.append(output.view(1, l_size * l_size, 5 + num_classes))

    prediction = torch.cat(predboxes, dim=1).view(-1, 5 + num_classes)
    # print(prediction[..., 4])
    # print(prediction.size(0))
    # print(torch.sum(prediction[..., 4]<conf_hold))
    # # print(pred_boxt)
    # pred_box = torch.cat([pred_boxt, conf.unsqueeze(-1), cls.unsqueeze(-1)], dim=-1)
    '''
    过滤掉class-specific confidence score低于阈值的框
    '''
    pred_box_t = prediction[prediction[..., 4] * prediction[..., 5] >= conf_hold]
    # predboxes.append(pred_box_t.view(-1, 5 + num_classes))
    '''
    对过滤之后得到的框进行非极大抑制得到最后的预测框
    '''
    # boxes_pred = torch.cat([predboxes[0], predboxes[1], predboxes[2]], dim=0)
    boxes = Nms_yolov4_self(pred_box_t, threshold=NMS_hold)  # [m, 5] [cls, zsx, zsy, w, h]
    boxes[..., :4] *= pic_shape / input_shape
    '''
    最后得到的这个预测框可以用来进行绘制或是计算P、R等信息
    '''
    image = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for b in boxes:
        b = b.tolist()
        draw.rectangle([b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2], outline='red', width=2)
        draw.text((b[0] - b[2] / 2, b[1] - b[3] / 2), 'wheat {:.2f}'.format(b[4] * b[5] * 100), fill='red')
        # draw.text((b[0] - b[2] / 2, b[1] - b[2] / 2), '{:.2f}'.format(b[4] * b[5] * 100), fill='red', stroke_width=1)

    image.save('/home/b201/gzx/yolox_self/test3.jpg')
    print('FPS为%.2f' % FPS)

else:
    for i, path in enumerate(x_test_path):
        # print(path)
        jpgname = path.split('/')[-1]
        img = image_preprocess_test(path).type(torch.FloatTensor).unsqueeze(0)
        # print(img.shape)
        if CUDA:
            img = img.to(device)

        st = time.time()
        outputs = model(img)
        ed = time.time()
        FPS += 1/(ed-st)

        predboxes = []
        for j, output in enumerate(outputs):
            l_size = input_shape//strides[j]
            output = output.view(1, 5 + num_classes, l_size, l_size).permute(0, 2, 3, 1).contiguous()
            output[..., :2] = torch.sigmoid(output[..., :2])
            output[..., 4:] = torch.sigmoid(output[..., 4:])
            output = decode(output, l_size, strides[j])
            predboxes.append(output.view(1, l_size*l_size, 5 + num_classes))

        prediction = torch.cat(predboxes, dim=1).view(-1, 5+num_classes)
        # print(prediction[..., 4])
        # print(prediction.size(0))
        # print(torch.sum(prediction[..., 4]<conf_hold))
        # # print(pred_boxt)
        # pred_box = torch.cat([pred_boxt, conf.unsqueeze(-1), cls.unsqueeze(-1)], dim=-1)
        '''
        过滤掉class-specific confidence score低于阈值的框
        '''
        pred_box_t = prediction[prediction[..., 4]*prediction[..., 5] >= conf_hold]
        # predboxes.append(pred_box_t.view(-1, 5 + num_classes))
        '''
        对过滤之后得到的框进行非极大抑制得到最后的预测框
        '''
        # boxes_pred = torch.cat([predboxes[0], predboxes[1], predboxes[2]], dim=0)
        boxes = Nms_yolov4_self(pred_box_t, threshold=NMS_hold) # [m, 5] [cls, zsx, zsy, w, h]
        boxes[..., :4] *= pic_shape/input_shape
        '''
        最后得到的这个预测框可以用来进行绘制或是计算P、R等信息
        '''
        image = Image.open(path).convert('RGB')
        draw = ImageDraw.Draw(image)
        for b in boxes:
            b = b.tolist()
            draw.rectangle([b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2], outline='red', width=2)
            draw.text((b[0]-b[2]/2, b[1]-b[3]/2), 'wheat {:.2f}'.format(b[4]*b[5]*100), fill='red')

        image.save('/home/b201/gzx/yolox_self/predict_img/' + jpgname)

    FPS /= len(x_test_path)
    print('test样本上的平均帧率FPS为%.2f' % FPS)

