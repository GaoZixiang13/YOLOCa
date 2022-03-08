import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import math

class yolodataset(Dataset):
    def __init__(self, images, targets, input_shape, num_classes, train=True, cuda=True):
        super(yolodataset, self).__init__()
        self.images = images
        self.targets = targets
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cuda = cuda
        self.eps = 1e-6
        self.train = train

    def __getitem__(self, index):
        tx = self.images[index]
        ty = self.targets[index]
        ty = np.array(ty)
        if self.train:
            x, ty, height, width = self.image_preprocess(tx, ty)
        else:
            x, ty, height, width = self.image_preprocess_test(tx, ty)

        y = self.getlabels(ty, height, width)

        return x, y

    def __len__(self):
        return len(self.images)


    def getlabels(self, labels, height, width):
        '''
        :param labels: python_list --> [num_gt, 5]
        :param label: python_list --> [x, y, w, h, cls]
        :return:
        '''
        # print(torch.tensor(labels).shape)

        if self.input_shape%32 != 0:
            raise RuntimeError('input shape can not be Divisible by 32')
        y_true = torch.zeros(300, 5 + self.num_classes).type(torch.FloatTensor).tolist()
        if labels.shape[0] == 0:
            return torch.tensor(y_true).type(torch.FloatTensor)

        # print(labels.shape)
        labels[:, 0], labels[:, 2] = labels[:, 0] * self.input_shape / width, labels[:, 2] * self.input_shape / width
        labels[:, 1], labels[:, 3] = labels[:, 1] * self.input_shape / height, labels[:, 3] * self.input_shape / height

        labels = labels.tolist()
        for label_i, label in enumerate(labels):
            y_true[label_i][0] = label[0]
            y_true[label_i][1] = label[1]
            y_true[label_i][2] = label[2]
            y_true[label_i][3] = label[3]
            y_true[label_i][4] = 1
            y_true[label_i][5 + int(label[4])] = 1

        return torch.tensor(y_true).type(torch.FloatTensor)

    def image_preprocess(self, img_path, labels):

        img = Image.open(img_path).convert('RGB')
        height, width = img.size
        # --------------------------------------------------------
        # normal data augmentation
        img = transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2)(img)
        img = transforms.RandomAdjustSharpness(0.5, p=0.3)(img)
        img = transforms.RandomAdjustSharpness(1.5, p=0.3)(img)
        img, labels = self.mirror_h(img, labels)
        img, labels = self.randomRotation(img, labels)
        # --------------------------------------------------------
        # strong data augmentation including Mosaic, CutMix etc.

        # --------------------------------------------------------
        img = img.resize((self.input_shape, self.input_shape), Image.BICUBIC)

        img = np.transpose(np.array(img)/255., (2, 0, 1))
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

        return img_tensor, labels, height, width

    def image_preprocess_test(self, img_path, labels):
        img = Image.open(img_path).convert('RGB')
        height, width = img.size

        img = img.resize((self.input_shape, self.input_shape), Image.BICUBIC)
        img = np.transpose(np.array(img) / 255., (2, 0, 1))
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

        return img_tensor, labels, height, width

    def mirror_h(self, img, label, prob=0.5):
        '''
        实现水平翻转图像
        '''
        rd = np.random.rand()
        if rd >= prob:
            height, width = img.size
            img = transforms.RandomHorizontalFlip(p=1)(img)
            if label.shape[0] > 0:
                label[:, 0] = width - label[:, 0]

        return img, label

    def basic_Rotation_90(self, img, label):
        '''
        右旋90度，y1=x0, x1=height-y0.
        '''
        height, width = img.size
        img = transforms.RandomRotation(degrees=(-90, -90))(img)
        if label.shape[0] > 0:
            x0, y0, w0, h0 = np.array(label[:, 0]), np.array(label[:, 1]), np.array(label[:, 2]), np.array(label[:, 3])
            label[:, 0], label[:, 1] = height - y0, x0
            label[:, 2], label[:, 3] = h0, w0

        return img, label

    def basic_Rotation90(self, img, label):
        '''
        左旋90度，y1=x0, x1=height-y0.
        '''
        height, width = img.size
        img = transforms.RandomRotation(degrees=(90, 90))(img)
        if label.shape[0] > 0:
            x0, y0, w0, h0 = np.array(label[:, 0]), np.array(label[:, 1]), np.array(label[:, 2]), np.array(label[:, 3])
            label[:, 0], label[:, 1] = y0, width - x0
            label[:, 2], label[:, 3] = h0, w0

        return img, label

    def randomRotation(self, img, label, prob=(.25, .5, .75)):
        '''
        实现随机垂直旋转
        '''
        rd = np.random.rand()

        if rd >= prob[0] and rd < prob[1]:
            img, label = self.basic_Rotation_90(img, label)

        elif rd >= prob[1] and rd < prob[2]:
            for i in range(2):
                img, label = self.basic_Rotation_90(img, label)

        elif rd >= prob[2]:
            img, label = self.basic_Rotation90(img, label)

        return img, label

