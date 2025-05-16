import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from utils.utils import *
import os


class FRCNNDataset(Dataset):
    def __init__(self, image_root, target_image_root, annotation_lines, input_shape=[256, 256], train=True):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train

        self.trainsize = 256
        self.original_images = [image_root + f for f in os.listdir(image_root) if
                                f.endswith('.jpg') or f.endswith('.png')]  # 获取图片文件路径列表
        self.target_images = [target_image_root + f for f in os.listdir(target_image_root) if
                              f.endswith('.jpg') or f.endswith('.png')]

        self.original_images = sorted(self.original_images)  # 排序图片文件路径列表
        self.target_images = sorted(self.target_images)
        self.size = len(self.original_images)  # 数据集大小，即图片的数量
        self.img_transform = transforms.Compose([  # 图片的数据转换操作
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4975], [0.1825])
        ])

    def __len__(self):
        return self.length

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __getitem__(self, index):
        index = index % self.length
        original_images = self.binary_loader(self.original_images[index])
        target_image = self.binary_loader(self.target_images[index])
        _, y = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random=self.train)
        box_data = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box = box_data[:, :4]
        label = box_data[:, -1]

        original_images, target_image = cv_random_flip(original_images, target_image)  # 图像左右翻转
        original_images, target_image = randomCrop(original_images, target_image)  # 随机生成裁剪窗口宽度
        original_images, target_image = randomRotation(original_images, target_image)  # 随机旋转
        original_images = self.img_transform(original_images)
        target_image = self.img_transform(target_image)
        return original_images, target_image, box, label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box


def frcnn_dataset_collate(batch):
    original = []
    target = []
    bboxes = []
    labels = []
    for original_images, target_image, box, label in batch:
        original.append(original_images)
        target.append(target_image)
        bboxes.append(box)
        labels.append(label)
    original = torch.from_numpy(np.array(original))
    target = torch.from_numpy(np.array(target))
    return original, target, bboxes, labels
