import os

import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from utils.utils import *
import cv2


# several data augumentation strategies
def cv_random_flip(img, target_img, label):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    return img, target_img, label


def randomCrop(image, target_image):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,

        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), target_image.crop(random_region)


def randomRotation(image, target_image):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        target_image = target_image.rotate(random_angle, mode)
    return image, target_image


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255

    return Image.fromarray(img)

# dataset for training

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, target_image_root, annotation_lines, trainsize):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = [256, 256]
        self.trainsize = trainsize
        self.train = True
        self.original_images = [image_root + f for f in os.listdir(image_root) if
                                f.endswith('.jpg') or f.endswith('.png')]
        self.target_images = [target_image_root + f for f in os.listdir(target_image_root) if
                              f.endswith('.jpg') or f.endswith('.png')]

        self.original_images = sorted(self.original_images)
        self.target_images = sorted(self.target_images)
        self.size = len(self.original_images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.6], [0.1183])
        ])
        self.target_img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4975], [0.7])
        ])

    def __getitem__(self, index):
        _, y = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random=self.train)
        box_data = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box = box_data[:, :4]
        label = box_data[:, -1]

        original_images = self.binary_loader(self.original_images[index])
        target_image = self.binary_loader(self.target_images[index])

        original_images, target_image = randomCrop(original_images, target_image)
        original_images, target_image = randomRotation(original_images, target_image)
        original_images = colorEnhance(original_images)

        original_images = self.img_transform(original_images)
        target_image = self.target_img_transform(target_image)

        return original_images, target_image, box, label

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape=[256, 256], jitter=0.3, hue=.1, sat=0.7, val=0.4,
                        random=True):
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
                box = box[np.logical_and(box_w > 1, box_h > 1)]

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

        x = np.arange(0, 256, dtype=r.dtype)
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

    def __len__(self):
        return self.size


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
    original = torch.stack(original, dim=0)
    target = torch.stack(target, dim=0)
    return original, target, bboxes, labels

# dataloader for training
def get_loader(image_root, target_image_root, annotation_lines, batchsize, trainsize, shuffle=True, num_workers=0,
               pin_memory=True):
    dataset = SalObjDataset(image_root, target_image_root, annotation_lines, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  collate_fn=frcnn_dataset_collate)
    return data_loader


def get_test_loader(image_root, target_image_root, gt_root, batchsize, testsize, shuffle=True, num_workers=0,
                    pin_memory=True):
    datasets = test_dataset(image_root, target_image_root, gt_root, testsize)
    data_loader = data.DataLoader(dataset=datasets,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset(data.Dataset):
    def __init__(self, image_root, target_image_root, gt_root, testsize):
        self.testsize = testsize
        self.original_images = [image_root + f for f in os.listdir(image_root) if
                                f.endswith('.jpg') or f.endswith('.png')]
        self.target_images = [target_image_root + f for f in os.listdir(target_image_root) if
                              f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.original_images = sorted(self.original_images)
        self.target_images = sorted(self.target_images)
        self.gts = sorted(self.gts)
        self.size = len(self.original_images)
        self.image_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4975], [0.1825])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4975], [0.1825])
        ])
        self.size = len(self.original_images)
        self.index = 0

    def load_data(self):
        original_images = self.binary_loader(self.original_images[self.index])
        original_images = self.image_transform(original_images).unsqueeze(0)

        target_images = self.binary_loader(self.target_images[self.index])
        tar = target_images
        target_images = self.image_transform(target_images).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        g = gt
        g = self.gt_transform(g)

        name = self.original_images[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size

        return original_images, target_images, tar, gt, name, g

    def __getitem__(self, index):
        original_image = self.binary_loader(self.original_images[index])
        target_image = self.binary_loader(self.target_images[index])
        gt = self.binary_loader(self.gts[index])

        original_image = self.image_transform(original_image)
        target_image = self.image_transform(target_image)
        gt = self.gt_transform(gt)
        filename = os.path.basename(self.target_images[index % len(self.target_images)])

        return {'A': target_image, 'B': original_image, "C": gt, 'filename': filename}

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
