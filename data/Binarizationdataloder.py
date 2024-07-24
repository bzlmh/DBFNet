from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random
from PIL import Image, ImageOps
from torchvision.transforms import Compose, ToTensor, Resize
from utils.utils import split_ext, correct_size
import torch

def get_gt_sobel(gt):
    gt = np.asarray(gt).astype(np.uint8)
    x = cv2.Sobel(gt, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gt, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Image.fromarray(dst)

def get_ostu(img):
    img = np.asarray(img).astype(np.uint8)
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(th2)

def get_file_name(filepath):
    name = os.listdir(filepath)
    name = [os.path.join(filepath, i) for i in name if os.path.splitext(i)[1] in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP', '.tiff', '.TIFF', '.tif', '.TIF']]
    return len(name), name

def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def color_jitter(imgs):
    # 随机变化图像的亮度、对比度、饱和度和色调
    brightness = np.random.uniform(0.8, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    saturation = np.random.uniform(0.8, 1.2)
    hue = np.random.uniform(-0.1, 0.1)

    # 将图像转换为 HSV 色彩空间
    img_hsv = cv2.cvtColor(imgs, cv2.COLOR_RGB2HSV)
    img_hsv = img_hsv.astype(np.float32)

    # 调整亮度、对比度、饱和度和色调
    img_hsv[:, :, 2] *= brightness
    img_hsv[:, :, 1] *= contrast
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    img_hsv[:, :, 0] *= saturation
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 255)
    img_hsv[:, :, 0] += hue * 255
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 255)

    # 将图像重新转换为 RGB 色彩空间
    imgs = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 将 numpy 数组转换为 PIL 图像对象
    imgs = Image.fromarray(imgs)

    return imgs


def random_rotate(imgs, color, angle):
    img = np.array(imgs).astype(np.uint8)
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), borderValue=color)
    img = Image.fromarray(img_rotation)
    return img

def ImageTransform(loadSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
        ToTensor(),
    ])

def ImageTransform_fullsize():
    return Compose(
        [ToTensor()]
    )

class DocData(Dataset):
    def __init__(self, imgRoot, loadsize, training=True, only_one=False):
        super(DocData, self).__init__()
        self.number, self.ImgFiles = get_file_name(imgRoot)
        self.loadsize = loadsize
        self.training = training
        self.Imgtrans = ImageTransform(loadsize)
        self.Imgtrans_fullsize = ImageTransform_fullsize()
        self.dataOriPath = r'./datasets'
        self.only_one = only_one

    def __getitem__(self, index):
        dataRoot, (dataName, dataExt) = split_ext(self.ImgFiles[index])
        dataName = dataName.split('_')
        if not self.only_one:
            if dataName[0] != 'H':
                if dataName[0] == 'DIBCO':
                    datasetsName = '_'.join(dataName[0:2])
                    img_x = dataName[-2]
                    img_y = dataName[-1]
                    imgName = '_'.join(dataName[2:-2])
                else:
                    datasetsName = dataName[0]
                    img_x = dataName[-2]
                    img_y = dataName[-1]
                    imgName = '_'.join(dataName[1:-2])
            else:
                datasetsName = '_'.join(dataName[0:3])
                img_x = dataName[-2]
                img_y = dataName[-1]
                imgName = '_'.join(dataName[3:-2])
        else:
            datasetsName = os.path.basename(os.path.dirname(dataRoot))
            img_x = dataName[-2]
            img_y = dataName[-1]
            imgName = '_'.join(dataName[0:-2])

        imgOriRoot = os.path.join(self.dataOriPath, os.path.join(datasetsName, 'img'))
        imgName_with_coordinates = f"{imgName}_{img_x}_{img_y}"
        imgRoot = os.path.join(imgOriRoot, imgName_with_coordinates + dataExt)

        img = Image.open(self.ImgFiles[index]).convert('RGB')
        gt = Image.open(self.ImgFiles[index].replace('img', 'gt')).convert('L')
        ostu = Image.open(self.ImgFiles[index].replace('img', 'ostu')).convert('L')
        sobel = Image.open(self.ImgFiles[index].replace('img', 'sobel')).convert('L')
        prewitt = Image.open(self.ImgFiles[index].replace('img', 'prewitt')).convert('L')
        gray = img.convert('L')
        gt_Sobel = get_gt_sobel(gt).convert('L')
        if self.training:
            # 旋转
            if random.random() < 0.3:
                max_angle = 10
                angle = random.random() * 2 * max_angle - max_angle
                img = random_rotate(img, 0, angle)
                ostu = random_rotate(ostu, (255, 255, 255), angle)
                prewitt = random_rotate(prewitt, 0, angle)
                gt = random_rotate(gt, (255, 255, 255), angle)
            # 颜色抖动
            if random.random() < 0.5:
                img = color_jitter(np.array(img))
        inputImg = self.Imgtrans(img)
        ostu = self.Imgtrans(ostu)
        sobel = self.Imgtrans(sobel)
        gt = self.Imgtrans(gt)
        gray = self.Imgtrans(gray)
        prewitt = self.Imgtrans(prewitt)
        gt_Sobel = self.Imgtrans(gt_Sobel)
        path, name = os.path.split(self.ImgFiles[index])
        return inputImg, ostu, sobel, gt, gt_Sobel, gray, prewitt, img_x, img_y, name

    def __len__(self):
        return len(self.ImgFiles)


class TestData(Dataset):
    def __init__(self, imgRoot, loadsize, training=True, only_one=False):
        super(TestData, self).__init__()  # 调用父类的构造方法
        self.number, self.ImgFiles = get_file_name(imgRoot)  # 获取图像文件数量和文件名列表
        self.loadsize = loadsize  # 设置加载大小
        self.training = training  # 设置是否为训练模式
        self.Imgtrans = ImageTransform(loadsize)  # 设置图像转换序列（包括缩放和转换为张量）  为了resize
        self.Imgtrans_fullsize = ImageTransform_fullsize()  # 设置图像转换序列（只转换为张量，不进行缩放）   只转换为张量
        self.dataOriPath = r'./datasets'  # 设置原始数据集路径
        self.only_one = only_one  # 设置是否只包含一个类别

    # 定义 __getitem__ 方法，用于按索引加载单个样本，并进行数据增强和转换
    def __getitem__(self, index):
        dataRoot, (dataName, dataExt) = split_ext(self.ImgFiles[index])  # 拆分文件名和扩展名
        dataName = dataName.split('_')  # 拆分文件名中的各个部分
        # 根据是否只包含一个类别来确定数据集的名称和图像名称
        if not self.only_one:
            if dataName[0] != 'H':
                if dataName[0] == 'DIBCO':
                    datasetsName = '_'.join(dataName[0:2])
                    img_x = dataName[-2]
                    img_y = dataName[-1]
                    imgName = '_'.join(dataName[2:-2])
                else:
                    datasetsName = dataName[0]
                    img_x = dataName[-2]
                    img_y = dataName[-1]
                    imgName = '_'.join(dataName[1:-2])
            else:
                datasetsName = '_'.join(dataName[0:3])
                img_x = dataName[-2]
                img_y = dataName[-1]
                imgName = '_'.join(dataName[3:-2])
        else:
            datasetsName = os.path.basename(os.path.dirname(dataRoot))
            img_x = dataName[-2]
            img_y = dataName[-1]
            imgName = '_'.join(dataName[0:-2])

        imgOriRoot = os.path.join(self.dataOriPath, os.path.join(datasetsName, 'img'))  # 构建原始图像路径
        imgName_with_coordinates = f"{imgName}_{img_x}_{img_y}"
        imgRoot = os.path.join(imgOriRoot, imgName_with_coordinates + dataExt)  # 构建图像文件完整路径
        # 输入模型里的就这6张图片  原始图片，真值图片，ostu图片，sobel图片，灰度图片，真值的sobel图片
        img = Image.open(self.ImgFiles[index]).convert('RGB')  # 打开当前图像文件并转换为 RGB 模式的 PIL Image 对象
        ostu = Image.open(self.ImgFiles[index].replace('img', 'ostu')).convert('L')  # 打开对应的 Otsu 二值化图像文件并转换为灰度图
        sobel = Image.open(self.ImgFiles[index].replace('img', 'sobel')).convert('L')  # 打开对应的 Sobel 边缘检测图像文件并转换为灰度图
        prewitt = Image.open(self.ImgFiles[index].replace('img', 'prewitt')).convert('L')  # 打开对应的 Sobel 边缘检测图像文件并转换为灰度图

        # 如果处于训练模式且随机数小于 0.3，则进行数据增强
        if self.training and random.random() < 0.3:
            max_angle = 10  # 最大旋转角度
            angle = random.random() * 2 * max_angle - max_angle  # 随机生成旋转角度
            # 对图像及其相关图像进行随机角度旋转
            img = random_rotate(img, 0, angle)
            ostu = random_rotate(ostu, (255, 255, 255), angle)
            sobel = random_rotate(sobel, 0, angle)

        inputImg = self.Imgtrans(img)  # 将当前图像转换为张量
        ostu = self.Imgtrans(ostu)  # 将当前 Otsu 二值化图像转换为张量
        sobel = self.Imgtrans(sobel)  # 将当前 Sobel 边缘检测图像转换为张量
        prewitt=self.Imgtrans(prewitt)
        # 将 Sobel 边缘检测后的 ground truth 图像转换为张量
        path, name = os.path.split(self.ImgFiles[index])  # 拆分图像文件的路径和文件名
        return inputImg, ostu, sobel, img_x, img_y, name,prewitt

    def __len__(self):
        return self.number  # 返回数据集的长度