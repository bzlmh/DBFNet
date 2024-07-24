import os
import cv2
import numpy as np
from PIL import Image

# 定义函数 get_gt_sobel(gt)，用于对输入图像 gt 进行 Sobel 边缘检测，并返回处理后的图像
def get_gt_sobel(gt):
    gt = np.asarray(gt).astype(np.uint8)  # 将输入图像转换为 NumPy 数组并转换数据类型为 uint8
    x = cv2.Sobel(gt, cv2.CV_16S, 1, 0)  # 对输入图像进行水平方向的 Sobel 边缘检测
    y = cv2.Sobel(gt, cv2.CV_16S, 0, 1)  # 对输入图像进行垂直方向的 Sobel 边缘检测
    absX = cv2.convertScaleAbs(x)  # 将水平方向的 Sobel 结果转换为绝对值图像
    absY = cv2.convertScaleAbs(y)  # 将垂直方向的 Sobel 结果转换为绝对值图像
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 将水平和垂直方向的 Sobel 结果加权合并
    return Image.fromarray(dst)  # 将合并后的图像转换为 PIL Image 对象并返回

# 定义函数 get_ostu(img)，用于对输入图像 img 进行 Otsu 二值化处理，并返回处理后的图像
def get_ostu(img):
    img = np.asarray(img).astype(np.uint8)  # 将输入图像转换为 NumPy 数组并转换数据类型为 uint8
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 使用 Otsu 方法进行二值化处理
    return Image.fromarray(th2)  # 将处理后的图像转换为 PIL Image 对象并返回

# 定义要处理的图像文件夹路径
img_folder = "../datasets/test/img"

# 创建保存结果的文件夹路径
otsu_folder = "../datasets/otsu"
sobel_folder = "../datasets/sobel"
os.makedirs(otsu_folder, exist_ok=True)
os.makedirs(sobel_folder, exist_ok=True)

# 遍历图像文件夹中的每个图像文件
for filename in os.listdir(img_folder):
    # 构建图像文件的完整路径
    img_path = os.path.join(img_folder, filename)

    # 打开图像文件
    img = Image.open(img_path)

    # 对图像进行 Otsu 二值化处理
    img_gray = img.convert('L')
    img_gray_np = np.asarray(img_gray).astype(np.uint8)
    ret2, th2 = cv2.threshold(img_gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_img = Image.fromarray(th2)

    # 对图像进行 Sobel 边缘检测处理
    gt = np.asarray(img).astype(np.uint8)
    x = cv2.Sobel(gt, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gt, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    sobel_img = Image.fromarray(dst)

    # 构建保存结果的文件路径
    otsu_save_path = os.path.join(otsu_folder, filename)
    sobel_save_path = os.path.join(sobel_folder, filename)

    # 保存处理后的图像
    otsu_img.save(otsu_save_path)
    sobel_img.save(sobel_save_path)
