import cv2
import os
from tqdm import tqdm
import numpy as np


# 创建输出文件夹
def create_output_folders(output_folder1, output_folder2):
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)


# 处理图像并保存到指定文件夹
def process_images(input_folder, prewitt_output_folder, sobel_output_folder):
    # 创建输出文件夹
    create_output_folders(prewitt_output_folder, sobel_output_folder)

    # 获取输入文件夹中的图像文件列表
    image_files = os.listdir(input_folder)

    # 使用 tqdm 显示处理进度
    for filename in tqdm(image_files, desc='Processing images', unit='image'):
        input_file = os.path.join(input_folder, filename)
        output_file_prewitt = os.path.join(prewitt_output_folder, filename)
        output_file_sobel = os.path.join(sobel_output_folder, filename)

        img = cv2.imread(input_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用Prewitt算子边缘检测算法
        prewitt_kernel_x = cv2.getDerivKernels(1, 0, 3)
        prewitt_kernel_y = cv2.getDerivKernels(0, 1, 3)
        prewitt_x = cv2.filter2D(gray, -1, prewitt_kernel_x[0] * prewitt_kernel_y[0].T)
        prewitt_y = cv2.filter2D(gray, -1, prewitt_kernel_x[1] * prewitt_kernel_y[1].T)

        # 将 prewitt_x 和 prewitt_y 转换为浮点类型
        prewitt_x = prewitt_x.astype(np.float32)
        prewitt_y = prewitt_y.astype(np.float32)

        prewitt_img = cv2.magnitude(prewitt_x, prewitt_y)

        cv2.imwrite(output_file_prewitt, prewitt_img)

        # 使用Sobel算子边缘检测算法
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_img = cv2.magnitude(sobel_x, sobel_y)

        cv2.imwrite(output_file_sobel, sobel_img)


if __name__ == "__main__":
    input_folder = "../datasets/test/img"  # 输入文件夹
    prewitt_output_folder = "prewitt"  # 输出到Prewitt文件夹
    sobel_output_folder = "sobel"  # 输出到Sobel文件夹

    process_images(input_folder, prewitt_output_folder, sobel_output_folder)
