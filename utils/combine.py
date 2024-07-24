import os
import cv2
import numpy as np


def add_alpha_channel(image):
    """
    将图像添加 alpha 通道，如果图像已经有 alpha 通道，则不进行修改。
    """
    if image.shape[2] == 3:  # 如果图像没有 alpha 通道，则添加一个全白的 alpha 通道
        alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 255
        image = np.concatenate((image, alpha_channel), axis=2)
    return image


def reconstruct_images(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像块文件，并按文件名排序
    block_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

    # 检查是否存在拼接图像块
    if not block_files:
        print("No image blocks found in the input folder.")
        return

    # 提取文件名中的行和列索引
    row_col_map = {}
    for block_file in block_files:
        parts = os.path.splitext(block_file)[0].split('_')
        filename = parts[0]
        row_col = tuple(map(int, parts[1:3]))
        if filename not in row_col_map:
            row_col_map[filename] = []
        row_col_map[filename].append(row_col)

    # 重构图像
    for filename, row_col_list in row_col_map.items():
        # 确定输出图像的行数和列数
        max_row = max(row for row, _ in row_col_list) + 1
        max_col = max(col for _, col in row_col_list) + 1

        # 初始化拼接后的图像
        reconstructed_image = None

        # 遍历每个小块，将其放置到相应位置
        for row, col in row_col_list:
            block_path = os.path.join(input_folder, f"{filename}_{row}_{col}.png")
            block = cv2.imread(block_path, cv2.IMREAD_UNCHANGED)

            # 如果图像没有 alpha 通道，则添加一个全白的 alpha 通道
            block = add_alpha_channel(block)

            if reconstructed_image is None:
                # 初始化拼接后的图像
                block_height, block_width = block.shape[:2]
                reconstructed_image = np.zeros((block_height * max_row, block_width * max_col, 4), dtype=np.uint8)

            x_start = col * block.shape[1]
            x_end = x_start + block.shape[1]
            y_start = row * block.shape[0]
            y_end = y_start + block.shape[0]

            reconstructed_image[y_start:y_end, x_start:x_end] = block

        # 保存重构后的图像
        output_image_path = os.path.join(output_folder, f"{filename}.png")
        cv2.imwrite(output_image_path, reconstructed_image)

        print(f"Reconstructed image saved: {output_image_path}")


# 指定输入文件夹和输出文件夹
input_folder = "../datasets/HDIBCO/gt"  # 更改为你的输入文件夹路径
output_folder = "gt"  # 更改为你的输出文件夹路径

# 调用函数进行图像重构
reconstruct_images(input_folder, output_folder)
