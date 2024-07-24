# 切割不够的地方填充空白
import os
import cv2
import numpy as np

def pad_image(image, target_height, target_width):
    """
    使用填充对图像进行调整，使其达到目标尺寸。
    """
    height, width = image.shape[:2]

    # 计算要添加的填充大小
    pad_height = target_height - height
    pad_width = target_width - width

    # 使用空白填充
    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # 将原始图像放置在左上角
    padded_image[:height, :width] = image

    return padded_image

def crop_images(input_folders, output_folders, block_size=(256, 256)):
    # 遍历输入文件夹列表
    for input_folder, output_folder in zip(input_folders, output_folders):
        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 获取输入文件夹中的所有图像文件
        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

        for image_file in image_files:
            # 读取图像
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)

            # 获取图像大小
            height, width = image.shape[:2]

            # 计算要切割的行数和列数
            rows = (height + block_size[1] - 1) // block_size[1]
            cols = (width + block_size[0] - 1) // block_size[0]

            # 遍历图像并切割成指定大小的块
            for i in range(rows):
                for j in range(cols):
                    # 计算当前块的位置
                    x_start = j * block_size[0]
                    y_start = i * block_size[1]
                    x_end = min(x_start + block_size[0], width)
                    y_end = min(y_start + block_size[1], height)

                    # 切割图像块
                    block = image[y_start:y_end, x_start:x_end]

                    # 对图像块进行填充，使其达到目标大小
                    if j == cols - 1:  # 最右侧的小块
                        block = pad_image(block, block_size[1], block_size[0])
                    if i == rows - 1:  # 最下侧的小块
                        block = pad_image(block, block_size[1], block_size[0])

                    # 生成输出文件名
                    output_filename = f"{os.path.splitext(image_file)[0]}_{i}_{j}.png"
                    output_path = os.path.join(output_folder, output_filename)

                    # 保存图像块
                    cv2.imwrite(output_path, block)

                    print(f"Saved {output_filename}")

# 指定输入和输出文件夹列表
input_folders = ["../datasets/test/gt","../datasets/test/otsu","../datasets/test/img","../datasets/test/sobel","../datasets/test/prewitt"]
output_folders = ["test/slice/slice/gt","test/slice/slice/otsu","test/slice/slice/image","test/slice/slice/sobel","test/slice/slice/prewitt"]

# 调用函数进行图像切割
crop_images(input_folders, output_folders)
