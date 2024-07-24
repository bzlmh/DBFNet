import cv2
import os

# 定义文件夹路径
folder_path = "gt"

# 新文件夹路径
output_folder_path = "fin_out"

# 创建新文件夹
os.makedirs(output_folder_path, exist_ok=True)

# 获取文件夹中所有文件的列表
file_list = os.listdir(folder_path)

# 循环处理每个文件
for file_name in file_list:
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file_name)

    # 读取图片
    img = cv2.imread(file_path)

    # 转成灰度图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化
    ret, img_binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

    # 构建保存路径，保存到新文件夹下，文件名不变
    save_path = os.path.join(output_folder_path, file_name)

    # 保存处理后的图像
    cv2.imwrite(save_path, img_binary)

print("Images processed and saved successfully.")
