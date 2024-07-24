import os
from PIL import Image


def convert_to_jpg(dataset_path, folder_names):
    # 遍历每个文件夹
    for folder_name in folder_names:
        folder_path = os.path.join(dataset_path, folder_name)
        # 获取文件列表
        files = os.listdir(folder_path)
        # 遍历文件列表进行格式转换
        for file in files:
            # 拼接文件路径
            file_path = os.path.join(folder_path, file)
            # 检查文件是否是图片文件
            if os.path.isfile(file_path):
                # 提取文件名和扩展名
                file_name, file_ext = os.path.splitext(file)
                # 如果不是 JPG 格式就进行转换
                if file_ext.lower() != '.jpg':
                    img = Image.open(file_path)
                    # 保存为 JPG 格式
                    img.save(os.path.join(folder_path, f"{file_name}.jpg"))
                    # 尝试删除原文件
                    try:
                        os.remove(file_path)
                    except PermissionError as e:
                        print(f"Error deleting file {file}: {e}")
            else:
                print(f"File {file} not found.")


# 要转换格式的文件夹名称
folder_names = ['img', 'gt']

# 转换格式，传入数据集路径和要转换格式的文件夹名称列表
convert_to_jpg("../datasets/HDIBCO", folder_names)
