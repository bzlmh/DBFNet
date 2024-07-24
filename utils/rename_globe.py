# import os
#
# # 指定要修改文件名的文件夹路径
# base_directory = '../globe'
#
# # 列出base_directory下的所有子文件夹
# subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if
#                   os.path.isdir(os.path.join(base_directory, d))]
#
# # 循环处理每个子文件夹
# for subdirectory in subdirectories:
#     # 列出子文件夹下所有文件
#     files = os.listdir(subdirectory)
#
#     # 循环处理每个文件名
#     for filename in files:
#         # 检查文件名是否以'H'开头
#         if filename.startswith('H'):
#             # 生成新的文件名
#             new_filename = 'globe_' + filename
#             # 重命名文件
#             os.rename(os.path.join(subdirectory, filename), os.path.join(subdirectory, new_filename))
#             print(f'Renamed {filename} to {new_filename} in {subdirectory}')
from PIL import Image
import os

# 指定原始文件夹路径和目标文件夹路径
base_directory = '../globe'
target_directory = '../resized_images'

# 创建目标文件夹
os.makedirs(target_directory, exist_ok=True)

# 列出原始文件夹下的所有子文件夹
subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if
                  os.path.isdir(os.path.join(base_directory, d))]

# 循环处理每个子文件夹
for subdirectory in subdirectories:
    # 创建子文件夹在目标文件夹下
    target_subdirectory = os.path.join(target_directory, os.path.basename(subdirectory))
    os.makedirs(target_subdirectory, exist_ok=True)

    # 列出子文件夹下所有文件
    files = os.listdir(subdirectory)

    # 循环处理每个文件名
    for filename in files:
        # 检查文件名是否以'.jpg'结尾，如果不是则跳过
        if not filename.lower().endswith('.jpg'):
            continue

        # 打开图片
        img_path = os.path.join(subdirectory, filename)
        img = Image.open(img_path)

        # resize图片
        resized_img = img.resize((256, 256))

        # 生成新的文件名
        new_filename = 'resized_' + filename

        # 保存resize后的图片到目标文件夹下
        resized_img.save(os.path.join(target_subdirectory, new_filename))
        print(f'Resized and saved {filename} to {new_filename} in {target_subdirectory}')
