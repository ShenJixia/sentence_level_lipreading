import os

# 输入要遍历的文件类型
# file_type = '.mp4'
file_type = '.jpg'
# file_type = '.mpg'
# file_type = '.align'
# file_type = '.txt'
# file_type = '.txt'

# 输入要遍历的目标路径
root_folder = r"E:\dataset\GRID\lips"


def count_xxx_files(root_folder):
    xxx_count = 0

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(file_type):
                xxx_count += 1

    return xxx_count


xxx_count = count_xxx_files(root_folder)
print(f"目标路径内（包含子文件夹）有 {xxx_count} 个 {file_type} 文件。")
