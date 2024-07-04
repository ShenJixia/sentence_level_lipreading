import os

def find_duplicate_files(folder):
    file_dict = {}  # 用于记录文件名的字典
    duplicate_files = {}  # 用于存储重复文件的字典

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            # 检查文件名是否已经出现过
            if file in file_dict:
                # 如果文件名已经出现过，则将其添加到重复文件字典中
                if file in duplicate_files:
                    duplicate_files[file].append(file_path)
                else:
                    duplicate_files[file] = [file_path]
            else:
                # 如果文件名是第一次出现，则将其添加到文件字典中
                file_dict[file] = file_path

    return duplicate_files

# 指定要检查的文件夹路径
folder_path = r'E:\dataset\GRID\video'

# 找到重复文件
duplicate_files = find_duplicate_files(folder_path)

# 打印重复文件信息
if duplicate_files:
    print("重复文件：")
    for file_name, file_paths in duplicate_files.items():
        print(f"文件名：{file_name}")
        print(f"重复文件路径：{file_paths}")
else:
    print("没有重复文件。")
