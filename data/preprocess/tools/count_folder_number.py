import os


def count_folders(root_folder):
    subfolder_count = 0
    next_layer_count = 0

    for root, dirs, files in os.walk(root_folder):
        if root == root_folder:
            next_layer_count = len(dirs)
        subfolder_count += len(dirs)
    return next_layer_count, subfolder_count


root_folder = r"E:\dataset\GRID\lips"
next_count, total_count = count_folders(root_folder)
print(f"目标路径下一层有 {next_count} 个文件夹。")
print(f"目标路径包含子文件夹共有 {total_count} 个子文件夹。")
