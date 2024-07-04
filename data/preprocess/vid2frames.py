import os
import cv2
from torch.utils.data import Dataset, DataLoader
import time


# 遍历视频文件
def list_vid_files(root_folder):
    vid_path_list = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mpg'):
                vid_path_list.append(os.path.join(root, file))

    return vid_path_list


class Vid2FrameDataset(Dataset):  # 用class创建了一个名为VideoFrameDataset的类，继承自pytorch中的 Dataset 类
    def __init__(self, root_folder):  # __init__是Dataset类的一个方法，至少要传入文件路径
        self.root_folder = root_folder  # 为了在类的不同方法之间共享这些成员变量，你需要使用 self 来定义它们并访问它们。
        self.video_files = list_vid_files(root_folder)
        self.new_video_files = []
        self.out_files = []
        for video_path in self.video_files:
            out_path = video_path.replace(r'video', 'frames').replace('.mpg', '')
            # if not os.path.exists(out_path) or len(os.listdir(out_path)) < 100:
            if not os.path.exists(out_path):
                self.new_video_files.append(video_path)
                self.out_files.append(out_path)

        print(self.new_video_files)
        print(len(self.new_video_files))

    def __len__(self):  # 返回样本数量
        return len(self.new_video_files)

    def __getitem__(self, idx):
        video_path = self.new_video_files[idx]
        output_folder = self.out_files[idx]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  # 文件夹不存在，进行创建

        cap = cv2.VideoCapture(video_path)
        # 截取帧图
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"{idx}已完成")
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 保存帧图像到目标文件夹
            filename=output_folder.split('\\')[-1]
            frame_filename = os.path.join(output_folder, f"{filename}_frame{frame_number}.jpg")
            cv2.imwrite(frame_filename, gray_frame)
            frame_number += 1
        # 释放内存
        cap.release()
        return []


if __name__ == "__main__":
    t1 = time.time()
    root_folder = r"E:\dataset\GRID\video\s2"
    dataset = Vid2FrameDataset(root_folder)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)
    # batch_size：指定每个小批次的样本数量
    # shuffle：布尔值，表示是否在每个周期之前对数据进行随机洗牌
    # num_workers：指定用于数据加载的工作线程数，建议设置为CPU核心数的一半
    # pin_memory：布尔值，如果设置为True，则将数据加载到CUDA固定内存中
    # drop_last：布尔值，当数据集的大小不能整除batch_size时，会出现最后一个小批次的样本数量小于batch_size的情况。设置为False（默认值），保留。为True，丢弃。
    for idx, data in enumerate(dataloader):  # 前面只是进行了创建，需要用for循环来进行迭代操作
        pass
    t2 = time.time()
    print('time cost:', t2 - t1)
