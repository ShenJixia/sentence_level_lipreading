import os
import cv2
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import dlib

# dlib核心
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# 遍历jpg图片，生成列表，每个元素是jpg图片的文件夹地址
def list_jpg_files(root_folder):
    jpg_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files


class JPGDataset(Dataset):  # 用class创建了一个名为JPGDataset的类，继承自pytorch中的 Dataset 类
    def __init__(self, root_folder):  # __init__是Dataset类的一个方法，至少要传入文件路径
        self.root_folder = root_folder  # 为了在类的不同方法之间共享这些成员变量，你需要使用 self 来定义它们并访问它们。
        self.jpg_files = list_jpg_files(root_folder)
        # 筛选还没有截取唇图的
        self.new_jpg_files = []  # 读取
        self.out_files = []  # 写入
        for jpg_path in self.jpg_files:
            out_path = jpg_path.replace(r'frame', r'lip')
            if not os.path.exists(out_path):
                self.new_jpg_files.append(jpg_path)
                self.out_files.append(out_path)
                output_folder, filename = os.path.split(out_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
        print(self.new_jpg_files)
        print(len(self.new_jpg_files))

    def __len__(self):  # 返回样本数量
        return len(self.new_jpg_files)

    def __getitem__(self, idx):
        jpg_file = self.new_jpg_files[idx]
        print(jpg_file)
        output_files = self.out_files[idx]
        print("Loading data for index:", idx)
        # 读取图片
        image = cv2.imread(jpg_file)
        if image is not None:  # 存在部分 图片读取会出问题
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 人脸数rects
            rects = detector(img, 0)  # 使用面部检测器 detector 检测人脸
            if len(rects) == 1:
                landmarks = np.array([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
                center = np.mean(landmarks[49:68], axis=0)
                w, h = 100, 50
                a = int(center[0] - w / 2)
                b = int(center[1] - h / 2)
                H, W = img.shape[0:2]
                crop = img[max(b, 0): min(H, b + h), max(a, 0): min(W, a + w)]
                cv2.imwrite(output_files, crop)
            else:
                # 如果没有有效的数据，引发异常
                # raise ValueError("No valid data available for index {}".format(idx))
                print("No valid data available for index:", idx)
                return []
        else:
            print(jpg_file,':加载图片出错')
            return []
        return []


if __name__ == "__main__":
    t1 = time.time()
    root_folder = r"E:\dataset\GRID"
    dataset = JPGDataset(root_folder)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=0, drop_last=False)
    # batch_size：指定每个小批次的样本数量
    # shuffle：布尔值，表示是否在每个周期之前对数据进行随机洗牌
    # num_workers：指定用于数据加载的工作线程数，建议设置为CPU核心数的一半
    # pin_memory：布尔值，如果设置为True，则将数据加载到CUDA固定内存中
    # drop_last：布尔值，当数据集的大小不能整除batch_size时，会出现最后一个小批次的样本数量小于batch_size的情况。设置为False（默认值），保留。为True，丢弃。
    for idx, data in enumerate(dataloader):  # 前面只是进行了创建，需要用for循环来进行迭代操作
        pass
    t2 = time.time()
    print('time cost:', t2 - t1)