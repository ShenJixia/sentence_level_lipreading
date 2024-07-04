import numpy as np
import glob
import cv2
import os
import editdistance
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import time

# bos  eos  unk  pad  blank(=pad)  space(=sep?)
PAD_TAG = '<pad>'
BOS_TAG = '<bos>'
EOS_TAG = '<eos>'
# UNK_TAG = '<unk>'

PAD = 0
BOS = 1
EOS = 2
# SPACE = 3

# https://github.com/arxrean/LipRead-seq2seq
class GridSeq2Seq(Dataset):
    def __init__(self, opt, phase='train'):
        self.phase = phase
        self.opt = opt
        assert phase in ['train', 'val', 'test']
        self.data = glob.glob(os.path.join(opt.video_root, 's*', '*'))
        # self.data = glob.glob(os.path.join(opt.video_root, 's2*', '*'))
        test_spks = ['s1', 's2', 's20', 's22']
        # test_spks = ['s20', 's22']
        if phase == 'train':
            self.cur_data = [x for x in self.data if
                             len(os.listdir(x)) > 5 and x.split(os.path.sep)[-2] not in test_spks]
        elif phase == 'val':
            self.cur_data = [x for x in self.data if len(os.listdir(x)) > 5 and x.split(os.path.sep)[-2] in test_spks]
            np.random.seed(123)
            np.random.shuffle(self.cur_data)
            self.cur_data = self.cur_data[:opt.val_batch * opt.batch_size]
        else:
            self.cur_data = [x for x in self.data if len(os.listdir(x)) > 0 and x.split(os.path.sep)[-2] in test_spks]

        # self.cur_data = list(filter(lambda fn: len(os.listdir(fn)) > 5, self.cur_data))
        self.char_list = [PAD_TAG, BOS_TAG, EOS_TAG] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                                        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                                                        'X', 'Y', 'Z']
        # 字符到索引的映射
        self.char_dict = {ch: idx for idx, ch in enumerate(self.char_list)}
        # 索引到字符的映射
        self.idx_dict = {idx: ch for idx, ch in enumerate(self.char_list)}
        self.transform = self.get_transform(phase)

    def __getitem__(self, index):
        item = self.cur_data[index]  # E:\dataset\GRID\lips\s1\bbaf2n
        video = self.load_video(item)
        align_path = os.path.join(self.opt.align_root, item.split(os.path.sep)[-2], item.split(os.path.sep)[-1])
        align_txt = self.load_align('{}.align'.format(align_path))
        align_idx = self.align2idx(align_txt)
        vid_len = video.shape[0]
        align_len = len(align_idx) - 2  # excluding [BOS] and [EOS]
        padded_align = self.align_pad(align_idx)

        if self.phase == 'test':
            return video, np.array(padded_align), align_txt, item
        else:
            return video, np.array(padded_align), vid_len, align_len

    def __len__(self):
        return len(self.cur_data)

    def load_video(self, name):
        files = os.listdir(name)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        # files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        files = sorted(files, key=self.extract_number)
        array = [cv2.imread(os.path.join(name, file)) for file in files]
        # array = list(filter(lambda im: im is not None, array))
        vid_pad = self.opt.max_vid_len
        if len(array) < vid_pad:
            array = np.concatenate([array, np.zeros([vid_pad - len(array)] + list(array[0].shape)).astype(np.uint8)])
        elif len(array) > vid_pad:
            array = array[:vid_pad]
        # array = [cv2.resize(img, (128, 64), interpolation=cv2.INTER_LANCZOS4) for img in array]
        array = [self.transform(img) for img in array]
        # array = np.stack(array, axis=0).astype(np.float32)
        array = torch.stack(array, dim=0)
        return array

    def load_align(self, name):
        with open(name, 'r') as f:
            txt = [line.strip().split(' ')[2] for line in f]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        return ' '.join(txt).upper()

    def extract_number(self,file_name):
        # 假设文件名格式为 'bwae1p_lip0.jpg'
        base_name = os.path.splitext(file_name)[0]
        parts = base_name.split('lip')
        return int(parts[-1])

    def align2idx(self, text):
        return [BOS] + [self.char_dict[x] for x in text] + [EOS]

    def align_pad(self, align):
        if len(align) == self.opt.max_dec_len + 2:  # including <bos> and <eos> token
            return align
        return align + [PAD] * (self.opt.max_dec_len + 2 - len(align))

    def get_transform(self, phase='train'):
        '''
        torchvision.transforms: 常用的数据预处理方法，提升泛化能力
        包括：数据中心化、数据标准化、缩放、裁剪、旋转、翻转、填充、噪声添加、灰度变换、线性变换、仿射变换、亮度、饱和度及对比度变换等
        '''
        # return transforms.Compose([
        #     transforms.Grayscale(),
        #     transforms.Resize((96, 96)),
        #     transforms.TenCrop((88, 88)),    # for testing  (bs, ncrops, c, h, w)
        #     # transforms.CenterCrop((88, 88)),  # for testing
        #     # transforms.RandomCrop((88, 88)),  # for training
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.Lambda(lambda crops: torch.stack(
        #         [transforms.ToTensor()(crop) for crop in crops])),
        # ])
        if phase == 'train':
            # 灰度图
            return transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Grayscale(),
                transforms.Resize((96, 96)),
                transforms.RandomCrop((88, 88)),  # for training
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
            # RGB图 (3通道)
            # return transforms.Compose([
            #     transforms.ToPILImage(),  # for 2 or 3 dimensional
            #     # transforms.Resize((64, 128)),  # H, W
            #     # transforms.RandomCrop((88, 88)),  # for training
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # 逐channel的对图像进行标准化(均值变为0，标准差变为1)，可以加快模型的收敛
            # ])
        else:
            # 灰度图
            return transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Grayscale(),
                # transforms.Resize((96, 96)),
                transforms.Resize((88, 88)),
                # transforms.CenterCrop((88, 88)),  # for testing
                transforms.ToTensor(),
            ])
            # RGB图 (3通道)
            # return transforms.Compose([
            #     transforms.ToPILImage(),
            #     transforms.Resize((64, 128)),  # H, W
            #     # transforms.CenterCrop((88, 88)),  # for testing
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # 逐channel的对图像进行标准化(均值变为0，标准差变为1)，可以加快模型的收敛
            # ])
