import os
import numpy as np
import random
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader
from evaluation_metric import edit_dist
from options import get_arg_parser
from data.dataset import BOS, EOS, PAD

def get_dataset(opt, mode='train'):
    from data.dataset_grid_overlap import GridSeq2Seq
    dataset_test = GridSeq2Seq(opt, phase='test')
    return dataset_test

def get_model(opt):
    from model_arch.model2406 import Seq2Seq
    model = Seq2Seq(opt)
    print(model)
    return model

def evaluate(opt):
    test_set = get_dataset(opt, 'test')
    print(f'test set: {len(test_set)}')
    test_loader = DataLoader(test_set, batch_size=opt.batch_size,
                             shuffle=False, num_workers=opt.num_workers)

    model = get_model(opt).to(opt.device)
    print(model)
    checkpoint = torch.load(opt.load, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()

    wer_list = []
    cer_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            vid, align, align_txt = batch[0], batch[1], batch[2]
            vid = vid.to(opt.device)
            # res = model.greedy_decoding(vid, bos_id=BOS, eos_id=EOS)
            res = model.beam_search_decoding(vid, bos_id=BOS, eos_id=EOS)
            pred = list(map(lambda x: ''.join([test_set.idx_dict[i] for i in x if i != EOS and i != PAD]), res))
            print(pred, align_txt)
            wer_list.extend([edit_dist(p.split(' '), t.split(' ')) / len(t.split(' ')) for p, t in zip(pred, align_txt)])
            cer_list.extend([edit_dist(p, t) / len(t) for p, t in zip(pred, align_txt)])
    print('overall wer: {:.4f}, cer: {:.4f}'.format(np.mean(wer_list), np.mean(cer_list)))


def set_seeds(seed=1349):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    
if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())
    opt = get_arg_parser()
    set_seeds(1347)
    opt.device = torch.device('cuda', opt.cuda) if torch.cuda.is_available() and opt.cuda >= 0 else torch.device('cpu')
    print(opt.device)
    print("cuda get device name：", torch.cuda.get_device_name(0))
    evaluate(opt)
