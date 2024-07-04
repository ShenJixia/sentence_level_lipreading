import os
import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from logger import logger
from options import get_arg_parser

def get_dataset(opt, mode='train'):
    if opt.dataset == 'grid':
        from data.dataset_grid_overlap import GridSeq2Seq
        if mode == 'train':
            dataset_train = GridSeq2Seq(opt, phase='train')
            dataset_val = GridSeq2Seq(opt, phase='val')
            return dataset_train, dataset_val
        else:
            dataset_test = GridSeq2Seq(opt, phase='test')
            return dataset_test
    else:
        raise


def get_model(opt):
    from model_arch.model2406 import Seq2Seq
    model = Seq2Seq(opt)
    print(model)
    return model


def train(opt):
    train_set, val_set = get_dataset(opt, 'train')
    print(f'train set: {len(train_set)}, val set: {len(val_set)}')
    train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.num_workers)

    model = get_model(opt).to(opt.device)
    param_nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {param_nums} trainable parameters')
    # model = nn.DataParallel(model)

    params = [{'params': [p for p in model.encoder.parameters() if p.requires_grad],
         'weight_decay': opt.weight_decay, 'lr': opt.enc_lr},
        {'params': [p for p in model.decoder.parameters() if p.requires_grad],
         'weight_decay': opt.weight_decay, 'lr': opt.dec_lr}]
    optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=opt.weight_decay)
    # 每过step_size个epoch，做一次lr更新
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_step, gamma=0.5)
    total_steps = len(train_loader) * opt.epochs
    warmup_linear_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    if opt.weights is not None:
        ckpt = torch.load(opt.weights, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('loading pre-trained weights ...')

    best_val_loss = 1e10
    for epoch in range(opt.epochs):
        model.train()
        ep_loss = 0.
        for step, batch in enumerate(train_loader):
            vid, align = batch[0], batch[1]
            vid_len, align_len = batch[2], batch[3]
            vid = vid.to(opt.device)
            align = align.to(opt.device)
            vid_len = vid_len.to(opt.device)
            align_len = align_len.to(opt.device)

            model.zero_grad()
            loss, _ = model(vid, align, vid_len, align_len)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            warmup_linear_scheduler.step()
            ep_loss += loss.item()

            if step % 5 == 0:
                logger.info('epoch: {} step: {}/{} train_loss: {:.4f}'.format(
                    epoch, (step+1), len(train_loader), loss.item()))
        logger.info('epoch loss: {}'.format(ep_loss / len(train_loader)))

        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                vid, align = batch[0], batch[1]
                vid_len, align_len = batch[2], batch[3]
                vid = vid.to(opt.device)
                align = align.to(opt.device)
                vid_len = vid_len.to(opt.device)
                align_len = align_len.to(opt.device)
                loss, _ = model(vid, align, vid_len, align_len)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            logger.info('=========== val loss: {:.4f} ==========='.format(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('saved!!!')
            if opt.weights is None:
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join('checkpoints', opt.dataset, 'best.ep{}.pt'.format(epoch)))
            else:
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join('checkpoints', opt.dataset, 'continue.best.ep{}.pt'.format(epoch)))
        else:
            # lr_scheduler.step()  # adjust lr for each epoch
            pass

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
    train(opt)
