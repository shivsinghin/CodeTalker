#!/usr/bin/env python
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
import cv2

from base.baseTrainer import save_checkpoint
from base.utilities import get_parser, get_logger, AverageMeter
from models import get_model
from torch.optim.lr_scheduler import StepLR
from visualize import visualize

import wandb

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import warnings
warnings.filterwarnings("ignore")


def main():
    cfg = get_parser()
    cfg.gpu = cfg.train_gpu

    if cfg.use_wandb:
        wandb.init(project="Audio2Vertex")

 # ####################### Model ####################### #
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)
    model = get_model(cfg)
    
    logger.info(cfg)
    logger.info("=> creating model ...")
   
    torch.cuda.set_device(cfg.gpu)
    model = model.cuda()

    if cfg.use_wandb:
        wandb.watch(model)
    # ####################### Loss ############################# #
    loss_fn = nn.MSELoss()

    # ####################### Optimizer ######################## #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=cfg.base_lr)

    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None

    # ####################### Data Loader ####################### #
    from dataset.data_loader import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    test_loader = dataset['test']
    if cfg.evaluate:
        val_loader = dataset['valid']

    # ####################### Train ############################# #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        loss_train, motion_loss_train, reg_loss_train = train(train_loader, model, loss_fn, optimizer, epoch, cfg)
        epoch_log = epoch + 1
        if cfg.StepLR:
            scheduler.step()
            
        logger.info('TRAIN Epoch: {} '
                    'loss_train: {} '
                    .format(epoch_log, loss_train)
                    )
        for m, s in zip([loss_train, motion_loss_train, reg_loss_train],
                        ["train/loss", "train/motion_loss", "train/reg_loss"]):
            writer.add_scalar(s, m, epoch_log)

        if cfg.use_wandb:
            wandb.log({"train_epoch": epoch_log , "train_loss": loss_train})
        
        if cfg.visualize and (epoch_log % cfg.visualize_freq == 0):
            video_path, gt_path = visualize_example(test_loader, model, loss_fn, cfg)

            if cfg.use_wandb:
                wandb.log({"video": wandb.Video(video_path, fps=30, format="mp4"),
                            "video_gt": wandb.Video(gt_path, fps=25, format="mp4")})   

        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            loss_val = validate(val_loader, model, loss_fn, cfg)
            logger.info('VAL Epoch: {} '
                        'loss_val: {} '
                        .format(epoch_log, loss_val)
                        )
            for m, s in zip([loss_val],
                            ["val/loss"]):
                writer.add_scalar(s, m, epoch_log)
            
            if cfg.use_wandb:
                wandb.log({"val_epoch": epoch_log , "val_loss": loss_val})

        if (epoch_log % cfg.save_freq == 0):
            save_checkpoint(model,
                            sav_path=os.path.join(cfg.save_path, 'model'),
                            stage=2
                            )
            
            if cfg.use_wandb:
                wandb.save(os.path.join(cfg.save_path, 'model.pth.tar'))


def train(train_loader, model, loss_fn, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_motion_meter = AverageMeter()
    loss_reg_meter = AverageMeter()


    model.train()
    model.autoencoder.eval()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, (audio, data, template, _) in enumerate(train_loader):
        # pdb.set_trace()
        ####################
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)

        if data.shape[1] > 600:
            continue
        #################### cpu to gpu
        audio = audio.cuda(cfg.gpu, non_blocking=True)
        data = data.cuda(cfg.gpu, non_blocking=True) 
        template = template.cuda(cfg.gpu, non_blocking=True)

        loss, loss_detail = model(audio, template, data, criterion=loss_fn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ######################
        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([loss_meter, loss_motion_meter, loss_reg_meter],
                        [loss, loss_detail[0], loss_detail[1]]):
            m.update(x.item(), 1)

        
        current_lr = optimizer.param_groups[0]['lr']

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain: {remain_time} '
                        'Loss: {loss_meter.val:.4f} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                loss_meter=loss_meter
                                ))
            for m, s in zip([loss_meter],
                            ["train_batch/loss"]):
                writer.add_scalar(s, m.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    return loss_meter.avg, loss_motion_meter.avg, loss_reg_meter.avg


def validate(val_loader, model, loss_fn, cfg):
    loss_meter = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (audio, vertice, template,_) in enumerate(val_loader):

            audio = audio.cuda(cfg.gpu, non_blocking=True)
            vertice = vertice.cuda(cfg.gpu, non_blocking=True)
            template = template.cuda(cfg.gpu, non_blocking=True)


            loss, _ = model(audio, template, vertice, criterion=loss_fn)
            loss_meter.update(loss.item(), 1)


    return loss_meter.avg

def visualize_example(val_loader, model, loss_fn, cfg):
    model.eval()
    save_folder = os.path.join(cfg.save_path, 'out')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with torch.no_grad():
        for i, (audio, vertice, template,file_name) in enumerate(val_loader):
            # if audio.shape[1] > 300000:
            #     continue
            print(audio.shape, vertice.shape, template.shape)
            audio = audio.cuda(cfg.gpu, non_blocking=True)
            vertice = vertice.cuda(cfg.gpu, non_blocking=True)
            template = template.cuda(cfg.gpu, non_blocking=True)
            prediction = model.predict(audio, template)
            print(prediction.shape, vertice.shape)
            pred_video_path = visualize( file_name[0], prediction[0], template[0], save_folder)
            gt_video_path = visualize( file_name[0], vertice[0], template[0], save_folder, gt=True)

            return pred_video_path, gt_video_path


if __name__ == '__main__':
    main()
