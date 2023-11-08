#!/usr/bin/env python
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
import cv2

import wandb
from visualize import visualize_frames, visualize

from base.baseTrainer import save_checkpoint
from base.utilities import get_parser, get_logger, AverageMeter
from models import get_model
from metrics.loss import calc_vq_loss
from torch.optim.lr_scheduler import StepLR

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main():
    cfg = get_parser()
    cfg.gpu = cfg.train_gpu

    if cfg.use_wandb:
        wandb.init(project="VertexVQ")

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
    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr)

    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None

    # ####################### Data Loader ####################### #
    from dataset.data_loader import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    if cfg.evaluate:
        val_loader = dataset['valid']
        test_loader = dataset['test']

    # ####################### Train ############################# #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        rec_loss_train, quant_loss_train, pp_train = train(train_loader, model, calc_vq_loss, optimizer, epoch, cfg)
        epoch_log = epoch + 1
        if cfg.StepLR:
            scheduler.step()
        logger.info('TRAIN Epoch: {} '
                    'loss_train: {} '
                    'pp_train: {} '
                    .format(epoch_log, rec_loss_train, pp_train)
                    )
        for m, s in zip([rec_loss_train, quant_loss_train, pp_train],
                        ["train/rec_loss", "train/quant_loss", "train/perplexity"]):
            writer.add_scalar(s, m, epoch_log)

        if cfg.use_wandb:
            wandb.log({"train_epoch": epoch_log , "train_loss": rec_loss_train})

        if cfg.visualize and (epoch_log % cfg.visualize_freq == 0):
            video_path, gt_path = visualize_example(test_loader, model, cfg)

            if cfg.use_wandb:
                wandb.log({"video": wandb.Video(video_path, fps=cfg.fps, format="mp4"),
                            "video_gt": wandb.Video(gt_path, fps=cfg.fps, format="mp4")})  

        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            rec_loss_val, quant_loss_val, pp_val = validate(val_loader, model, calc_vq_loss, epoch, cfg)
            logger.info('VAL Epoch: {} '
                        'loss_val: {} '
                        'pp_val: {} '
                        .format(epoch_log, rec_loss_val, pp_val)
                        )
            for m, s in zip([rec_loss_val, quant_loss_val, pp_val],
                            ["val/rec_loss", "val/quant_loss", "val/perplexity"]):
                writer.add_scalar(s, m, epoch_log)
            
            if cfg.use_wandb:
                wandb.log({"val_epoch": epoch_log , "val_loss": rec_loss_val})


        if (epoch_log % cfg.save_freq == 0):
            save_checkpoint(model,
                            sav_path=os.path.join(cfg.save_path, 'model')
                            )
            
            if cfg.use_wandb:
                wandb.save(os.path.join(cfg.save_path, 'model.pth.tar'))




def train(train_loader, model, loss_fn, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    rec_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, (data, template, _) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)

        if data.shape[1] < 4:
            continue

        data = data.cuda(cfg.gpu, non_blocking=True)
        template = template.cuda(cfg.gpu, non_blocking=True)

        out, quant_loss, info = model(data, template)

        # LOSS
        loss, loss_details = loss_fn(out, data, quant_loss, quant_loss_weight=cfg.quant_loss_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([rec_loss_meter, quant_loss_meter, pp_meter],
                        [loss_details[0], loss_details[1], info[0]]): #info[0] is perplexity
            m.update(x.item(), 1)
        
        # Adjust lr
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
                                loss_meter=rec_loss_meter
                                ))
            for m, s in zip([rec_loss_meter, quant_loss_meter],
                            ["train_batch/loss", "train_batch/loss_2"]):
                writer.add_scalar(s, m.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    return rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg


def validate(val_loader, model, loss_fn, epoch, cfg):
    rec_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (data, template, _) in enumerate(val_loader):

            if data.shape[1] < 4:
                continue

            data = data.cuda(cfg.gpu, non_blocking=True)
            template = template.cuda(cfg.gpu, non_blocking=True)

            out, quant_loss, info = model(data, template)

            # LOSS
            loss, loss_details = loss_fn(out, data, quant_loss, quant_loss_weight=cfg.quant_loss_weight)


            for m, x in zip([rec_loss_meter, quant_loss_meter, pp_meter],
                            [loss_details[0], loss_details[1], info[0]]):
                m.update(x.item(), 1) #batch_size = 1 for validation


    return rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg

def visualize_example(val_loader, model, cfg):
    model.eval()
    save_folder = os.path.join(cfg.save_path, 'out')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with torch.no_grad():
        for i, (data, template, file_name)  in enumerate(val_loader):
            if data.shape[1] < 4:
                continue
            
            data = data.cuda(cfg.gpu, non_blocking=True)
            template = template.cuda(cfg.gpu, non_blocking=True)

            out, quant_loss, info = model(data, template)
            
            pred_video_path = visualize_frames( file_name, out, save_folder, fps=cfg.fps)
            gt_video_path = visualize_frames( file_name, data, save_folder, gt=True, fps=cfg.fps)

            return pred_video_path, gt_video_path



if __name__ == '__main__':
    main()
