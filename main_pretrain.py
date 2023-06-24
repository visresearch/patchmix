import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets

from config.pretrain.vit_base_pretrain import vit_base_pretrain
from config.pretrain.vit_small_pretrain import vit_small_pretrain
from config.pretrain.vit_tiny_pretrain import vit_tiny_pretrain
from module.augmentation import TwoCropsTransform, MultiCropsTransform
from module.frame.contrast_momentum import ContrastMomentum_ViT
from module.frame.contrast_no_momentum import ContrastNoMomentum_ViT
from module.loss import MultiTempContrastiveLoss
from module.mix import PatchMixer
from module.vits import ViT
from utils import misc
from utils.logger import Logger, console_logger
from utils.misc import AverageMeter, adjust_moco_momentum


def train_epoch(train_loader, model, criterion, optimizer, lr_schedule, wd_schedule, temp_schedule, mixer, scaler,
                loggers, epoch, args):
    model.train()
    logger_tb, logger_console = loggers

    src_losses = AverageMeter('Src Loss', ':.4e')
    mix_losses = AverageMeter('Mix Loss', ':.4e')
    mix2_losses = AverageMeter('Mix Mix Loss', ':.4e')
    multi_losses = AverageMeter('Multi Loss', ':.4e')
    multi_mix_losses = AverageMeter('Multi Mix Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    learning_rates = AverageMeter('LR', ':.4e')
    weight_decays = AverageMeter('WD', ':.4e')

    num_iter = len(train_loader)
    niter_global = epoch * num_iter
    no_mixer = PatchMixer(mix_s=args.mix_size,
                          num_classes=int(args.batch_size * args.world_size), mix_p=0.0, mix_n=1)

    moco_m = args.moco_m
    temp = temp_schedule[epoch]
    for i, (images, _) in enumerate(train_loader):
        # update weight decay and learning rate according to their schedule
        it = num_iter * epoch + i  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0 and args.use_wd_cos:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        if args.use_moco:
            if args.moco_m_cos:
                moco_m = adjust_moco_momentum(epoch + i / num_iter, args)
            with torch.no_grad():  # no gradient
                model.module.update_momentum_encoder(moco_m)

        images[0] = images[0].cuda(args.rank, non_blocking=True)
        images[1] = images[1].cuda(args.rank, non_blocking=True)
        if args.multi_crop_num != 0:
            for id in range(args.multi_crop_num):
                images[2][id] = images[2][id].cuda(
                    args.rank, non_blocking=True)

        N = images[0].size(0)
        target = torch.arange(N, dtype=torch.long).cuda()
        optimizer.zero_grad()
        mix_image1, mix_target, mix2_target = mixer(images[0], target)
        mix_image2, mix_target, mix2_target = mixer(images[1], target)
        images[0], target0, _ = no_mixer(images[0], target)
        images[1], target0, _ = no_mixer(images[1], target)
        

        with torch.cuda.amp.autocast(True):
            q1, k1 = model(images[0])
            _, k2 = model(images[1], q=False)
            _, m_k1 = model(mix_image1, q=False)
            m_q2, _ = model(mix_image2, k=False)
            src_loss = criterion(q1, k2.detach(), target0, temp)
            mix_loss = criterion(m_q2, k1.detach(), mix_target, temp) / 2.0
            mix2_loss = criterion(m_q2, m_k1.detach(), mix2_target, temp) / 2.0
            multi_loss = 0.
            multi_mix_loss = 0.
            if args.multi_crop_num != 0:
                multi_image = []
                multi_mix_image = []
                for id in range(args.multi_crop_num):
                    images[2][id], _, _ = no_mixer(images[2][id], target)
                    mix_image, _, _ = mixer(images[2][id], target)
                    multi_image.append(images[2][id])
                    multi_mix_image.append(mix_image)
                multi_image = torch.cat(multi_image, dim=0)
                multi_mix_image = torch.cat(multi_mix_image, dim=0)
                with torch.cuda.amp.autocast(True):
                    multi_q_, _ = model(multi_image, k=False)
                    multi_m_q_, _ = model(multi_mix_image, k=False)
                    mts1 = 0.
                    mts2 = 0.
                    mtms1 = 0.
                    mtms2 = 0.
                    for id in range(args.multi_crop_num):
                        mts1 += criterion(multi_q_[N * id:N * (id + 1)], k1.detach(), target0, temp)
                        mts2 += criterion(multi_q_[N * id:N * (id + 1)], k2.detach(), target0, temp)
                        mtms1 += criterion(multi_m_q_[N * id:N * (id + 1)], k1.detach(), mix_target, temp)
                        mtms2 += criterion(multi_m_q_[N * id:N * (id + 1)], k2.detach(), mix_target, temp)
                    multi_loss = (mts1 + mts2) / args.multi_crop_num
                    multi_mix_loss = (mtms1 + mtms2) / args.multi_crop_num
            loss = src_loss + mix_loss + mix2_loss + multi_loss + multi_mix_loss
        scaler.scale(loss).backward()
        

        scaler.step(optimizer)
        scaler.update()

        src_losses.update(src_loss.item(), N)
        mix_losses.update(mix_loss.item(), N)
        mix2_losses.update(mix2_loss.item(), N)
        multi_losses.update(
            multi_loss.item() if args.multi_crop_num != 0 else 0.0, N)
        multi_mix_losses.update(
            multi_mix_loss.item() if args.multi_crop_num != 0 else 0.0, N)
        losses.update(loss.item(), N)

        learning_rates.update(lr_schedule[it])
        weight_decays.update(wd_schedule[it])
        niter_global += 1

    if args.distributed:
        src_losses.synchronize_between_processes()
        mix_losses.synchronize_between_processes()
        mix_losses.synchronize_between_processes()
        multi_losses.synchronize_between_processes()
        multi_mix_losses.synchronize_between_processes()
        losses.synchronize_between_processes()

    if logger_console is not None and args.rank == 0:
        logger_console.info(f'Epoch [{epoch}][{i + 1}/{num_iter}] - '
                            f'lr: {lr_schedule[it]:.5f},   '
                            f'wd: {wd_schedule[it]:.5f},   '
                            f'src loss: {src_losses.avg:.3f},   '
                            f'mix loss: {mix_losses.avg:.3f},   '
                            f'mix2 loss: {mix2_losses.avg:.3f},   '
                            f'multi loss: {multi_losses.avg:.3f},   '
                            f'multi mix loss: {multi_mix_losses.avg:.3f},   '
                            f'loss: {losses.avg:.3f}'
                            )

    if logger_tb is not None and args.rank == 0:
        logger_tb.add_scalar('Epoch/Src Loss', src_losses.avg, epoch + 1)
        logger_tb.add_scalar('Epoch/Mix Loss', mix_losses.avg, epoch + 1)
        logger_tb.add_scalar('Epoch/Mix2 Loss', mix2_losses.avg, epoch + 1)
        logger_tb.add_scalar('Epoch/Multi Loss', multi_losses.avg, epoch + 1)
        logger_tb.add_scalar('Epoch/Multi Mix Loss',
                             multi_mix_losses.avg, epoch + 1)
        logger_tb.add_scalar('Epoch/Loss', losses.avg, epoch + 1)
        logger_tb.add_scalar('Epoch/lr', lr_schedule[it], epoch + 1)
        logger_tb.add_scalar('Epoch/wd', wd_schedule[it], epoch + 1)


def main_worker(gpu, ngpus_per_node, args):
    rank = args.rank * ngpus_per_node + gpu
    if args.distributed:
        dist.init_process_group(
            backend='nccl', init_method=args.init_method, rank=rank, world_size=args.world_size)
        torch.distributed.barrier()
    args.rank = rank
    misc.fix_random_seeds(args.seed)

    # ------------------------------ logger -----------------------------#
    if args.rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
        log_root = args.exp_dir
        name = f'vit_encoder_projection_{args.proj_layer}layers_with_BN_prediction_{args.pred_layer}layers'f'_dim{args.out_dim}'f'_hidden_dim{args.hidden_dim}'
        logger_tb = Logger(log_root, name)
        logger_console = console_logger(logger_tb.log_dir, 'console')
    else:
        logger_tb, logger_console = None, None

    # --------------------------------- model ------------------------------#

    if args.arch == 'vit-tiny':
        base_encoder = ViT(patch_size=args.patch_size, img_size=args.input_size, num_classes=args.out_dim,
                      embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, drop_path_rate=args.drop_path)
        momentum_encoder = ViT(patch_size=args.patch_size, img_size=args.input_size,
                      num_classes=args.out_dim, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4)
    elif args.arch == 'vit-small':
        base_encoder = ViT(patch_size=args.patch_size, img_size=args.input_size, num_classes=args.out_dim,
                      embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=args.drop_path)
        momentum_encoder = ViT(patch_size=args.patch_size, img_size=args.input_size,
                      num_classes=args.out_dim, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4)
    elif args.arch == 'vit-base':
        base_encoder = ViT(patch_size=args.patch_size, img_size=args.input_size, num_classes=args.out_dim,
                      embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=args.drop_path)
        momentum_encoder = ViT(patch_size=args.patch_size, img_size=args.input_size,
                      num_classes=args.out_dim, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4)

    if args.use_moco:
        model = ContrastMomentum_ViT(base_encoder, momentum_encoder, args.proj_layer,
                                     args.pred_layer, args.out_dim, args.hidden_dim)
    else:
        model = ContrastNoMomentum_ViT(base_encoder, args.proj_layer,
                                       args.pred_layer, args.out_dim, args.hidden_dim)

    model = model.cuda(args.rank)

    args.lr = args.lr * args.batch_size / 256

    if args.distributed:
        torch.cuda.set_device(args.rank)
        args.batch_size = int(args.batch_size / args.world_size)
        args.num_workers = int(
            (args.num_workers + args.world_size - 1) / args.world_size)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.rank], broadcast_buffers=False)

    # --------------------------- data load -----------------------#
    transform = TwoCropsTransform(
        args) if args.multi_crop_num == 0 else MultiCropsTransform(args)
    if args.dataset == 'cifar10':
        train_set = datasets.CIFAR10(root=args.data_root, train=True, download=False,
                                     transform=transform)
    elif args.dataset == 'cifar100':
        train_set = datasets.CIFAR100(root=args.data_root, train=True, download=False,
                                      transform=transform)
    elif args.dataset == 'imagenet1k':
        train_set = datasets.ImageFolder(root=os.path.join(args.data_root, 'train'),
                                         transform=transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              pin_memory=args.pin_memory,
                              drop_last=True,
                              prefetch_factor=args.prefetch_factor)

    args.niters_per_epoch = len(train_set) // args.batch_size

    # ----------------------------- patchmix ----------------------------------#
    mixer = PatchMixer(
        num_classes=int(args.batch_size * args.world_size), mix_s=args.mix_size,
        mix_n=args.mix_num, mix_p=args.mix_p, smoothing=args.smoothing)

    # ---------------------------- loss ---------------------------#
    criterion = MultiTempContrastiveLoss()

    # ---------------------------- optimizer ---------------------------#
    if args.use_wd_cos:
        parameters = model.module.named_parameters() if isinstance(
            model, DDP) else model.named_parameters()
        params_groups = misc.get_params_groups(parameters)
        optimizer = torch.optim.AdamW(params_groups)
    else:
        parameters = model.module.parameters() if isinstance(
            model, DDP) else model.parameters()
        optimizer = torch.optim.AdamW(
            parameters, weight_decay=args.weight_decay)

    scaler = GradScaler()

    start_epoch = 0

    # ---------------------------- scheduler ---------------------------#
    lr_schedule = misc.cosine_scheduler(
        args.lr,  # linear scaling rule
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epoch,
    )
    wd_schedule = misc.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )

    temp_schedule = np.concatenate((
        np.linspace(args.warmup_temp, args.temp, args.warmup_temp_epochs),
        np.ones(args.epochs - args.warmup_temp_epochs) * args.temp
    ))

    # ---------------------------- checkpoint ---------------------------#
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.rank == 0:
        path_save = os.path.join(args.exp_dir, logger_tb.log_name)

    # ---------------------------- training ---------------------------#
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(train_loader, model, criterion, optimizer, lr_schedule, wd_schedule, temp_schedule,
                    mixer, scaler, (logger_tb, logger_console), epoch, args)

        if (epoch + 1) % args.save_freq == 0 and args.rank == 0:
            _epoch = epoch + 1
            state_dict = model.module.state_dict() if isinstance(
                model, DDP) else model.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, f'{path_save}/{_epoch:0>4d}.pth')

    if args.rank == 0:
        state_dict = model.module.state_dict() \
            if isinstance(model, DDP) else model.state_dict()

        torch.save({'state_dict': state_dict}, f'{path_save}/last.pth')


def main(args):
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.world_size * ngpus_per_node
    if args.distributed:
        mp.spawn(main_worker, args=(ngpus_per_node, args),
                 nprocs=args.world_size)
    else:
        main_worker(args.rank, ngpus_per_node, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default='vit-small',
                        choices=['vit-tiny', 'vit-small', 'vit-base'])
    return parser


if __name__ == '__main__':
    parser = parse_args()
    _args = parser.parse_args()
    if _args.arch == 'vit-tiny':
        args = vit_tiny_pretrain()
    elif _args.arch == 'vit-small':
        args = vit_small_pretrain()
    elif _args.arch == 'vit-base':
        args = vit_base_pretrain()
    main(args)
