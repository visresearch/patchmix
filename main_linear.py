import argparse
import os
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

from config.linear.vit_base_linear import vit_base_linear
from config.linear.vit_small_linear import vit_small_linear
from config.linear.vit_tiny_linear import vit_tiny_linear
from module.vits import ViT
from utils import misc
from utils.logger import Logger, console_logger
from utils.misc import AverageMeter, copy_files


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(
            args.data_root, train=is_train, transform=transform)
        nb_classes = 100
    elif args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            args.data_root, train=is_train, transform=transform)
        nb_classes = 10
    elif args.dataset == 'imagenet1k':
        dataset = datasets.ImageFolder(
            root=os.path.join(args.data_root, 'train' if is_train else 'val'), transform=transform)
        nb_classes = 1000
    return dataset, nb_classes


def build_transform(is_train, args):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    else:
        return transforms.Compose([
            transforms.Resize(
                int(args.input_size / 224 * 256), interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])


def get_model_from_frame(checkpoint, args):
    encoder = args.encoder
    state_dict = checkpoint['state_dict']
    encoder = ('module.' if 'module' in list(
        state_dict.keys())[0] else '') + encoder
    for k in list(state_dict.keys()):
        if k.startswith(encoder) and not k.startswith(encoder + '.head'):
            state_dict[k[len(encoder + "."):]] = state_dict[k]
        del state_dict[k]
    return state_dict


def sanity_check(state_dict, pretrained_weights, args):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    encoder = args.encoder
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']
    module_kw = 'module.' if 'module' in list(state_dict_pre.keys())[0] else ''
    for k in list(state_dict.keys()):
        if 'head.weight' in k or 'head.bias' in k:
            continue
        pre_k = module_kw + encoder + '.' + (k[len('module.'):] if 'module' in k else k)
        assert ((state_dict[k].cpu() == state_dict_pre[pre_k]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def train_one_epoch(model: torch.nn.Module, criterion,
                    train_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, loggers, args, max_norm: float = 0,
                    ):
    model.train()
    logger_tb, logger_console = loggers

    losses = AverageMeter('Loss', ':.4e')

    num_iter = len(train_loader)
    niter_global = epoch * num_iter

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(args.rank, non_blocking=True)
        targets = targets.to(args.rank, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        niter_global += 1
        if args.rank == 0:
            logger_tb.add_scalar('Finetune/Iter/loss',
                                 losses.val, niter_global)

        if (i + 1) % args.print_freq == 0 and logger_console is not None and args.rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger_console.info(f'Epoch [{epoch}][{i + 1}/{num_iter}] - '
                                f'lr: {lr:.5f},     '
                                f'loss: {losses.avg:.3f}')
    if args.pretrained_weights and epoch == args.start_epoch and args.rank == 0:
        sanity_check(model.state_dict(), args.pretrained_weights, args)
    if args.distributed:
        losses.synchronize_between_processes()

    return losses.avg


@torch.no_grad()
def evaluate(data_loader, model, args):
    accs1 = AverageMeter('Acc@1', ':6.2f')
    accs5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()

    for i, (images, target) in enumerate(data_loader):
        images = images.to(args.rank, non_blocking=True)
        target = target.to(args.rank, non_blocking=True, dtype=torch.long)

        with torch.cuda.amp.autocast():
            output = model(images)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        accs1.update(acc1.item(), batch_size)
        accs5.update(acc5.item(), batch_size)
    if args.distributed:
        accs1.synchronize_between_processes()
        accs5.synchronize_between_processes()

    return accs1.avg, accs5.avg


def main_ddp(args):
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.world_size * ngpus_per_node
        mp.spawn(main, args=(args,), nprocs=args.world_size)
    else:
        main(args.rank, args)


def main(rank, args):
    args.rank = rank
    if args.distributed:
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    misc.fix_random_seeds(args.seed)

    cudnn.benchmark = True
    if not args.evaluate:
        if args.rank == 0:
            for k, v in sorted(vars(args).items()):
                print(k, '=', v)
            name = str(args.arch) + "_" + str(args.dataset) + \
                   "_epochs_" + str(args.epochs) + "_lr_" + str(args.lr)
            logger_tb = Logger(args.output_dir, name)
            logger_console = console_logger(logger_tb.log_dir, 'console_eval')
            dst_dir = os.path.join(logger_tb.log_dir, 'code/')
            copy_files('./', dst_dir, args.exclude_file_list)
        else:
            logger_tb, logger_console = None, None
        if args.rank == 0:
            path_save = os.path.join(args.output_dir, logger_tb.log_name)

    dataset_train, num_class = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = args.world_size
        global_rank = args.rank
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        args.num_workers = int((args.num_workers + 1) / args.world_size)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.arch == 'vit-tiny':
        model = ViT(patch_size=args.patch_size, img_size=args.input_size,
                    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4)
    elif args.arch == 'vit-small':
        model = ViT(patch_size=args.patch_size, img_size=args.input_size,
                    embed_dim=384, depth=12, num_heads=12, mlp_ratio=4)

    elif args.arch == 'vit-base':
        model = ViT(patch_size=args.patch_size, img_size=args.input_size,
                    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4)

    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print("=> loading checkpoint '{}'".format(args.pretrained_weights))
            checkpoint = torch.load(
                args.pretrained_weights, map_location=torch.device(args.rank))

            state_dict = get_model_from_frame(checkpoint, args)

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg.missing_keys)
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained_weights))
        else:
            print("=> no checkpoint found at '{}'".format(
                args.pretrained_weights))

    model.head = nn.Linear(model.head.in_features, num_class)

    for name, param in model.named_parameters():
        if name not in ['head.weight', 'head.bias']:
            param.requires_grad = False
    model.cuda(args.rank)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.rank])
        torch.cuda.set_device(args.rank)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        args.batch_size = int(args.batch_size / args.world_size)
        model_without_ddp = model.module

    if args.distributed:
        args.lr = args.lr * args.batch_size * args.world_size / 256
    else:
        args.lr = args.lr * args.batch_size / 256

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0)

    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=False
    )

    acc_best = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            acc_best = checkpoint['acc_best']
            if args.gpu is not None:
                acc_best = acc_best.to(args.gpu)
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            loss_scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            model = torch.load(
                args.evaluate, map_location=torch.device(args.rank))
            print("=> loaded pre-trained model '{}'".format(args.evaluate))
        else:
            print("=> no checkpoint found at '{}'".format(args.evaluate))
        acc1, acc5 = evaluate(data_loader_val, model, args)
        print('Acc1 :' + str(acc1) + '\tAcc5 :' + str(acc5))
        return

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        loss = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, loss_scaler, (logger_tb, logger_console), args,
            args.clip_grad
        )
        if args.rank == 0:
            logger_tb.add_scalar('Finetune/Epoch/loss', loss, epoch)

        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        if epoch % args.save_freq == 0 and args.rank == 0:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'acc_best': acc_best,
                    'optimizer': optimizer.state_dict(),
                    'scaler': loss_scaler.state_dict(),
                },
                f'{path_save}/{epoch:0>4d}.pth'
            )

        lr_scheduler.step(epoch)
        acc1, acc5 = evaluate(data_loader_val, model, args)
        if args.rank == 0:
            logger_tb.add_scalar('Finetune/Epoch/Accuracy', acc1, epoch)
            logger_console.info(
                f'Epoch: {epoch}\t'
                f'Acc1: {round(acc1, 3)}\t'
                f'Acc5: {round(acc5, 3)}'
            )

        if acc1 > acc_best:
            acc_best = acc1
            epoch_best = epoch
            if args.rank == 0:
                torch.save(
                    model_without_ddp,
                    f'{path_save}/best.pth'
                )

        if args.rank == 0:
            logger_console.info(
                f'Epoch: {epoch_best}, '
                f'Best Acc1: {round(acc_best, 3)}'
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default='vit-small',
                        choices=['vit-tiny', 'vit-small', 'vit-base'])
    parser.add_argument("--pretrained-weights", type=str,
                        default='')
    parser.add_argument("--evaluate", type=str, default=None)
    return parser


if __name__ == '__main__':
    parser = parse_args()
    _args = parser.parse_args()

    if _args.arch == 'vit-tiny':
        args = vit_tiny_linear()
    elif _args.arch == 'vit-small':
        args = vit_small_linear()
    elif _args.arch == 'vit-base':
        args = vit_base_linear()
    args.pretrained_weights = _args.pretrained_weights
    args.evaluate = _args.evaluate
    main_ddp(args)
