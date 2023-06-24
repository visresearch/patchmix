import argparse
import os


def vit_small_pretrain():
    args = argparse.Namespace()
    args.arch = 'vit-small'
    args.resume = None
    args.dataset = 'imagenet1k'
    args.seed = 7

    if args.dataset == 'imagenet1k':
        args.data_root = '/path/to/ILSVRC2012'
        args.input_size = 224
        args.patch_size = 16
        args.num_workers = 32
        args.prefetch_factor = 3
        args.pin_memory = True
        args.save_freq = 10
        args.epochs = 300
        args.batch_size = 1024
        args.warmup_epoch = 10
        args.multi_crop_size = 96
        # multi-crop params
        args.multi_crop_num = 8  # unuse multi-crop when set 0
        args.global_crop = 0.35
        args.mix_num = 2
        args.mix_size = args.patch_size
        args.smoothing = 0.0
        args.min_crop = 0.05
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        args.data_root = '/path/to/dataset'
        args.input_size = 32
        args.patch_size = 2
        args.num_workers = 8
        args.prefetch_factor = 2
        args.pin_memory = True
        args.save_freq = 100
        args.epochs = 800
        args.batch_size = 512
        args.warmup_epoch = 100
        args.multi_crop_size = 14
        # multi-crop params
        args.multi_crop_num = 0  # unuse multi-crop when set 0
        args.global_crop = 0.35
        args.mix_num = 2
        args.mix_size = args.patch_size
        args.smoothing = 0.0
        args.min_crop = 0.1

    args.drop_path = 0.1

    # lr params
    args.lr = 5e-4
    args.min_lr = 1e-6
    args.weight_decay = 0.04
    args.weight_decay_end = 0.4
    args.use_wd_cos = True
    if not args.use_wd_cos:
        args.weight_decay_end = args.weight_decay

    # moco params
    args.use_moco = True
    args.moco_m = 0.996
    args.moco_m_cos = True

    args.print_freq = None

    args.out_dim = 256
    args.hidden_dim = 4096
    args.proj_layer = 3
    args.pred_layer = 2
    args.temp = 0.2
    args.warmup_temp = 0.2
    args.warmup_temp_epochs = 30
    args.mix_p = 1.0

    args.exp_dir = f'./log/pretrain/{args.dataset}/ckpts_{args.arch}_p{args.patch_size}' \
                   f'_moco_{args.use_moco}_mm{args.moco_m}_min_crop{args.min_crop}' \
                   f'_t{args.temp}_lr{args.lr}_wd{args.weight_decay}' \
                   f'_bs{args.batch_size}_epoch{args.epochs}' \
                   f'_global_crop{args.global_crop}' \
                   f'_mc_n{args.multi_crop_num}_dp{args.drop_path}'

    args.rank = 0
    args.distributed = True
    args.use_mix_precision = True
    args.init_method = 'tcp://localhost:17991'
    args.world_size = 1

    args.exclude_file_list = ['__pycache__', '.vscode',
                              'log', 'ckpt', '.git', 'out', 'dataset', 'weight']

    return args
