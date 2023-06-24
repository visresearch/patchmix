import argparse
import os


def vit_small_finetune():
    args = argparse.Namespace()

    args.dataset = 'imagenet1k'
    args.arch = 'vit-small'
    args.pretrained_weights = ''
    args.resume = None
    args.evaluate = None
    args.epochs = 100
    args.start_epoch = 0
    args.output_dir = './out'
    args.seed = 7

    if args.dataset == 'imagenet1k':
        args.num_workers = 12
        args.prefetch_factor = 3
        args.pin_memory = True
        args.patch_size = 16
        args.input_size = 224
        args.batch_size = 1024
        args.data_root = '/path/to/ILSVRC2012'
        args.distributed = True
    else:
        args.num_workers = 4
        args.prefetch_factor = 2
        args.pin_memory = True
        args.patch_size = 2
        args.input_size = 32
        args.batch_size = 256
        args.data_root = '/path/to/dataset'
        args.distributed = False

    args.encoder = 'momentum_encoder'  # [base_encoder,momentum_encoder]

    # ---ema----------
    args.model_ema = True
    args.model_ema_decay = 0.99996
    args.model_ema_force_cpu = False
    args.drop_path = 0.1

    # Optimizer parameters
    args.opt = 'adamw'
    args.opt_eps = 1e-8
    args.opt_betas = None
    args.clip_grad = None
    args.momentum = 0.9

    # Learning rate schedule parameters
    args.sched = 'cosine'

    if args.dataset == 'cifar10':
        args.lr = 5e-4
        args.warmup_lr = 1e-6
        args.min_lr = 1e-5
        args.weight_decay = 0.05
    elif args.dataset == 'cifar100':
        args.lr = 5e-4
        args.warmup_lr = 1e-6
        args.min_lr = 1e-5
        args.weight_decay = 0.05
    elif args.dataset == 'imagenet1k':
        args.lr = 5e-4
        args.warmup_lr = 1e-6
        args.min_lr = 1e-5
        args.weight_decay = 0.05

    # learning schedule parameters
    args.layer_decay = 0.75
    args.lr_noise = None
    args.lr_noise_pct = 0.67
    args.lr_noise_std = 1.0
    args.decay_epochs = 30
    args.warmup_epochs = 10
    args.cooldown_epochs = 10
    args.patience_epochs = 10
    args.decay_rate = 0.1

    # Augmentation parameters
    args.color_jitter = 0.4
    args.aa = 'rand-m9-mstd0.5-inc1'
    args.smoothing = 0.1
    args.train_interpolation = 'bicubic'
    args.repeated_aug = True

    # Random Erase params
    args.reprob = 0.25
    args.remode = 'pixel'
    args.recount = 1
    args.resplit = False

    # Mixup params
    args.mixup = 0.8
    args.cutmix = 1.0
    args.cutmix_minmax = None  # float
    args.mixup_prob = 1.0
    args.mixup_switch_prob = 0.5
    args.mixup_mode = 'batch'

    # ----------------#
    args.dist_url = 'tcp://localhost:12613'
    args.dist_backend = 'nccl'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    args.world_size = 1

    args.print_freq = 100
    args.save_freq = 20

    args.rank = 0
    args.distributed = False
    args.gpu = None
    args.exclude_file_list = ['__pycache__', '.vscode', 'log', 'ckpt', '.git', 'out', 'dataset', 'weight']

    return args
