import argparse
import os


def vit_small_linear():
    args = argparse.Namespace()

    args.dataset = 'imagenet1k'
    args.arch = 'vit-small'
    args.pretrained_weights = ''
    args.resume = None
    args.evaluate = ''
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
    else:
        args.num_workers = 4
        args.prefetch_factor = 2
        args.pin_memory = True
        args.patch_size = 2
        args.input_size = 32
        args.batch_size = 256
        args.data_root = '/path/to/dataset'

    args.encoder = 'momentum_encoder'  # [base_encoder,momentum_encoder]

    # Optimizer parameters
    args.opt = 'sgd'
    args.opt_eps = 1e-8
    args.opt_betas = None
    args.clip_grad = None
    args.momentum = 0.9

    if args.dataset == 'cifar10':
        args.lr = 0.02
        args.weight_decay = 0.0
    elif args.dataset == 'cifar100':
        args.lr = 0.02
        args.weight_decay = 0.0
    elif args.dataset == 'imagenet1k':
        args.lr = 0.02
        args.weight_decay = 0.0

    # ----------------#
    args.dist_url = 'tcp://localhost:12612'
    args.dist_backend = 'nccl'
    args.rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args.world_size = 1

    args.print_freq = 100
    args.save_freq = 20

    args.distributed = True
    args.gpu = None
    args.exclude_file_list = ['__pycache__', '.vscode',
                              'log', 'ckpt', '.git', 'out', 'dataset', 'weight']

    return args
