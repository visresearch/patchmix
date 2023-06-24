
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms as pth_transforms

from config.knn.knn import knn
from module.vits import ViT
from utils import misc


class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index


def build_dataset(is_train, args):
    transform = build_transform(args)
    dataset = ImageFolderInstance(
        root=os.path.join(args.data_root, 'train' if is_train else 'val'), transform=transform)
    return dataset


def build_transform(args):
    return transforms.Compose([
        pth_transforms.Resize(int(args.input_size / 224 * 256), interpolation=3),
        pth_transforms.CenterCrop(args.input_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


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


def eval_knn(rank, args):
    args.rank = rank
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    misc.fix_random_seeds(args.seed)

    cudnn.benchmark = True

    if args.load_features:
        try:
            print("loading features...")
            train_features = torch.load(os.path.join(
                args.load_features, "trainfeat.pth"))
            test_features = torch.load(os.path.join(
                args.load_features, "testfeat.pth"))
            train_labels = torch.load(os.path.join(
                args.load_features, "trainlabels.pth"))
            test_labels = torch.load(os.path.join(
                args.load_features, "testlabels.pth"))
        except:
            train_features, test_features, train_labels, test_labels = extract_feature_pipeline(
                args)
    else:
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(
            args)

    if args.rank == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top5 = knn_classifier(train_features, train_labels,
                                        test_features, test_labels, k, args.temperature, args.use_cuda)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
    dist.barrier()


def extract_feature_pipeline(args):
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    if args.arch == 'vit-tiny':
        model = ViT(patch_size=args.patch_size, img_size=args.input_size,
                    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4)
    elif args.arch == 'vit-small':
        model = ViT(patch_size=args.patch_size, img_size=args.input_size,
                    embed_dim=384, depth=12, num_heads=12, mlp_ratio=4)

    elif args.arch == 'vit-base':
        model = ViT(patch_size=args.patch_size, img_size=args.input_size,
                    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4)

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()

    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print("=> loading checkpoint '{}'".format(args.pretrained_weights))
            checkpoint = torch.load(
                args.pretrained_weights, map_location=torch.device(args.rank))

            state_dict = get_model_from_frame(checkpoint, args)

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained_weights))
        else:
            print("=> no checkpoint found at '{}'".format(
                args.pretrained_weights))

    model.eval()

    print("Extracting features for train set...")
    train_features, train_labels = extract_features(
        model, data_loader_train, args)
    print("Extracting features for val set...")
    test_features, test_labels = extract_features(
        model, data_loader_val, args)

    if args.rank == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    if args.dump_features and args.get_rank() == 0:
        print("Dumping features ...")
        torch.save(train_features.cpu(), os.path.join(
            args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(
            args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(
            args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(
            args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, args, multiscale=False):
    metric_logger = misc.MetricLogger(delimiter="  ")
    features = None
    labels = None
    for samples, labs, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labs = labs.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        def forward_single(samples):
            output = model(samples)
            return output

        if multiscale:
            v = None
            for s in [1, 1 / 2 ** (1 / 2), 1 / 2]:
                if s == 1:
                    inp = samples.clone()
                else:
                    inp = nn.functional.interpolate(
                        samples, scale_factor=s, mode='bilinear', align_corners=False)
                feats = forward_single(inp)
                if v is None:
                    v = feats
                else:
                    v += feats
            v /= 3
            v /= v.norm()
            feats = v
        else:
            feats = forward_single(samples)

        if args.rank == 0 and features is None:
            features = torch.zeros(
                len(data_loader.dataset), feats.shape[-1]).to(feats.dtype)
            labels = torch.zeros(len(data_loader.dataset)).to(labs.dtype)
            if args.use_cuda:
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
            print(f"Storing labels into tensor of shape {labels.shape}")

        y_all = torch.empty(args.world_size, index.size(
            0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        feats_all = torch.empty(
            args.world_size,
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(
            output_l, feats, async_op=True)
        output_all_reduce.wait()

        labels_all = torch.empty(
            args.world_size,
            labs.size(0),
            dtype=labs.dtype,
            device=labs.device,
        )
        label_l = list(labels_all.unbind(0))
        label_all_reduce = torch.distributed.all_gather(
            label_l, labs, async_op=True)
        label_all_reduce.wait()

        if args.rank == 0:
            if args.use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
                labels.index_copy_(0, index_all, torch.cat(label_l))
            else:
                features.index_copy_(0, index_all.cpu(),
                                     torch.cat(output_l).cpu())
                labels.index_copy_(0, index_all.cpu(),
                                   torch.cat(label_l).cpu())
    return features, labels


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, use_cuda=True, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes)
    if use_cuda:
        retrieval_one_hot = retrieval_one_hot.cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        features = test_features[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        targets = test_labels[idx: min(
            (idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def main_ddp(args):
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.world_size * ngpus_per_node
    mp.spawn(eval_knn, args=(args,), nprocs=args.world_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default='vit-small',
                        choices=['vit-tiny', 'vit-small', 'vit-base'])
    parser.add_argument("--pretrained-weights", type=str,
                        default='')
    return parser


if __name__ == '__main__':
    parser = parse_args()
    _args = parser.parse_args()
    args = knn()
    args.pretrained_weights = _args.pretrained_weights
    args.arch = _args.arch
    main_ddp(args)
