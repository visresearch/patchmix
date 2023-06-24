# -*-coding:utf-8-*-
import numpy as np
import torch
from einops import rearrange, repeat


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 1, repeat(indexes, 'b t -> b t c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, patches: torch.Tensor):
        B, T, C = patches.shape
        indexes = random_indexes(T)
        forward_indexes = torch.as_tensor(indexes[0], dtype=torch.long).to(
            patches.device)
        forward_indexes = repeat(
            forward_indexes, 't -> g t', g=B)
        backward_indexes = torch.as_tensor(indexes[1], dtype=torch.long).to(
            patches.device)
        backward_indexes = repeat(
            backward_indexes, 't -> g t', g=B)

        patches = take_indexes(patches, forward_indexes)
        return patches, forward_indexes, backward_indexes


class PatchMix(torch.nn.Module):

    def forward(self, patches: torch.Tensor, m):
        B, T, C = patches.shape
        S = T // m
        mix_offset = int(S * m)
        mix_patches = patches[:, :mix_offset, :]
        mix_patches = rearrange(
            mix_patches, 'g (m s) c -> (g m) s c', s=S)

        L = mix_patches.shape[0]

        ids = torch.arange(L).cuda()
        indexes = (ids + ids % m * m) % L
        mix_patches = torch.gather(mix_patches, 0, repeat(
            indexes, 'l -> l s c', c=mix_patches.shape[-1], s=S))

        ids = torch.arange(B).view(-1, 1)
        target = (ids + torch.arange(m)) % B
        mix_target = ((ids - m + 1) + torch.arange(m * 2 - 1) + B) % B

        mix_patches = rearrange(mix_patches, '(g m) s c -> g (m s) c', g=B)
        patches[:, :mix_offset, :] = mix_patches
        return patches, target, mix_target


class PatchMixer:
    def __init__(self, num_classes, mix_s, mix_n=1, mix_p=0.0, smoothing=0.1):
        self.mix_s = mix_s
        self.mix_p = mix_p
        self.mix_n = mix_n
        self.patch_shuffle = PatchShuffle()
        self.mix = PatchMix()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def _one_hot(self, target, num_classes, on_value=1., off_value=0., device='cuda'):
        return torch.full((target.size()[0], num_classes), off_value, device=device).scatter_(1, target, on_value)

    @torch.no_grad()
    def __call__(self, X, target):
        N = X.shape[0]
        m = np.random.choice(self.mix_n) if isinstance(self.mix_n, list) else self.mix_n
        use_mix = np.random.rand() < self.mix_p and m > 1
        if use_mix:
            patch_size = np.random.choice(self.mix_s) if isinstance(self.mix_s, list) else self.mix_s

            patches = rearrange(
                X, 'b c (w p1) (h p2) -> b (w h) (c p1 p2)', p1=patch_size, p2=patch_size)
            patches, forward_indexes, backward_indexes = self.patch_shuffle(
                patches)
            patches, target, mix_target = self.mix(patches, m)
            patches = take_indexes(patches, backward_indexes)
            X = rearrange(patches, 'b (w h) (c p1 p2) -> b c (w p1) (h p2)', p1=patch_size, p2=patch_size,
                          w=int(np.sqrt(patches.shape[1])))
        else:
            m = 1
            target = target.view(-1, 1)
            mix_target = target
        # add offset
        offset = N * torch.distributed.get_rank()
        target = (target + offset).cuda()
        mix_target = (mix_target + offset).cuda()

        off_value = self.smoothing / self.num_classes
        true_num = target.shape[1]
        on_value = (1.0 - self.smoothing) / true_num + off_value
        soft_target = self._one_hot(
            target, self.num_classes, on_value, off_value)

        ids = torch.arange(mix_target.shape[1])
        weights = (1.0 - torch.abs(m - ids - 1) / m)
        on_value = (1.0 - self.smoothing) * weights / m + off_value
        soft_mix_target = self._one_hot(
            mix_target, self.num_classes, on_value.expand([mix_target.shape[0], -1]).cuda(),
            off_value)
        return X, soft_target, soft_mix_target

