# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor, target: torch.Tensor, use_neg=True) -> torch.Tensor:
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        if use_neg:
            k = concat_all_gather(k)
            logits = torch.einsum('nc,mc->nm', [q, k]) / self.temperature
            return SoftTargetCrossEntropy()(logits, target)
        else:
            k = k.detach()
            return -(nn.CosineSimilarity(dim=1)(q, k).mean())


class MultiTempContrastiveLoss(nn.Module):

    def forward(self, q: torch.Tensor, k: torch.Tensor, target: torch.Tensor, temperature,
                use_neg=True) -> torch.Tensor:
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        if use_neg:
            k = concat_all_gather(k)
            logits = torch.einsum('nc,mc->nm', [q, k]) / temperature
            return SoftTargetCrossEntropy()(logits, target)
        else:
            k = k.detach()
            return -(nn.CosineSimilarity(dim=1)(q, k).mean())


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
