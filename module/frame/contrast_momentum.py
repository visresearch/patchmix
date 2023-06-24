import copy

import torch
import torch.nn as nn

class ContrastMomentum(nn.Module):

    def __init__(self, base_encoder, momentum_encoder, proj_layer, pred_layer, dim=256, mlp_dim=4096):
        super(ContrastMomentum, self).__init__()
        # build encoders
        self.base_encoder = base_encoder
        self.momentum_encoder = momentum_encoder

        self._build_projector_and_predictor_mlps(
            proj_layer, pred_layer, dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
    def _build_projector_and_predictor_mlps(self, proj_layer, pred_layer, dim, mlp_dim):
        pass

    @torch.no_grad()
    def update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def forward(self, x, k=True, q=True):
        q = self.predictor(self.base_encoder(x)) if q else None
        k = self.momentum_encoder(x) if k else None
        return q, k


class ContrastMomentum_ViT(ContrastMomentum):
    def _build_projector_and_predictor_mlps(self, proj_layer, pred_layer, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head  # remove original fc layer
        # # projectors
        self.base_encoder.head = self._build_mlp(proj_layer, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(proj_layer, hidden_dim, mlp_dim, dim)
        # predictor
        self.predictor = self._build_mlp(pred_layer, dim, mlp_dim, dim)
