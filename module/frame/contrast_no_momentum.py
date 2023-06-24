import torch.nn as nn
import torch
class ContrastNoMomentum(nn.Module):

    def __init__(self, student, proj_layer, pred_layer, dim=256, mlp_dim=4096):
        super(ContrastNoMomentum, self).__init__()
        # build encoders
        self.base_encoder = student

        self._build_projector_and_predictor_mlps(
            proj_layer, pred_layer, dim, mlp_dim)

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

    def forward(self, x, k=True, q=True):
        if q:
            k = self.base_encoder(x)
            q = self.predictor(k)
        elif k:
            with torch.no_grad():
                k = self.base_encoder(x)
            q = None
        return q, k


class ContrastNoMomentum_ViT(ContrastNoMomentum):
    def _build_projector_and_predictor_mlps(self, proj_layer, pred_layer, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head  # remove original fc layer
        # projectors
        self.base_encoder.head = self._build_mlp(
            proj_layer, hidden_dim, mlp_dim, dim)
        # predictor
        self.predictor = self._build_mlp(pred_layer, dim, mlp_dim, dim)
