import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer


class MultiPatchEmbed(PatchEmbed):
    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patch_embed = MultiPatchEmbed(
            img_size=self.patch_embed.img_size[0],
            patch_size=self.patch_embed.patch_size[0],
            embed_dim=self.embed_dim,
            bias=True,
        )
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    val = math.sqrt(
                        6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            val = math.sqrt(
                6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

    def _pos_embed(self, x, w, h):
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(
                x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(
                math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(
            w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_features(self, x):
        b, t, w, h = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x, w, h)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def ViT(**kwargs):
    model = VisionTransformerMoCo(
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
