"""ViT (Dosovitskiy et al., 2021) - Eq. 1-4 of the paper, initialization as in the official code."""
import math

import torch
import torch.nn as nn


class Block(nn.Module):
    """Pre-LN Transformer encoder block."""

    def __init__(self, D, n_heads, mlp_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(D, eps=1e-6)
        self.attn = nn.MultiheadAttention(D, n_heads, dropout=0.0, batch_first=True)   # official: attention_dropout_rate = 0
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(D, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(D, mlp_dim),      # [B, N+1, D] -> [B, N+1, mlp_dim]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, D),      # [B, N+1, mlp_dim] -> [B, N+1, D]
            nn.Dropout(dropout),
        )

    def forward(self, z):                                               # z: [B, N+1, D]
        h = self.ln1(z)
        z = z + self.drop(self.attn(h, h, h, need_weights=False)[0])    # Eq. 2: z' = MSA(LN(z)) + z
        z = z + self.mlp(self.ln2(z))                                   # Eq. 3: z = MLP(LN(z')) + z'
        return z                                                        # [B, N+1, D]


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, D=256, n_heads=8, n_blocks=12,
                 mlp_dim=1024, n_classes=100, dropout=0.1, representation_size=None):
        super().__init__()
        N = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_ch, D, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, D))
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(*[Block(D, n_heads, mlp_dim, dropout) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(D, eps=1e-6)
        # Pre-training: MLP head with one hidden layer (tanh); fine-tuning: single linear layer
        self.pre_logits = nn.Sequential(nn.Linear(D, representation_size), nn.Tanh()) if representation_size else nn.Identity()
        self.head = nn.Linear(representation_size or D, n_classes)
        self._init_weights(in_ch * patch_size ** 2)

    def _init_weights(self, patch_fan_in):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)
        nn.init.trunc_normal_(self.patch_embed.weight, std=math.sqrt(1 / patch_fan_in))   # lecun_normal
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x):                             # [B, C, H, W]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)     # [B, N, D]
        cls = self.cls_token.expand(x.shape[0], -1, -1)        # [B, 1, D]
        z = torch.cat([cls, x], dim=1) + self.pos_embed        # Eq. 1: z0 [B, N+1, D]
        z = self.encoder(self.dropout(z))                      # Eq. 2, 3: [B, N+1, D]
        return self.ln(z[:, 0])                                # Eq. 4: y = LN(z_L^0) [B, D]

    def forward(self, x):
        return self.head(self.pre_logits(self.forward_features(x)))   # [B, n_classes]
