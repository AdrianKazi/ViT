"""Official ViT weights (Google, ImageNet-21k) loaded 1:1 into our ViT class."""
import torch
import torch.nn.functional as F

from .model import ViT

# Paper checkpoints in timm, pre-trained on ImageNet-21k at 224 px (head removed)
OFFICIAL = {
    "B/16": dict(timm_name="vit_base_patch16_224.orig_in21k", patch_size=16, D=768, n_heads=12, n_blocks=12, mlp_dim=3072),
    "L/16": dict(timm_name="vit_large_patch16_224.orig_in21k", patch_size=16, D=1024, n_heads=16, n_blocks=24, mlp_dim=4096),
    "H/14": dict(timm_name="vit_huge_patch14_224.orig_in21k", patch_size=14, D=1280, n_heads=16, n_blocks=32, mlp_dim=5120),
}
PRETRAIN_IMG_SIZE = 224


def interpolate_pos_embed(pos_embed, new_n):
    """[1, 1 + n*n, D] -> [1, 1 + new_n*new_n, D], 2D bilinear interpolation of the patch grid, [CLS] untouched."""
    cls_pos, grid = pos_embed[:, :1], pos_embed[:, 1:]
    n = int(grid.shape[1] ** 0.5)
    if n == new_n:
        return pos_embed
    grid = grid.reshape(1, n, n, -1).permute(0, 3, 1, 2)                              # [1, D, n, n]
    grid = F.interpolate(grid, size=(new_n, new_n), mode="bilinear", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_n * new_n, -1)                     # [1, new_n², D]
    return torch.cat([cls_pos, grid], dim=1)


def official_state_dict(variant="B/16"):
    import timm
    official = timm.create_model(OFFICIAL[variant]["timm_name"], pretrained=True, num_classes=0)
    return official.state_dict()


def map_state_dict(sd, n_blocks, new_grid):
    """timm key names -> our key names. timm's fused qkv has the same q, k, v order as nn.MultiheadAttention.in_proj."""
    new = {
        "cls_token": sd["cls_token"],
        "pos_embed": interpolate_pos_embed(sd["pos_embed"], new_grid),
        "patch_embed.weight": sd["patch_embed.proj.weight"],
        "patch_embed.bias": sd["patch_embed.proj.bias"],
        "ln.weight": sd["norm.weight"],
        "ln.bias": sd["norm.bias"],
    }
    for i in range(n_blocks):
        o, m = f"blocks.{i}.", f"encoder.{i}."
        new |= {
            m + "ln1.weight": sd[o + "norm1.weight"],               m + "ln1.bias": sd[o + "norm1.bias"],
            m + "attn.in_proj_weight": sd[o + "attn.qkv.weight"],   m + "attn.in_proj_bias": sd[o + "attn.qkv.bias"],
            m + "attn.out_proj.weight": sd[o + "attn.proj.weight"], m + "attn.out_proj.bias": sd[o + "attn.proj.bias"],
            m + "ln2.weight": sd[o + "norm2.weight"],               m + "ln2.bias": sd[o + "norm2.bias"],
            m + "mlp.0.weight": sd[o + "mlp.fc1.weight"],           m + "mlp.0.bias": sd[o + "mlp.fc1.bias"],
            m + "mlp.3.weight": sd[o + "mlp.fc2.weight"],           m + "mlp.3.bias": sd[o + "mlp.fc2.bias"],
        }
    return new


def load_official(variant="B/16", img_size=384, n_classes=100, dropout=0.0, sd=None):
    """Our ViT with official pre-trained weights; the new head stays zero-initialized (paper, Sec. 3.2)."""
    arch = {k: v for k, v in OFFICIAL[variant].items() if k != "timm_name"}
    config = dict(img_size=img_size, in_ch=3, n_classes=n_classes, dropout=dropout, **arch)
    model = ViT(**config)
    sd = sd if sd is not None else official_state_dict(variant)
    missing, unexpected = model.load_state_dict(map_state_dict(sd, arch["n_blocks"], img_size // arch["patch_size"]), strict=False)
    assert set(missing) == {"head.weight", "head.bias"} and not unexpected, (missing, unexpected)
    return model, config
