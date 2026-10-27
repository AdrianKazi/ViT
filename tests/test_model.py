import os

import pytest
import torch

from vit.model import ViT
from vit.pretrained import interpolate_pos_embed


def test_forward_shape():
    model = ViT(img_size=32, patch_size=4, D=64, n_heads=4, n_blocks=2, mlp_dim=128, n_classes=100)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)


def test_patch_embed_equals_manual_patching():
    """Conv2d(kernel=stride=P) == unfold -> permute -> reshape -> Linear."""
    B, C, H, P, D = 2, 3, 32, 4, 64
    model = ViT(img_size=H, patch_size=P, D=D, n_heads=4, n_blocks=1, mlp_dim=128)
    x = torch.randn(B, C, H, H)
    conv = model.patch_embed(x).flatten(2).transpose(1, 2)                            # [B, N, D]
    patches = x.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * P * P)
    manual = patches @ model.patch_embed.weight.reshape(D, -1).T + model.patch_embed.bias
    assert torch.allclose(conv, manual, atol=1e-5)


def test_interpolate_pos_embed():
    pos = torch.randn(1, 1 + 14 * 14, 32)
    new = interpolate_pos_embed(pos, 24)
    assert new.shape == (1, 1 + 24 * 24, 32)
    assert torch.equal(new[:, 0], pos[:, 0])                                            # [CLS] untouched
    assert torch.equal(interpolate_pos_embed(pos, 14), pos)


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads official weights (~350 MB), set RUN_SLOW=1")
def test_official_weights_match_timm():
    """Official ViT-B/16 weights loaded into our class give the same [CLS] representation as timm."""
    import timm
    from vit.pretrained import OFFICIAL, load_official

    official = timm.create_model(OFFICIAL["B/16"]["timm_name"], pretrained=True, num_classes=0).eval()
    ours, _ = load_official("B/16", img_size=224, sd=official.state_dict())
    ours.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(ours.forward_features(x), official.forward_features(x)[:, 0], atol=1e-4)
