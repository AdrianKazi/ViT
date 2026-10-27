"""Publish a checkpoint to the Hugging Face Hub (safetensors + config + results + model card).

    python -m vit.hub checkpoints/vit_b16_in21k_cifar100 --repo <user>/vit-b16-in21k-cifar100
"""
import argparse
import json
import os
import shutil
import tempfile

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file

from .model import ViT

CARD = """---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-classification
datasets:
- uoft-cs/cifar100
tags:
- vision-transformer
- vit
- image-classification
- cifar100
metrics:
- accuracy
---

# {title}

Vision Transformer ([Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)) implemented from scratch in PyTorch, {what}

| | |
|---|---|
| Architecture | ViT, D={D}, {n_blocks} blocks, {n_heads} heads, MLP {mlp_dim}, patch {patch_size}, input {img_size}px |
| Parameters | {params:.1f}M |
| Dataset | CIFAR-100 |
| Test top-1 | **{top1:.2f}%** |
| Test top-5 | {top5:.2f}% |

## Training

```json
{train_config}
```

## Usage

```python
from vit.hub import from_pretrained
model = from_pretrained("{repo}")   # needs the `vit` package from {code}
```

Inputs: RGB images resized to {img_size}x{img_size}, pixels scaled to [-1, 1]. Classes: the 100 CIFAR-100 fine labels.

Code: {code}
"""


def to_safetensors(ckpt_path, out_dir):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in ckpt["model"].items()}, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(ckpt["config"], f, indent=2)
    return ckpt


def from_pretrained(repo, device="cpu"):
    from safetensors.torch import load_file
    with open(hf_hub_download(repo, "config.json")) as f:
        config = json.load(f)
    model = ViT(**config)
    model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt_dir", help="folder with vit.pt and results.json")
    p.add_argument("--repo", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--code", default="https://github.com/AdrianKazi/ViT")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    with open(os.path.join(args.ckpt_dir, "results.json")) as f:
        results = json.load(f)
    finetuned = "minival_top1" in results
    what = ("with the official ViT-B/16 weights (Google, ImageNet-21k) loaded 1:1 into the implementation and fine-tuned on CIFAR-100."
            if finetuned else "trained from scratch on CIFAR-100 (no pre-training).")

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = to_safetensors(os.path.join(args.ckpt_dir, "vit.pt"), tmp)
        shutil.copy(os.path.join(args.ckpt_dir, "results.json"), tmp)
        cfg = ckpt["config"]
        params = sum(p.numel() for p in ViT(**cfg).parameters()) / 1e6
        card = CARD.format(title=args.title or args.repo.split("/")[-1], what=what, params=params, repo=args.repo,
                           top1=results["test_top1"] * 100, top5=results["test_top5"] * 100, code=args.code,
                           train_config=json.dumps(results.get("train_config", {}), indent=2), **cfg)
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(card)

        api = HfApi()
        api.create_repo(args.repo, private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=tmp, commit_message="Upload ViT checkpoint")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
