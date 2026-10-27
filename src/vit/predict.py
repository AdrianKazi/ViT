"""Inference on a single image.

    python -m vit.predict path/to/image.jpg --ckpt checkpoints/vit_cifar100/vit.pt
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms

from .data import class_names, normalize
from .engine import get_device, load_checkpoint


class Predictor:
    def __init__(self, ckpt_path, device=None, names=None):
        self.device = device or get_device()
        self.model, self.config = load_checkpoint(ckpt_path, self.device)
        self.names = names or class_names()
        size = self.config["img_size"]
        self.tf = transforms.Compose([transforms.Resize((size, size)), normalize])

    @torch.no_grad()
    def __call__(self, img, k=5):
        """img: PIL image (any size) -> top-k [(class, prob)]"""
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)   # [1, 3, H, W]
        probs = self.model(x).softmax(1)[0].float().cpu()
        p, idx = probs.topk(k)
        return [(self.names[i], v.item()) for v, i in zip(p, idx)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("--ckpt", default="checkpoints/vit_cifar100/vit.pt")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    for cls, prob in Predictor(args.ckpt)(Image.open(args.image), k=args.k):
        print(f"{cls:15s} {prob:.3f}")


if __name__ == "__main__":
    main()
