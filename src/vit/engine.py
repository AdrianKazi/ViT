"""Shared training utilities: device, LR schedule, evaluation, checkpoints."""
import json
import math
import os

import torch

from .model import ViT


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    """bf16 autocast on CUDA only; MPS / CPU run in fp32."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def warmup_cosine(optimizer, warmup_steps, total_steps):
    """Linear warmup, then cosine decay to 0 (per step)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (top1, top5) accuracy."""
    was_training = model.training
    model.eval()
    top1 = top5 = n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with autocast(device):
            logits = model(xb)
        top1 += (logits.argmax(1) == yb).sum().item()
        top5 += (logits.topk(5, dim=1).indices == yb[:, None]).any(1).sum().item()
        n += len(yb)
    model.train(was_training)
    return top1 / n, top5 / n


def save_checkpoint(out_dir, model, config, results, optimizer=None):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {"model": model.state_dict(), "config": config}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, os.path.join(out_dir, "vit.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"config": config, **results}, f, indent=2)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ViT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["config"]
