"""Train a small ViT from scratch on CIFAR-100.

    python -m vit.train --out checkpoints/vit_cifar100
"""
import argparse
import time

import torch
import torch.nn as nn

from .data import load_cifar100, make_loader, scratch_transforms
from .engine import autocast, evaluate, get_device, save_checkpoint, warmup_cosine
from .model import ViT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--mlp-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training - paper recipe (Adam, warmup + cosine, grad clip)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--out", default="checkpoints/vit_cifar100")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    torch.manual_seed(0)

    ds = load_cifar100()
    tf_train, tf_test = scratch_transforms(args.img_size)
    train_loader = make_loader(ds["train"], tf_train, args.batch_size, train=True, num_workers=args.num_workers)
    test_loader = make_loader(ds["test"], tf_test, 1000, train=False, num_workers=args.num_workers)

    config = dict(img_size=args.img_size, patch_size=args.patch_size, in_ch=3, D=args.dim, n_heads=args.heads,
                  n_blocks=args.blocks, mlp_dim=args.mlp_dim, n_classes=100, dropout=args.dropout)
    model = ViT(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = warmup_cosine(optimizer, args.warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss()
    print(f"device: {device} | params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | steps: {total_steps}")

    for epoch in range(args.epochs):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with autocast(device):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
        log = (f"epoch {epoch:3d} | loss {loss_sum / seen:.3f} | train acc {correct / seen:.3f} | "
               f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s")
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            log += f" | test acc {evaluate(model, test_loader, device)[0]:.3f}"
        print(log, flush=True)

    test_top1, test_top5 = evaluate(model, test_loader, device)
    train_config = {k: v for k, v in vars(args).items() if k not in ("out", "num_workers")}
    save_checkpoint(args.out, model, config, {"train_config": train_config, "test_top1": test_top1, "test_top5": test_top5},
                    optimizer=optimizer)
    print(f"saved to {args.out} | test top1 {test_top1:.4f} | top5 {test_top5:.4f}")


if __name__ == "__main__":
    main()
