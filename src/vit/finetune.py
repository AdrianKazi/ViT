"""Fine-tune official ViT weights (ImageNet-21k) on CIFAR-100, as in the paper (Appendix B.1.1, Table 4).

Paper setup (default, needs a CUDA GPU, ~2 h per learning rate on an A100):
    python -m vit.finetune

Reduced setup that fits a laptop (MPS, ~3.5 h on an M2 Pro):
    python -m vit.finetune --img-size 224 --steps 500 --warmup-steps 25 --lrs 0.03 --micro-batch 32
"""
import argparse
import time

import torch
import torch.nn as nn

from .data import finetune_transforms, load_cifar100, make_loader, minival_split
from .engine import autocast, evaluate, get_device, save_checkpoint, warmup_cosine
from .pretrained import OFFICIAL, load_official, official_state_dict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="B/16", choices=list(OFFICIAL))
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=64, help="per-step batch; gradients are accumulated up to --batch-size")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--lrs", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0, help="official fine-tuning config: no dropout")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--out", default="checkpoints/vit_b16_in21k_cifar100")
    return p.parse_args()


def infinite(loader):
    while True:
        yield from loader


def finetune(lr, sd, train_loader, args, device):
    model, config = load_official(args.variant, args.img_size, n_classes=100, dropout=args.dropout, sd=sd)
    model = model.to(device).train()
    # SGD with momentum, no weight decay, warmup + cosine, grad clip at global norm 1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=0.0)
    scheduler = warmup_cosine(optimizer, args.warmup_steps, args.steps)
    loss_fn = nn.CrossEntropyLoss()
    accum_steps = args.batch_size // args.micro_batch

    data = infinite(train_loader)
    t0 = time.time()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum_steps):                          # e.g. 8 x 64 = 512
            xb, yb = next(data)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with autocast(device):
                loss = loss_fn(model(xb), yb) / accum_steps
            loss.backward()
            loss_sum += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"lr {lr} | step {step:5d}/{args.steps} | loss {loss_sum:.3f} | {time.time() - t0:.0f}s", flush=True)
    return model, config


def main():
    args = parse_args()
    assert args.batch_size % args.micro_batch == 0, "--batch-size must be divisible by --micro-batch"
    device = get_device()
    torch.manual_seed(0)

    ds = load_cifar100()
    tf_train, tf_test = finetune_transforms(args.img_size)
    train_split, val_split = minival_split(ds["train"])
    train_loader = make_loader(train_split, tf_train, args.micro_batch, train=True, num_workers=args.num_workers)
    val_loader = make_loader(val_split, tf_test, 256, train=False, num_workers=args.num_workers)
    test_loader = make_loader(ds["test"], tf_test, 256, train=False, num_workers=args.num_workers)

    sd = official_state_dict(args.variant)
    print(f"device: {device} | ViT-{args.variant} @ {args.img_size} | lrs {args.lrs}")

    # LR sweep: each lr starts from the same official weights, best one is picked on minival (not on test)
    minival, best_lr, best = {}, None, None
    for lr in args.lrs:
        model, config = finetune(lr, sd, train_loader, args, device)
        minival[lr], _ = evaluate(model, val_loader, device)
        print(f"lr {lr} -> minival top1 {minival[lr]:.4f}", flush=True)
        if best_lr is None or minival[lr] > minival[best_lr]:
            best_lr, best = lr, model
        else:
            del model

    test_top1, test_top5 = evaluate(best, test_loader, device)
    train_config = {k: v for k, v in vars(args).items() if k not in ("out", "num_workers")} | {"best_lr": best_lr}
    save_checkpoint(args.out, best, config, {"train_config": train_config, "minival_top1": minival,
                                             "test_top1": test_top1, "test_top5": test_top5})
    print(f"saved to {args.out} | best lr {best_lr} | test top1 {test_top1:.4f} | top5 {test_top5:.4f}")


if __name__ == "__main__":
    main()
