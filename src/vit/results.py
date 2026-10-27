"""Compare our models with the paper (Table 2, CIFAR-100) and save the plot.

    python -m vit.results
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = {   # Table 2, CIFAR-100 top-1 [%]
    "Paper ViT-L/16\n(ImageNet-21k)": 93.25,
    "Paper BiT-L\n(ResNet, JFT)": 93.51,
    "Paper ViT-L/16\n(JFT)": 93.90,
    "Paper ViT-H/14\n(JFT)": 94.55,
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scratch", default="checkpoints/vit_cifar100/results.json")
    p.add_argument("--finetuned", default="checkpoints/vit_b16_in21k_cifar100/results.json")
    p.add_argument("--out", default="assets/results.png")
    args = p.parse_args()

    results, colors = {"Random": 1.0}, ["grey"]
    if scratch := load(args.scratch):
        results["Our ViT\n(scratch)"] = scratch["test_top1"] * 100
        colors.append("tab:red")
    if finetuned := load(args.finetuned):
        size = finetuned["config"]["img_size"]
        results[f"Our ViT-B/16\n(ImageNet-21k, FT {size})"] = finetuned["test_top1"] * 100
        colors.append("tab:red")
    results |= PAPER
    colors += ["tab:orange", "tab:blue", "tab:orange", "tab:orange"]

    for name, acc in results.items():
        print(f"{name.replace(chr(10), ' '):40s} {acc:5.1f}%")

    plt.figure(figsize=(12, 5))
    bars = plt.bar(results.keys(), results.values(), color=colors)
    plt.bar_label(bars, fmt="%.1f%%")
    plt.ylabel("CIFAR-100 top-1 accuracy [%]")
    plt.title("Our ViT vs paper results (Table 2)")
    plt.ylim(0, 105)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
