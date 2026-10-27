"""CIFAR-100 from Hugging Face, preprocessing as in the official ViT code (pixels in [-1, 1])."""
import os
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

DATASET = "uoft-cs/cifar100"

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def scratch_transforms(img_size):
    """From-scratch training on 32x32: random crop + flip."""
    train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])
    return train, normalize


def finetune_transforms(img_size):
    """Fine-tuning: Inception crop + flip for train, plain resize for eval."""
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])
    test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        normalize,
    ])
    return train, test


def collate(batch, t):
    x = torch.stack([t(ex["img"].convert("RGB")) for ex in batch])   # [B, 3, H, W]
    y = torch.tensor([ex["fine_label"] for ex in batch])             # [B]
    return x, y


def default_num_workers():
    return min(8, os.cpu_count() or 1)


def load_cifar100():
    return load_dataset(DATASET)


def class_names(ds=None):
    ds = ds or load_cifar100()
    return ds["train"].features["fine_label"].names


def make_loader(split, t, batch_size, train, num_workers=None):
    return DataLoader(split, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=default_num_workers() if num_workers is None else num_workers,
                      collate_fn=partial(collate, t=t), persistent_workers=False)


def minival_split(train_split, frac=0.02, seed=0):
    """Paper: 2% of the training set is held out as minival to pick hyperparameters."""
    perm = torch.randperm(len(train_split), generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = int(frac * len(perm))
    return Subset(train_split, perm[n_val:]), Subset(train_split, perm[:n_val])
