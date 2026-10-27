"""Gradio demo: upload an image, get CIFAR-100 top-5 predictions.

    python app/app.py --ckpt checkpoints/vit_b16_in21k_cifar100/vit.pt
"""
import argparse

import gradio as gr

from vit.predict import Predictor


def build(ckpt):
    predictor = Predictor(ckpt)
    size = predictor.config["img_size"]

    def classify(img):
        if img is None:
            return {}
        return dict(predictor(img, k=5))

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Image"),
        outputs=gr.Label(num_top_classes=5, label="Top-5"),
        title="Vision Transformer - CIFAR-100",
        description=f"ViT implemented from scratch in PyTorch ({predictor.config['D']}-dim, "
                    f"{predictor.config['n_blocks']} blocks, {size}px input). Classifies into the 100 CIFAR-100 classes.",
        flagging_mode="never",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/vit_cifar100/vit.pt")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    build(args.ckpt).launch(server_port=args.port)
