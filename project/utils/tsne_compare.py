#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run t-SNE for saved CLIP video/attention embeddings and compare distributions."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="Directory containing *_video_embed.pt, *_attn_embed.pt, *_label.pt",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0",
        help="Fold name prefix used in saved embedding files, e.g., 0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <embeddings-dir>/<fold>_tsne_compare.png",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=2000)
    return parser.parse_args()


def load_embeddings(emb_dir: Path, fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    video_path = emb_dir / f"{fold}_video_embed.pt"
    attn_path = emb_dir / f"{fold}_attn_embed.pt"
    label_path = emb_dir / f"{fold}_label.pt"

    if not video_path.exists() or not attn_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Missing embedding files for fold={fold} under {emb_dir}. "
            f"Need {video_path.name}, {attn_path.name}, {label_path.name}."
        )

    video = torch.load(video_path, map_location="cpu").numpy()
    attn = torch.load(attn_path, map_location="cpu").numpy()
    labels = torch.load(label_path, map_location="cpu").numpy()

    if video.shape[0] != attn.shape[0] or video.shape[0] != labels.shape[0]:
        raise ValueError("video/attn/label sample counts are inconsistent.")

    return video, attn, labels


def run_tsne(video: np.ndarray, attn: np.ndarray, perplexity: float, random_state: int, n_iter: int):
    features = np.concatenate([video, attn], axis=0)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=n_iter,
        init="pca",
        learning_rate="auto",
    )
    z = tsne.fit_transform(features)
    n = video.shape[0]
    return z[:n], z[n:]


def plot_tsne(video_z: np.ndarray, attn_z: np.ndarray, labels: np.ndarray, output: Path, fold: str) -> None:
    classes = sorted(np.unique(labels).tolist())
    cmap = plt.cm.get_cmap("tab10", max(10, len(classes)))

    plt.figure(figsize=(10, 8))

    for idx, cls in enumerate(classes):
        mask = labels == cls
        color = cmap(idx)

        plt.scatter(
            video_z[mask, 0],
            video_z[mask, 1],
            c=[color],
            marker="o",
            alpha=0.65,
            s=24,
            label=f"video class={cls}",
        )
        plt.scatter(
            attn_z[mask, 0],
            attn_z[mask, 1],
            c=[color],
            marker="^",
            alpha=0.65,
            s=24,
            label=f"attn class={cls}",
        )

    plt.title(f"t-SNE Comparison (fold={fold})")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.grid(True, alpha=0.2)

    handles, labels_txt = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels_txt, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    output = args.output or (args.embeddings_dir / f"{args.fold}_tsne_compare.png")

    video, attn, labels = load_embeddings(args.embeddings_dir, args.fold)
    video_z, attn_z = run_tsne(
        video,
        attn,
        perplexity=args.perplexity,
        random_state=args.random_state,
        n_iter=args.n_iter,
    )
    plot_tsne(video_z, attn_z, labels, output, args.fold)
    print(f"saved tsne figure to: {output}")


if __name__ == "__main__":
    main()
