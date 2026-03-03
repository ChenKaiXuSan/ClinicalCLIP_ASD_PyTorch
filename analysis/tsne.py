#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
t-SNE visualization script for ClinicalCLIP experiments.

This script generates t-SNE visualizations comparing different experimental 
configurations for paper publication.

Usage:
    python tsne.py --train-root /path/to/logs/train --output-root /path/to/output
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CLASS_NAMES = {
    2: ["ASD", "non-ASD"],
    3: ["ASD", "DHS", "LCS_HipOA"],
    4: ["ASD", "DHS", "LCS_HipOA", "normal"],
}

# Color palettes for different number of classes
COLOR_PALETTES = {
    2: ["#FF6B6B", "#4ECDC4"],  # Red, Teal
    3: ["#FF6B6B", "#FFD93D", "#6BCB77"],  # Red, Yellow, Green
    4: ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF"],  # Red, Yellow, Green, Blue
}

EXPERIMENT_GROUPS = {
    "B": ["B1_clip_only", "B2_map_only", "B3_full", "B4_full_token"],
    "C": ["C1_channel_gate", "C2_weighted_pool", "C3_sigmoid_gate"],
}

EXPERIMENT_LABELS = {
    "B1_clip_only": "CLIP Only",
    "B2_map_only": "Map Only",
    "B3_full": "Full",
    "B4_full_token": "Full + Token",
    "C1_channel_gate": "Channel Gate",
    "C2_weighted_pool": "Weighted Pool",
    "C3_sigmoid_gate": "Sigmoid Gate",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate t-SNE visualizations for ClinicalCLIP experiments."
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("/work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/logs/train"),
        help="Root directory containing experiment logs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/logs/tsne_results"),
        help="Output directory for t-SNE visualizations.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Specific experiments to visualize. If not set, all are used.",
    )
    parser.add_argument(
        "--embed-type",
        type=str,
        default="video_embed",
        choices=["video_embed", "attn_embed", "both"],
        help="Type of embeddings to visualize.",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="Perplexity for t-SNE.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="Number of iterations for t-SNE.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output figures.",
    )
    return parser.parse_args()


def load_embeddings(
    embeddings_dir: Path,
    embed_type: str = "video_embed",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from directory.
    
    Args:
        embeddings_dir: Directory containing embedding files
        embed_type: Type of embeddings to load ("video_embed" or "attn_embed")
    
    Returns:
        Tuple of (embeddings, labels) as numpy arrays
    """
    embeddings_list = []
    labels_list = []
    
    fold = 0
    while True:
        embed_file = embeddings_dir / f"{fold}_{embed_type}.pt"
        label_file = embeddings_dir / f"{fold}_label.pt"
        
        if not embed_file.exists() or not label_file.exists():
            if fold == 0:
                raise FileNotFoundError(
                    f"No embeddings found in {embeddings_dir}"
                )
            break
        
        try:
            embed = torch.load(embed_file, map_location="cpu")
            label = torch.load(label_file, map_location="cpu")
            
            if isinstance(embed, torch.Tensor):
                embed = embed.numpy()
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            
            embeddings_list.append(embed)
            labels_list.append(label)
            logger.info(f"Loaded fold {fold}: embeddings {embed.shape}, labels {label.shape}")
            
            fold += 1
        except Exception as e:
            logger.error(f"Error loading fold {fold}: {e}")
            break
    
    if not embeddings_list:
        raise ValueError(f"No valid embeddings found in {embeddings_dir}")
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    logger.info(f"Total embeddings: {embeddings.shape}, labels: {labels.shape}")
    return embeddings, labels


def find_latest_run(experiment_dir: Path) -> Optional[Path]:
    """Find the latest run directory for an experiment."""
    if not experiment_dir.exists():
        return None
    
    date_dirs = sorted(experiment_dir.glob("20*"))
    if not date_dirs:
        return None
    
    latest_date = date_dirs[-1]
    time_dirs = sorted(latest_date.glob("*-*-*"))
    if not time_dirs:
        return None
    
    embeddings_dir = time_dirs[-1] / "embeddings"
    if embeddings_dir.exists():
        return embeddings_dir
    
    return None


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_seed: int = 42,
) -> np.ndarray:
    """Compute t-SNE representation.
    
    Args:
        embeddings: Input embeddings array
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_seed: Random seed
    
    Returns:
        t-SNE embeddings
    """
    logger.info(
        f"Computing t-SNE with perplexity={perplexity}, "
        f"max_iter={n_iter}, random_seed={random_seed}"
    )
    
    # Adjust perplexity based on dataset size
    effective_perplexity = min(perplexity, (embeddings.shape[0] - 1) // 3)
    
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=n_iter,  # Use max_iter instead of n_iter for newer sklearn versions
        random_state=random_seed,
        verbose=1,
        n_jobs=-1,  # Use all CPU cores
    )
    
    tsne_embeddings = tsne.fit_transform(embeddings)
    logger.info(f"t-SNE computation complete: {tsne_embeddings.shape}")
    
    return tsne_embeddings


def plot_tsne(
    tsne_embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Visualization",
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Plot t-SNE visualization.
    
    Args:
        tsne_embeddings: t-SNE embeddings
        labels: Class labels
        title: Plot title
        output_path: Path to save figure
        dpi: DPI for output
    """
    num_classes = len(np.unique(labels))
    class_names = CLASS_NAMES.get(num_classes, [f"Class {i}" for i in range(num_classes)])
    colors = COLOR_PALETTES.get(num_classes, sns.color_palette("husl", num_classes))
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
    
    # Plot each class separately for better legend
    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            tsne_embeddings[mask, 0],
            tsne_embeddings[mask, 1],
            c=[colors[i]],
            label=class_name,
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )
    
    ax.set_xlabel("t-SNE 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=12, loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")
    
    plt.close()


def plot_multiple_tsne(
    tsne_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Plot multiple t-SNE visualizations in a grid.
    
    Args:
        tsne_data: Dictionary of {experiment_name: (tsne_embeddings, labels)}
        output_path: Path to save figure
        dpi: DPI for output
    """
    num_experiments = len(tsne_data)
    ncols = min(3, num_experiments)
    nrows = (num_experiments + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 5 * nrows),
        dpi=dpi
    )
    
    if num_experiments == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    
    # Hide extra subplots
    for idx in range(num_experiments, len(axes)):
        axes[idx].set_visible(False)
    
    for idx, (exp_name, (tsne_embeddings, labels)) in enumerate(tsne_data.items()):
        ax = axes[idx]
        
        num_classes = len(np.unique(labels))
        class_names = CLASS_NAMES.get(num_classes, [f"Class {i}" for i in range(num_classes)])
        colors = COLOR_PALETTES.get(num_classes, sns.color_palette("husl", num_classes))
        
        # Plot each class
        for i, class_name in enumerate(class_names):
            mask = labels == i
            ax.scatter(
                tsne_embeddings[mask, 0],
                tsne_embeddings[mask, 1],
                c=[colors[i]],
                label=class_name,
                s=80,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )
        
        exp_label = EXPERIMENT_LABELS.get(exp_name, exp_name)
        ax.set_title(exp_label, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        
        if idx == 0:
            ax.legend(fontsize=10, loc="best", framealpha=0.95)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")
    
    plt.close()


def main():
    """Main function."""
    args = parse_args()
    
    train_root = args.train_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Determine experiments to process
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = []
        for exp_list in EXPERIMENT_GROUPS.values():
            experiments.extend(exp_list)
    
    logger.info(f"Processing experiments: {experiments}")
    
    # Process individual experiments
    tsne_data_by_group = {"B": {}, "C": {}}
    
    for exp_name in experiments:
        exp_dir = train_root / exp_name
        embeddings_dir = find_latest_run(exp_dir)
        
        if embeddings_dir is None:
            logger.warning(f"No embeddings found for {exp_name}")
            continue
        
        logger.info(f"\nProcessing {exp_name} from {embeddings_dir.parent.parent}")
        
        # Load embeddings for both types
        embed_types_to_process = []
        if args.embed_type in ["video_embed", "both"]:
            embed_types_to_process.append("video_embed")
        if args.embed_type in ["attn_embed", "both"]:
            embed_types_to_process.append("attn_embed")
        
        for embed_type in embed_types_to_process:
            try:
                embeddings, labels = load_embeddings(embeddings_dir, embed_type)
                
                # Compute t-SNE
                tsne_embeddings = compute_tsne(
                    embeddings,
                    perplexity=args.perplexity,
                    n_iter=args.n_iter,
                    random_seed=args.random_seed,
                )
                
                # Save individual plot
                output_file = output_root / f"{exp_name}_{embed_type}_tsne.png"
                plot_tsne(
                    tsne_embeddings,
                    labels,
                    title=f"{EXPERIMENT_LABELS.get(exp_name, exp_name)} - {embed_type.replace('_', ' ').title()}",
                    output_path=output_file,
                    dpi=args.dpi,
                )
                
                # Store for group comparison
                group = exp_name[0]  # 'B' or 'C'
                if group in tsne_data_by_group:
                    tsne_data_by_group[group][exp_name] = (tsne_embeddings, labels)
                    
            except Exception as e:
                logger.error(f"Error processing {exp_name} with {embed_type}: {e}")
                continue
    
    # Generate group comparison plots
    for group, experiments_list in EXPERIMENT_GROUPS.items():
        group_data = {
            exp: tsne_data_by_group[group][exp]
            for exp in experiments_list
            if exp in tsne_data_by_group[group]
        }
        
        if group_data:
            output_file = output_root / f"comparison_group_{group}_tsne.png"
            logger.info(f"\nGenerating comparison plot for group {group}")
            plot_multiple_tsne(
                group_data,
                output_path=output_file,
                dpi=args.dpi,
            )
    
    logger.info(f"\nAll t-SNE visualizations saved to {output_root}")


if __name__ == "__main__":
    main()
