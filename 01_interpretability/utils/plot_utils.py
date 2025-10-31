"""
Plotting Utilities for ESM2 Attention Analysis

This module contains plotting and visualization functions for attention matrices and statistics
from ESM2 protein foundation models. It is intended to be used alongside utils.py, which provides
data extraction and analysis utilities.

Functions in this module include:
    - plot_attention_heatmap
    - compare_attention_layers
    - plot_attention_difference
    - plot_variant_attention_profile
    - compare_attention_layers_difference
    - compare_attention_layers_overlay
    - compare_specific_heads

Requirements:
    - numpy, matplotlib
    - ESM2 model and tokenizer (for upstream data extraction)

References:
    - ESM2: https://github.com/facebookresearch/esm
    - Attention mechanisms: https://arxiv.org/abs/1706.03762

Note: Many functions display or save plots using matplotlib. See individual docstrings for details.
"""

from typing import Optional, Tuple, Union
from utils import get_attention_stats
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


def plot_attention_heatmap(
    attention: np.ndarray,
    title: str,
    variant_pos: Optional[int] = None,
    save_path: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    window: Optional[int] = None,
) -> Axes:
    """
    Plot attention heatmap with optional variant position marker.

    Args:
        attention: [seq_len, seq_len] attention matrix
        title: Plot title
        variant_pos: Position of variant (1-indexed from variant_info)
        save_path: Path to save figure (None = don't save)
        ax: Matplotlib axis to plot on (None = create new figure)
        figsize: Figure size tuple (only used if ax is None)
        cmap: Colormap name
        window: If specified, show only +/- window residues around variant_pos

    Returns:
        ax: The axis object

    Note:
        - If ax is None, a new figure is created and optionally saved to save_path.
        - If ax is provided, the plot is drawn on the given axis and not shown or saved
        automatically.
        - The plot is displayed only if called in an interactive environment or if plt.show() is
        called externally.
    """
    # Create figure if no axis provided
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Apply windowing if requested
    if window is not None and variant_pos is not None:
        # Convert to 0-indexed for slicing
        center = variant_pos - 1
        start = max(0, center - window)
        end = min(attention.shape[0], center + window + 1)

        attention = attention[start:end, start:end]

        # Adjust variant position to windowed coordinates
        variant_pos_plot = center - start

        # Update title to show window
        title = f"{title} (window: {start+1}-{end})"
    else:
        variant_pos_plot = variant_pos - 1 if variant_pos is not None else None

    im = ax.imshow(attention, cmap=cmap, aspect="auto")
    ax.set_xlabel("Attending to position", fontsize=12)
    ax.set_ylabel("Attending from position", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Colorbar
    plt.colorbar(im, ax=ax, label="Attention weight")

    # Mark variant position if provided
    if variant_pos_plot is not None:
        ax.axhline(
            variant_pos_plot,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Variant position",
        )
        ax.axvline(
            variant_pos_plot,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
        ax.legend(fontsize=10)

    if save_path and ax is None:  # Only save if we created the figure
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")

    return ax


def compare_attention_layers(
    all_layers_attention: np.ndarray,
    save_path: Optional[str] = None,
    axes: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> np.ndarray:
    """
    Plot attention statistics across all layers.

    Args:
        all_layers_attention: [num_layers, seq_len, seq_len]
        save_path: Path to save figure (only if axes=None)
        axes: Optional 2x2 array of matplotlib axes
        title: Optional suptitle for the plot

    Returns:
        axes: The 2x2 axes array

    Note:
        - If axes is None, a new figure is created and optionally saved to save_path.
        - The plot is displayed only if plt.show() is called externally.
    """
    # Compute stats
    stats = []
    for i, layer_attn in enumerate(all_layers_attention):
        layer_stats = get_attention_stats(layer_attn)
        layer_stats["layer"] = i
        stats.append(layer_stats)

    # Create figure if no axes provided
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(12, 8))
        created_fig = True
    else:
        created_fig = False

    layers = [s["layer"] for s in stats]

    # Plot on provided/created axes
    axes[0, 0].plot(layers, [s["mean"] for s in stats], marker="o")
    axes[0, 0].set_title("Mean Attention")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Mean")

    axes[0, 1].plot(layers, [s["max"] for s in stats], marker="o", color="red")
    axes[0, 1].set_title("Max Attention")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Max")

    axes[1, 0].plot(layers, [s["std"] for s in stats], marker="o", color="green")
    axes[1, 0].set_title("Std Attention")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Std")

    axes[1, 1].plot(
        layers, [s["diagonal_mean"] for s in stats], marker="o", color="purple"
    )
    axes[1, 1].set_title("Self-Attention (Diagonal)")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Diagonal Mean")

    # Add suptitle if provided or if we created the figure
    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold")
    elif created_fig:
        plt.suptitle(
            "Attention Statistics Across Layers", fontsize=14, fontweight="bold"
        )

    # Only save/show if we created the figure
    if created_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

    return axes


def plot_attention_difference(
    wt_attn: np.ndarray,
    mut_attn: np.ndarray,
    title: str,
    variant_pos: Optional[int] = None,
    variant_type: str = "substitution",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    window: Optional[int] = None,
) -> Axes:
    """
    Plot the DIFFERENCE between mutant and WT attention (mut - wt).
    For indels, aligns matrices by removing indel positions.

    Args:
        wt_attn: WT attention [seq_len, seq_len]
        mut_attn: Mutant attention [seq_len, seq_len]
        title: Plot title
        variant_pos: 1-indexed variant position
        variant_type: 'substitution', 'deletion', or 'insertion'
        ax: Optional axis
        figsize: Figure size
        window: Optional window around variant

    Returns:
        ax: The axis object

    Note:
        - If ax is None, a new figure is created and shown/saved if plt.show() or save_path is used.
        - If ax is provided, the plot is drawn on the given axis and not shown or saved
        automatically.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Handle indels by aligning matrices
    if wt_attn.shape != mut_attn.shape:
        pos = variant_pos - 1  # 0-indexed

        if variant_type == "deletion":
            # Remove deleted position from WT to match mutant
            print(f"Aligning deletion: removing position {variant_pos} from WT matrix")
            mask = np.ones(wt_attn.shape[0], dtype=bool)
            mask[pos] = False
            wt_attn_aligned = wt_attn[mask, :][:, mask]
            mut_attn_aligned = mut_attn

            # Adjust title and variant marker
            title = f"{title} (aligned: removed pos {variant_pos} from WT)"
            variant_pos_plot = None  # Can't mark deleted position

        elif variant_type == "insertion":
            # Remove inserted position(s) from mutant to match WT
            # Assuming single amino acid insertion after variant_pos
            print(
                f"Aligning insertion: removing position {variant_pos+1} from Mutant matrix"
            )
            mask = np.ones(mut_attn.shape[0], dtype=bool)
            mask[pos] = False  # Remove inserted position
            wt_attn_aligned = wt_attn
            mut_attn_aligned = mut_attn[mask, :][:, mask]

            title = f"{title} (aligned: removed inserted pos from Mut)"
            variant_pos_plot = pos  # Mark insertion point

        else:
            print(f"⚠️  Shape mismatch but variant_type='{variant_type}'")
            print(f"   Shapes: WT {wt_attn.shape} vs Mut {mut_attn.shape}")
            ax.text(
                0.5,
                0.5,
                f"Cannot align: check variant_type\nWT: {wt_attn.shape}\nMut: {mut_attn.shape}",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.axis("off")
            return ax

        wt_attn = wt_attn_aligned
        mut_attn = mut_attn_aligned

        print(f"Aligned shapes: WT {wt_attn.shape}, Mut {mut_attn.shape}")

    else:
        # Shapes already match (substitution)
        variant_pos_plot = variant_pos - 1 if variant_pos is not None else None

    # Compute difference
    diff = mut_attn - wt_attn

    # Apply windowing if requested
    if window is not None and variant_pos is not None:
        if variant_type == "deletion":
            # Window around deletion site (which is now absent)
            center = variant_pos - 1  # Would have been here
            start = max(0, center - window)
            end = min(diff.shape[0], center + window)
        else:
            center = variant_pos - 1
            start = max(0, center - window)
            end = min(diff.shape[0], center + window + 1)

        diff = diff[start:end, start:end]

        if variant_pos_plot is not None:
            variant_pos_plot = variant_pos_plot - start

        title = f"{title} (window: {start+1}-{end})"

    # Use diverging colormap centered at 0
    vmax = max(abs(diff.min()), abs(diff.max()))
    if vmax < 1e-10:  # Essentially no difference
        print(f"⚠️  Very small differences (max: {vmax:.2e}), likely numerical noise")

    im = ax.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xlabel("Attending to position", fontsize=12)
    ax.set_ylabel("Attending from position", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ Attention (Mut - WT)", fontsize=11)

    # Mark variant position (if applicable)
    if variant_pos_plot is not None:
        ax.axhline(
            variant_pos_plot,
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
        ax.axvline(
            variant_pos_plot,
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )

    return ax


def plot_variant_attention_profile(
    wt_attn: np.ndarray,
    mut_attn: np.ndarray,
    variant_pos: int,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Union[Axes, Tuple[Axes, Axes]]:
    """
    Plot attention TO and FROM variant position for WT vs Mutant.

    Args:
        wt_attn: WT attention [seq_len, seq_len]
        mut_attn: Mutant attention [seq_len, seq_len]
        variant_pos: 1-indexed variant position
        ax: Optional axis
        figsize: Figure size

    Returns:
        ax: The axis object
    """
    if ax is None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = ax if isinstance(ax, (list, np.ndarray)) else (ax, None)

    pos = variant_pos - 1  # 0-indexed
    positions = np.arange(wt_attn.shape[0])

    # Plot 1: Attention FROM variant
    if ax1 is not None:
        ax1.plot(positions, wt_attn[pos, :], "b-", label="WT", alpha=0.7, linewidth=2)
        ax1.plot(
            positions,
            mut_attn[pos, :],
            "r-",
            label="Mutant",
            alpha=0.7,
            linewidth=2,
        )
        ax1.axvline(pos, color="black", linestyle="--", alpha=0.5, label="Variant pos")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Attention weight")
        ax1.set_title(f"Attention FROM position {variant_pos}")
        ax1.legend()
        ax1.grid(alpha=0.3)

    # Plot 2: Attention TO variant
    if ax2 is not None:
        ax2.plot(positions, wt_attn[:, pos], "b-", label="WT", alpha=0.7, linewidth=2)
        ax2.plot(
            positions,
            mut_attn[:, pos],
            "r-",
            label="Mutant",
            alpha=0.7,
            linewidth=2,
        )
        ax2.axvline(pos, color="black", linestyle="--", alpha=0.5, label="Variant pos")
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Attention weight")
        ax2.set_title(f"Attention TO position {variant_pos}")
        ax2.legend()
        ax2.grid(alpha=0.3)

    return (ax1, ax2) if ax2 is not None else ax1


def compare_attention_layers_difference(
    wt_all_layers: np.ndarray,
    mut_all_layers: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot DIFFERENCE in attention statistics across layers (Mutant - WT).

    Args:
        wt_all_layers: WT attention [num_layers, seq_len, seq_len]
        mut_all_layers: Mutant attention [num_layers, seq_len, seq_len]
        title: Optional plot title
        save_path: Path to save figure
        ax: Optional axis (should be 2x2 array)

    Returns:
        axes: The 2x2 axes array
    """
    # Compute stats for both
    wt_stats = []
    mut_stats = []
    for i, (wt_layer, mut_layer) in enumerate(zip(wt_all_layers, mut_all_layers)):
        wt_stat = get_attention_stats(wt_layer)
        wt_stat["layer"] = i
        wt_stats.append(wt_stat)

        mut_stat = get_attention_stats(mut_layer)
        mut_stat["layer"] = i
        mut_stats.append(mut_stat)

    # Create figure if no axes provided
    if ax is None:
        _, axes = plt.subplots(2, 2, figsize=(12, 8))
        created_fig = True
    else:
        axes = ax
        created_fig = False

    layers = [s["layer"] for s in wt_stats]

    # Plot differences
    mean_diff = [m["mean"] - w["mean"] for w, m in zip(wt_stats, mut_stats)]
    max_diff = [m["max"] - w["max"] for w, m in zip(wt_stats, mut_stats)]
    std_diff = [m["std"] - w["std"] for w, m in zip(wt_stats, mut_stats)]
    diag_diff = [
        m["diagonal_mean"] - w["diagonal_mean"] for w, m in zip(wt_stats, mut_stats)
    ]

    axes[0, 0].plot(layers, mean_diff, marker="o", color="purple", linewidth=2)
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.3)
    axes[0, 0].set_title("Δ Mean Attention (Mut - WT)")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Δ Mean")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(layers, max_diff, marker="o", color="red", linewidth=2)
    axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.3)
    axes[0, 1].set_title("Δ Max Attention (Mut - WT)")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Δ Max")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(layers, std_diff, marker="o", color="green", linewidth=2)
    axes[1, 0].axhline(0, color="black", linestyle="--", alpha=0.3)
    axes[1, 0].set_title("Δ Std Attention (Mut - WT)")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Δ Std")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(layers, diag_diff, marker="o", color="orange", linewidth=2)
    axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.3)
    axes[1, 1].set_title("Δ Self-Attention (Mut - WT)")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Δ Diagonal")
    axes[1, 1].grid(alpha=0.3)

    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold")

    if created_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

    return axes


def compare_attention_layers_overlay(
    wt_all_layers: np.ndarray,
    mut_all_layers: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Overlay WT and Mutant layer statistics on same plots for direct comparison.

    Args:
        wt_all_layers: WT attention [num_layers, seq_len, seq_len]
        mut_all_layers: Mutant attention [num_layers, seq_len, seq_len]
        title: Optional plot title
        save_path: Path to save figure
        ax: Optional axis (should be 2x2 array)

    Returns:
        axes: The 2x2 axes array
    """
    # Compute stats
    wt_stats = []
    mut_stats = []
    for i, (wt_layer, mut_layer) in enumerate(zip(wt_all_layers, mut_all_layers)):
        wt_stat = get_attention_stats(wt_layer)
        wt_stat["layer"] = i
        wt_stats.append(wt_stat)

        mut_stat = get_attention_stats(mut_layer)
        mut_stat["layer"] = i
        mut_stats.append(mut_stat)

    # Create figure if no axes provided
    if ax is None:
        _, axes = plt.subplots(2, 2, figsize=(12, 8))
        created_fig = True
    else:
        axes = ax
        created_fig = False

    layers = [s["layer"] for s in wt_stats]

    # Plot overlays
    axes[0, 0].plot(
        layers,
        [s["mean"] for s in wt_stats],
        marker="o",
        color="blue",
        label="WT",
        linewidth=2,
        alpha=0.7,
    )
    axes[0, 0].plot(
        layers,
        [s["mean"] for s in mut_stats],
        marker="s",
        color="red",
        label="Mutant",
        linewidth=2,
        alpha=0.7,
    )
    axes[0, 0].set_title("Mean Attention")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Mean")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(
        layers,
        [s["max"] for s in wt_stats],
        marker="o",
        color="blue",
        label="WT",
        linewidth=2,
        alpha=0.7,
    )
    axes[0, 1].plot(
        layers,
        [s["max"] for s in mut_stats],
        marker="s",
        color="red",
        label="Mutant",
        linewidth=2,
        alpha=0.7,
    )
    axes[0, 1].set_title("Max Attention")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Max")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(
        layers,
        [s["std"] for s in wt_stats],
        marker="o",
        color="blue",
        label="WT",
        linewidth=2,
        alpha=0.7,
    )
    axes[1, 0].plot(
        layers,
        [s["std"] for s in mut_stats],
        marker="s",
        color="red",
        label="Mutant",
        linewidth=2,
        alpha=0.7,
    )
    axes[1, 0].set_title("Std Attention")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Std")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(
        layers,
        [s["diagonal_mean"] for s in wt_stats],
        marker="o",
        color="blue",
        label="WT",
        linewidth=2,
        alpha=0.7,
    )
    axes[1, 1].plot(
        layers,
        [s["diagonal_mean"] for s in mut_stats],
        marker="s",
        color="red",
        label="Mutant",
        linewidth=2,
        alpha=0.7,
    )
    axes[1, 1].set_title("Self-Attention (Diagonal)")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Diagonal Mean")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold")

    if created_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

    return axes


def compare_specific_heads(
    wt_attn_per_head: np.ndarray,
    mut_attn_per_head: np.ndarray,
    head_indices: list,
    layer_idx: int,
    variant_info: dict,
    window: Optional[int] = None,
) -> None:
    """
    Visualize specific heads that show interesting patterns.

    Args:
        wt_attn_per_head: [num_heads, seq_len, seq_len]
        mut_attn_per_head: [num_heads, seq_len, seq_len]
        head_indices: List of head indices to plot
        layer_idx: Which layer these heads are from
        variant_info: Variant dictionary
        window: Optional windowing

    Note:
        - Displays a matplotlib figure with heatmaps for each selected head
          (WT, Mutant, and Difference).
        - The plot is shown automatically via plt.show().
    """
    n_heads = len(head_indices)
    fig, axes = plt.subplots(n_heads, 3, figsize=(18, 6 * n_heads))

    if n_heads == 1:
        axes = axes.reshape(1, -1)

    for i, head_idx in enumerate(head_indices):
        wt = wt_attn_per_head[head_idx]
        mut = mut_attn_per_head[head_idx]

        # WT
        plot_attention_heatmap(
            wt,
            f"Layer {layer_idx}, Head {head_idx} - WT",
            variant_pos=variant_info["pos"],
            ax=axes[i, 0],
            window=window,
        )

        # Mutant
        plot_attention_heatmap(
            mut,
            f"Layer {layer_idx}, Head {head_idx} - Mutant",
            variant_pos=variant_info["pos"],
            ax=axes[i, 1],
            window=window,
        )

        # Difference
        plot_attention_difference(
            wt,
            mut,
            f"Layer {layer_idx}, Head {head_idx} - Δ",
            variant_pos=variant_info["pos"],
            variant_type="substitution",
            ax=axes[i, 2],
            window=window,
        )

    fig.suptitle(
        (
            f"{variant_info['gene']} "
            f"{variant_info['wt']}{variant_info['pos']}{variant_info['mut']}",
            "- Head-Level Analysis",
        ),
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
