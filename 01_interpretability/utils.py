"""
Utility functions for attention extraction and visualization.
Week 1: ESM2 protein foundation models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Union


def extract_attention(
    model,
    tokenizer,
    sequence: str,
    layer_idx: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract attention weights from ESM2 model for a protein sequence.

    Args:
        model: ESM2 model with output_attentions=True
        tokenizer: ESM2 tokenizer
        sequence: Protein sequence string (amino acids)
        layer_idx: Specific layer to extract (None = all layers)
        device: 'cpu', 'cuda', or 'mps'

    Returns:
        attention: numpy array
            - If layer_idx specified: [seq_len, seq_len]
            - If layer_idx=None: [num_layers, seq_len, seq_len]

    Example:
        >>> attention = extract_attention(model, tokenizer, "MVHLTPEEKS", layer_idx=-1)
        >>> attention.shape
        (12, 12)  # Last layer attention for 10 AA + 2 special tokens
    """
    # Tokenize sequence
    tokens = tokenizer(sequence, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**tokens)

    # Extract attention: tuple of (batch, heads, seq, seq) per layer
    attentions = outputs.attentions

    if layer_idx is not None:
        # Single layer: average across heads
        attn = attentions[layer_idx].squeeze(0).mean(dim=0).cpu().numpy()
        return attn
    else:
        # All layers: average across heads for each layer
        all_layers = []
        for layer_attn in attentions:
            attn = layer_attn.squeeze(0).mean(dim=0).cpu().numpy()
            all_layers.append(attn)
        return np.stack(all_layers, axis=0)


def aggregate_heads(attention_tensor: torch.Tensor, method: str = "mean") -> np.ndarray:
    """
    Aggregate attention across multiple heads.

    Args:
        attention_tensor: Shape [batch, heads, seq_len, seq_len] or [heads, seq_len, seq_len]
        method: 'mean', 'max', or 'min'

    Returns:
        aggregated: [seq_len, seq_len]
    """
    if attention_tensor.dim() == 4:
        attention_tensor = attention_tensor.squeeze(0)  # Remove batch dim

    if method == "mean":
        return attention_tensor.mean(dim=0).cpu().numpy()
    elif method == "max":
        return attention_tensor.max(dim=0)[0].cpu().numpy()
    elif method == "min":
        return attention_tensor.min(dim=0)[0].cpu().numpy()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def plot_attention_heatmap(
    attention: np.ndarray,
    title: str,
    variant_pos: Optional[int] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    window: Optional[int] = None,
) -> plt.Axes:
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
    """
    # Create figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

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
            variant_pos_plot, color="red", linestyle="--", linewidth=2, alpha=0.7
        )
        ax.legend(fontsize=10)

    if save_path and ax is None:  # Only save if we created the figure
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")

    return ax


def get_attention_stats(attention: np.ndarray) -> dict:
    """
    Compute summary statistics for attention matrix.

    Args:
        attention: [seq_len, seq_len] attention matrix

    Returns:
        stats: Dictionary with mean, max, std, diagonal_mean
    """
    return {
        "mean": attention.mean(),
        "max": attention.max(),
        "std": attention.std(),
        "diagonal_mean": attention.diagonal().mean(),  # Self-attention
    }


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
    """
    # Compute stats
    stats = []
    for i, layer_attn in enumerate(all_layers_attention):
        layer_stats = get_attention_stats(layer_attn)
        layer_stats["layer"] = i
        stats.append(layer_stats)

    # Create figure if no axes provided
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    window: Optional[int] = None,
) -> plt.Axes:
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
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

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
            variant_pos_plot, color="black", linestyle="--", linewidth=2, alpha=0.7
        )
        ax.axvline(
            variant_pos_plot, color="black", linestyle="--", linewidth=2, alpha=0.7
        )

    return ax


def analyze_variant_attention_changes(
    wt_attn: np.ndarray,
    mut_attn: np.ndarray,
    variant_pos: int,
    variant_type: str = "substitution",  # "substitution", "deletion", "insertion"
    window: int = 10,
) -> dict:
    """
    Analyze how attention changes around the variant position.

    Args:
        wt_attn: WT attention [seq_len, seq_len]
        mut_attn: Mutant attention [seq_len, seq_len]
        variant_pos: 1-indexed variant position
        variant_type: Type of variant (substitution, deletion, insertion)
        window: Residues around variant to analyze

    Returns:
        dict with analysis results
    """
    pos = variant_pos - 1  # 0-indexed

    # Check if shapes match
    if wt_attn.shape != mut_attn.shape:
        print(f"⚠️  Shape mismatch: WT {wt_attn.shape} vs Mut {mut_attn.shape}")
        print(f"   This is expected for indels. Computing regional statistics only.")

        # For indels, compare regions that exist in both
        if variant_type == "deletion":
            # Compare regions before deletion
            comparable_len = min(wt_attn.shape[0], mut_attn.shape[0])

            # Region before variant (unaffected)
            if pos > window:
                before_start = max(0, pos - window)
                before_wt = wt_attn[before_start:pos, before_start:pos].mean()
                before_mut = mut_attn[before_start:pos, before_start:pos].mean()
            else:
                before_wt = before_mut = np.nan

            # Region after variant (shifted by 1 in mutant)
            if pos + window < comparable_len:
                after_wt = wt_attn[
                    pos + 1 : pos + window + 1, pos + 1 : pos + window + 1
                ].mean()
                after_mut = mut_attn[
                    pos : pos + window, pos : pos + window
                ].mean()  # Shifted
            else:
                after_wt = after_mut = np.nan

            return {
                "variant_pos": variant_pos,
                "variant_type": variant_type,
                "shape_mismatch": True,
                "wt_shape": wt_attn.shape,
                "mut_shape": mut_attn.shape,
                "region_before_deletion_wt": before_wt,
                "region_before_deletion_mut": before_mut,
                "region_after_deletion_wt": after_wt,
                "region_after_deletion_mut": after_mut,
                "global_mean_wt": wt_attn.mean(),
                "global_mean_mut": mut_attn.mean(),
                "global_mean_change": mut_attn.mean() - wt_attn.mean(),
            }

        elif variant_type == "insertion":
            # Similar logic for insertions
            comparable_len = min(wt_attn.shape[0], mut_attn.shape[0])

            return {
                "variant_pos": variant_pos,
                "variant_type": variant_type,
                "shape_mismatch": True,
                "wt_shape": wt_attn.shape,
                "mut_shape": mut_attn.shape,
                "global_mean_wt": wt_attn.mean(),
                "global_mean_mut": mut_attn.mean(),
                "global_mean_change": mut_attn.mean() - wt_attn.mean(),
            }

    # For substitutions, shapes match - do full analysis
    start = max(0, pos - window)
    end = min(wt_attn.shape[0], pos + window + 1)

    # Attention FROM variant position
    attn_from_var_wt = wt_attn[pos, :]
    attn_from_var_mut = mut_attn[pos, :]

    # Attention TO variant position
    attn_to_var_wt = wt_attn[:, pos]
    attn_to_var_mut = mut_attn[:, pos]

    # Regional attention (within window)
    regional_wt = wt_attn[start:end, start:end].mean()
    regional_mut = mut_attn[start:end, start:end].mean()

    results = {
        "variant_pos": variant_pos,
        "variant_type": variant_type,
        "shape_mismatch": False,
        "attn_from_variant_change": attn_from_var_mut.sum() - attn_from_var_wt.sum(),
        "attn_to_variant_change": attn_to_var_mut.sum() - attn_to_var_wt.sum(),
        "regional_attention_change": regional_mut - regional_wt,
        "max_increase_pos": np.argmax(mut_attn - wt_attn),
        "max_decrease_pos": np.argmin(mut_attn - wt_attn),
        "mean_abs_change": np.abs(mut_attn - wt_attn).mean(),
    }

    return results


def plot_variant_attention_profile(
    wt_attn: np.ndarray,
    mut_attn: np.ndarray,
    variant_pos: int,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Axes:
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = ax if isinstance(ax, (list, np.ndarray)) else (ax, None)

    pos = variant_pos - 1  # 0-indexed
    positions = np.arange(wt_attn.shape[0])

    # Plot 1: Attention FROM variant
    if ax1 is not None:
        ax1.plot(positions, wt_attn[pos, :], "b-", label="WT", alpha=0.7, linewidth=2)
        ax1.plot(
            positions, mut_attn[pos, :], "r-", label="Mutant", alpha=0.7, linewidth=2
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
            positions, mut_attn[:, pos], "r-", label="Mutant", alpha=0.7, linewidth=2
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
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
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
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
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
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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


def analyze_layer_wise_changes(
    wt_all_layers: np.ndarray,
    mut_all_layers: np.ndarray,
) -> dict:
    """
    Quantify which layers show biggest attention changes.

    Args:
        wt_all_layers: WT attention [num_layers, seq_len, seq_len]
        mut_all_layers: Mutant attention [num_layers, seq_len, seq_len]

    Returns:
        dict with layer-wise analysis
    """
    layer_changes = []

    for i, (wt_layer, mut_layer) in enumerate(zip(wt_all_layers, mut_all_layers)):
        diff = mut_layer - wt_layer
        layer_changes.append(
            {
                "layer": i,
                "mean_abs_change": np.abs(diff).mean(),
                "max_change": diff.max(),
                "min_change": diff.min(),
                "std_change": diff.std(),
            }
        )

    # Find most changed layers
    sorted_by_change = sorted(
        layer_changes, key=lambda x: x["mean_abs_change"], reverse=True
    )

    return {
        "layer_changes": layer_changes,
        "most_changed_layers": [x["layer"] for x in sorted_by_change[:5]],
        "mean_change_across_layers": np.mean(
            [x["mean_abs_change"] for x in layer_changes]
        ),
    }


def extract_attention_per_head(
    model,
    tokenizer,
    sequence: str,
    layer_idx: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract attention for each head separately (no averaging).

    Args:
        model: ESM2 model with output_attentions=True
        tokenizer: ESM2 tokenizer
        sequence: Protein sequence
        layer_idx: Which layer to extract
        device: 'cpu', 'cuda', or 'mps'

    Returns:
        attention: [num_heads, seq_len, seq_len]
    """
    tokens = tokenizer(sequence, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)

    # Extract specific layer: [batch, heads, seq, seq]
    attn = outputs.attentions[layer_idx].squeeze(0).cpu().numpy()
    return attn


def find_most_different_heads(
    wt_attn_per_head: np.ndarray, mut_attn_per_head: np.ndarray, top_k: int = 5
) -> list:
    """
    Find which heads show biggest changes between WT and mutant.

    Args:
        wt_attn_per_head: [num_heads, seq_len, seq_len]
        mut_attn_per_head: [num_heads, seq_len, seq_len]
        top_k: Return top K most different heads

    Returns:
        List of (head_idx, mean_abs_diff) tuples
    """
    head_diffs = []
    for head_idx in range(wt_attn_per_head.shape[0]):
        diff = np.abs(mut_attn_per_head[head_idx] - wt_attn_per_head[head_idx]).mean()
        head_diffs.append((head_idx, diff))

    # Sort by difference
    head_diffs.sort(key=lambda x: x[1], reverse=True)
    return head_diffs[:top_k]


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
        f"{variant_info['gene']} {variant_info['wt']}{variant_info['pos']}{variant_info['mut']} - Head-Level Analysis",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
