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
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
) -> None:
    """
    Plot attention heatmap with optional variant position marker.

    Args:
        attention: [seq_len, seq_len] attention matrix
        title: Plot title
        variant_pos: Position of variant (0-indexed, will draw red crosshairs)
        save_path: Path to save figure (None = don't save)
        figsize: Figure size tuple
        cmap: Colormap name

    Example:
        >>> plot_attention_heatmap(
        ...     attention,
        ...     "BRCA1 R1699W",
        ...     variant_pos=99,
        ...     save_path="outputs/brca1.png"
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attention, cmap=cmap, aspect="auto")
    ax.set_xlabel("Attending to position", fontsize=12)
    ax.set_ylabel("Attending from position", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention weight", fontsize=11)

    # Mark variant position if provided
    if variant_pos is not None:
        ax.axhline(
            variant_pos,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Variant position",
        )
        ax.axvline(variant_pos, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")

    plt.show()


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
    all_layers_attention: np.ndarray, save_path: Optional[str] = None
) -> None:
    """
    Plot attention statistics across all layers.

    Args:
        all_layers_attention: [num_layers, seq_len, seq_len]
        save_path: Path to save figure
    """
    stats = []
    for i, layer_attn in enumerate(all_layers_attention):
        layer_stats = get_attention_stats(layer_attn)
        layer_stats["layer"] = i
        stats.append(layer_stats)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    layers = [s["layer"] for s in stats]

    axes[0, 0].plot(layers, [s["mean"] for s in stats], marker="o")
    axes[0, 0].set_title("Mean Attention")
    axes[0, 0].set_xlabel("Layer")

    axes[0, 1].plot(layers, [s["max"] for s in stats], marker="o", color="red")
    axes[0, 1].set_title("Max Attention")
    axes[0, 1].set_xlabel("Layer")

    axes[1, 0].plot(layers, [s["std"] for s in stats], marker="o", color="green")
    axes[1, 0].set_title("Std Attention")
    axes[1, 0].set_xlabel("Layer")

    axes[1, 1].plot(
        layers, [s["diagonal_mean"] for s in stats], marker="o", color="purple"
    )
    axes[1, 1].set_title("Self-Attention (Diagonal)")
    axes[1, 1].set_xlabel("Layer")

    plt.suptitle("Attention Statistics Across Layers", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")

    plt.show()


# TODO for Week 2:
def get_variant_window(uniprot_id: str, variant_pos: int, window: int = 100) -> str:
    """
    Fetch sequence window around variant position from UniProt.

    Args:
        uniprot_id: UniProt accession (e.g., "P38398")
        variant_pos: 1-indexed position of variant
        window: Number of residues on each side

    Returns:
        sequence: Protein sequence window

    Note: Requires UniProt API integration (implement in Week 2)
    """
    raise NotImplementedError("UniProt API integration coming in Week 2")
