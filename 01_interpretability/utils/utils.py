"""
Attention Extraction and Analysis Utilities for ESM2 Protein Models

This module provides a comprehensive set of functions for extracting and analyzing
attention weights from ESM2 protein foundation models. It is designed to support
interpretability studies, especially in the context of protein variant analysis.

Main utilities include:
    - Extraction of attention matrices from ESM2 models (per layer, per head)
    - Aggregation and statistical analysis of attention patterns
    - Specialized tools for analyzing the impact of protein variants (substitutions, insertions, deletions)

Plotting and visualization functions have been moved to plot_utils.py.

Requirements:
    - ESM2 model and tokenizer (must support output_attentions=True and HuggingFace-style tokenization)
    - torch, numpy

References:
    - ESM2: https://github.com/facebookresearch/esm
    - Attention mechanisms: https://arxiv.org/abs/1706.03762

Note: For plotting and visualization, use the functions in plot_utils.py.
"""


from typing import Optional
import torch
import numpy as np


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
        model: ESM2 model (must have output_attentions=True and a .attentions output)
        tokenizer: ESM2 tokenizer (must support __call__ with return_tensors="pt")
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


def aggregate_heads(
    attention_tensor: torch.Tensor, method: str = "mean"
) -> np.ndarray:
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
        print("   This is expected for indels. Computing regional statistics only.")

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
        "attn_from_variant_change": attn_from_var_mut.sum()
        - attn_from_var_wt.sum(),
        "attn_to_variant_change": attn_to_var_mut.sum() - attn_to_var_wt.sum(),
        "regional_attention_change": regional_mut - regional_wt,
        "max_increase_pos": np.argmax(mut_attn - wt_attn),
        "max_decrease_pos": np.argmin(mut_attn - wt_attn),
        "mean_abs_change": np.abs(mut_attn - wt_attn).mean(),
    }

    return results


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

    for i, (wt_layer, mut_layer) in enumerate(
        zip(wt_all_layers, mut_all_layers)
    ):
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
        diff = np.abs(
            mut_attn_per_head[head_idx] - wt_attn_per_head[head_idx]
        ).mean()
        head_diffs.append((head_idx, diff))

    # Sort by difference
    head_diffs.sort(key=lambda x: x[1], reverse=True)
    return head_diffs[:top_k]

