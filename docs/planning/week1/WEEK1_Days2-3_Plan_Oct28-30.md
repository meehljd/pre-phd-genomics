# Week 1 Days 2-3: Oct 28-30, 2025
## Track A Phase 1: Reading + Utility Infrastructure

---

## TUESDAY OCT 28 (1.5 hours) - Reading + Analysis Planning

### Hour 1: Enformer Attention Paper (1 hour)

**Reading: Avsec et al. 2021 - "Effective gene expression prediction from sequence by integrating long-range interactions"**
- **Paper:** Nature (2021), Sections 3-4 only
- **Focus areas:**
  - How they visualize attention for biological signals
  - Interpretation of attention heatmaps for regulatory elements
  - Connection between attention patterns and functional genomic regions

**Reading Notes Template** (`docs/reading_notes/enformer_attention.md`):

```markdown
# Enformer Attention Analysis - Reading Notes
## Avsec et al. 2021, Sections 3-4

### Key Questions to Answer:

1. **How do they visualize attention across layers?**
   - [Your notes here]

2. **What biological patterns emerge from attention?**
   - TFBS (transcription factor binding sites)?
   - Regulatory elements (promoters, enhancers)?
   - Conservation patterns?
   - [Your observations]

3. **Why is attention useful for rare disease variants?**
   - [Your analysis]

### Visualization Techniques:
- [List techniques they use]

### Biological Insights:
- [Key findings relevant to your work]

### Application to Your Project:
- How can you apply these visualization approaches to ESM2?
- What biological signals should you look for in protein attention?
```

---

### 30 min: Analysis Plan Document

**Create: `docs/week1_analysis_plan.md`**

```markdown
# Week 1 Analysis Plan: ESM2 Attention Patterns

## Expected Biological Signals for ESM2

### Protein Structure Features:
1. **Secondary structure boundaries**
   - Alpha helices
   - Beta sheets
   - Loop regions

2. **Functional domains**
   - Active sites
   - Binding domains
   - Post-translational modification sites

3. **Conservation patterns**
   - Highly conserved residues (catalytic, structural)
   - Variable regions (surface, linkers)

### Variant-Specific Expectations:

**Pathogenic variants (e.g., BRCA1 R1699W):**
- High attention to variant position?
- Disrupted attention patterns compared to wild-type?
- Attention focused on functional domains affected by variant?

**Benign variants:**
- Attention patterns similar to wild-type?
- Located in low-attention (less functionally important) regions?

## Hypotheses to Test:

### H1: Layer Progression
- **Early layers:** Local sequence patterns (amino acid neighbors)
- **Middle layers:** Secondary structure elements
- **Late layers:** Global protein structure, functional domains

### H2: Pathogenic vs Benign
- Pathogenic variants disrupt normal attention patterns more than benign
- Pathogenic variants located in high-attention regions

### H3: Functional Relevance
- Positions receiving highest attention correspond to:
  - Known functional residues (from UniProt annotations)
  - Conserved positions (from multiple sequence alignments)
  - Disease-associated sites (from ClinVar)

## Validation Strategy:

1. **Domain annotations:** Compare attention peaks to UniProt domain boundaries
2. **Conservation scores:** Correlate attention weights with PhyloP/PhastCons
3. **Known pathogenic sites:** Check if known disease mutations have high attention

## Next Steps After Week 1:
- Test hypotheses with 10-variant systematic analysis (Week 2)
- Compare ESM2 patterns to DNABERT2 (if time permits)
- Begin SHAP analysis for mechanistic explanations
```

**Output by end of Tuesday:**
- ✅ `docs/reading_notes/enformer_attention.md` completed
- ✅ `docs/week1_analysis_plan.md` completed
- ✅ Clear hypotheses about what to expect from ESM2 attention

---

## WEDNESDAY OCT 30 (1.5 hours) - Utility Functions + Multi-Variant Prep

**Note:** Since you completed Monday's notebook work ahead of schedule, Wednesday focuses on building reusable infrastructure for systematic analysis.

---

### 30 min: Feature Visualization Reading

**Reading: Olah et al. 2017 - "Feature Visualization"**
- **Source:** https://distill.pub/2017/feature-visualization/
- **Focus:** Making ML interpretability accessible to non-ML audiences (clinicians)

**Reading Notes** (`docs/reading_notes/feature_visualization.md`):

```markdown
# Feature Visualization - Reading Notes
## Olah et al. 2017

### Key Principles for Clinical Communication:

1. **What makes visualizations interpretable?**
   - [Your notes]

2. **How to explain attention to clinicians?**
   - Avoid ML jargon
   - Connect to biological function
   - [Your strategies]

3. **Visualization best practices:**
   - Color schemes that work
   - Annotations that help
   - [Your takeaways]

### Application to Attention Heatmaps:

**For clinicians, emphasize:**
- "This shows which protein regions interact during folding"
- "Red areas indicate the model focuses here for predictions"
- [Your clinical-friendly explanations]

**Avoid:**
- "Attention weights in transformer layers"
- "Head aggregation across multi-head self-attention"
- [ML jargon to skip]
```

---

### 1 hour: Build Utility Module + Variant Dataset

#### Part 1: Create `01_interpretability/utils.py` (40 min)

```python
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


def aggregate_heads(
    attention_tensor: torch.Tensor,
    method: str = "mean"
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
    all_layers_attention: np.ndarray,
    save_path: Optional[str] = None
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
    
    axes[1, 1].plot(layers, [s["diagonal_mean"] for s in stats], marker="o", color="purple")
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
```

#### Part 2: Create `01_interpretability/variant_dataset.py` (20 min)

```python
"""
Variant dataset for systematic attention analysis.
Week 1: Define 10 variants (5 pathogenic, 5 benign)
"""

# Pathogenic variants from ClinVar
PATHOGENIC_VARIANTS = [
    {
        "gene": "BRCA1",
        "uniprot": "P38398",
        "pos": 1699,  # 1-indexed
        "wt": "R",
        "mut": "W",
        "disease": "Breast/ovarian cancer",
        "clinvar_id": "43082",
    },
    {
        "gene": "TP53",
        "uniprot": "P04637",
        "pos": 175,
        "wt": "R",
        "mut": "H",
        "disease": "Li-Fraumeni syndrome",
        "clinvar_id": "12345",  # TODO: Find actual ClinVar ID
    },
    {
        "gene": "CFTR",
        "uniprot": "P13569",
        "pos": 508,
        "wt": "F",
        "mut": "del",
        "disease": "Cystic fibrosis",
        "clinvar_id": "35",
    },
    # TODO: Add 2 more pathogenic variants
    # Suggestions: BRCA2, MLH1, ATM, or other cancer/rare disease genes
]

# Benign variants from gnomAD (common polymorphisms)
BENIGN_VARIANTS = [
    # TODO: Find 5 benign variants with high population frequency (>1%)
    # These should be common SNPs that don't cause disease
    # Example format:
    # {
    #     "gene": "GENE_NAME",
    #     "uniprot": "P12345",
    #     "pos": 100,
    #     "wt": "A",
    #     "mut": "V",
    #     "gnomad_freq": 0.15,  # 15% population frequency
    # }
]


def get_variant_sequence(variant_info: dict, window: int = 100) -> str:
    """
    Get protein sequence for variant with surrounding context.
    
    Args:
        variant_info: Dictionary with gene, uniprot, pos, wt, mut
        window: Residues on each side of variant
    
    Returns:
        sequence: Protein sequence string
    
    Note: For now, returns placeholder. Will integrate with UniProt in Week 2.
    """
    # Placeholder for now
    raise NotImplementedError("UniProt integration coming in Week 2")


def validate_variant_dataset() -> None:
    """
    Check that all variants have required fields.
    Run this before starting systematic analysis.
    """
    required_fields = ["gene", "uniprot", "pos", "wt", "mut"]
    
    print("=== Validating Pathogenic Variants ===")
    for i, var in enumerate(PATHOGENIC_VARIANTS, 1):
        missing = [f for f in required_fields if f not in var]
        if missing:
            print(f"❌ Variant {i}: Missing fields: {missing}")
        else:
            print(f"✓ Variant {i}: {var['gene']} {var['wt']}{var['pos']}{var['mut']}")
    
    print("\n=== Validating Benign Variants ===")
    if not BENIGN_VARIANTS:
        print("⚠️  No benign variants defined yet - add 5 for Week 2")
    else:
        for i, var in enumerate(BENIGN_VARIANTS, 1):
            missing = [f for f in required_fields if f not in var]
            if missing:
                print(f"❌ Variant {i}: Missing fields: {missing}")
            else:
                print(f"✓ Variant {i}: {var['gene']} {var['wt']}{var['pos']}{var['mut']}")


if __name__ == "__main__":
    validate_variant_dataset()
```

---

### Quick Test: Verify Utils Work (15 min at end)

**In Jupyter notebook or Python script:**

```python
# Test utils.py functions
import sys
sys.path.append('01_interpretability')

from utils import extract_attention, plot_attention_heatmap, get_attention_stats
from transformers import EsmModel, EsmTokenizer

# Load model
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", output_attentions=True)
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.eval()

# Test extraction
test_seq = "MVHLTPEEKSAVTALWGKVN"
attention = extract_attention(model, tokenizer, test_seq, layer_idx=-1)
print(f"✓ Extracted attention shape: {attention.shape}")

# Test stats
stats = get_attention_stats(attention)
print(f"✓ Attention stats: mean={stats['mean']:.4f}, max={stats['max']:.4f}")

# Test plot (should create visualization)
plot_attention_heatmap(
    attention, 
    "Test Sequence Attention",
    save_path="outputs/test_utils.png"
)
print("✓ All utils working correctly")
```

---

## OUTPUTS BY END OF WEDNESDAY

### Documentation:
- ✅ `docs/reading_notes/enformer_attention.md`
- ✅ `docs/reading_notes/feature_visualization.md`
- ✅ `docs/week1_analysis_plan.md`

### Code:
- ✅ `01_interpretability/utils.py` (6 functions implemented)
- ✅ `01_interpretability/variant_dataset.py` (3 pathogenic variants defined)
- ✅ Quick test showing utils work

### Understanding:
- ✅ Know how Enformer visualizes attention for biology
- ✅ Know how to explain attention to clinicians
- ✅ Have reusable infrastructure for systematic analysis

---

## READY FOR THURSDAY

With utilities built, Thursday you can:
1. Systematically analyze 10 variants (loop over variant list)
2. Generate comparative visualizations
3. Document attention pattern differences between pathogenic/benign

**Time saved:** By building reusable utils, Thursday's analysis will be much faster (2-3 hours instead of 5-6 hours).

---

## OPTIONAL: If Time Remains

**Research benign variants for dataset:**
- Search gnomAD browser for common SNPs (>1% frequency)
- Pick 5 variants in genes with protein structures available
- Add to `variant_dataset.py`

This sets you up for a productive Thursday-Friday systematic analysis sprint.
