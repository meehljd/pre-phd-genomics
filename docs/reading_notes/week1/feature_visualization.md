# Feature Visualization - Reading Notes
**Source:** https://distill.pub/2017/feature-visualization/  
**Author:** Olah et al. 2017  
**Date Read:** Oct 30, 2024  
**Time:** 30 min

---

## Core Question for This Project:
**How do I explain ESM2 attention patterns to clinicians who don't know ML?**

---

## Key Takeaways (3-5 bullets)

1.  There is feature attribution (e.g., which pixels in an image are important for a classification decision) and there is feature visualization (e.g., optimizing inputs to maximally activate neurons).

2. Hierarchical feature representations can be visualized by optimizing inputs to activate specific neurons or layers.  Layers start with edges and textures, and progress to complex objects.

3.  Regularization is critical for interpretability, including natural image priors, frequency penalties, and transformation robustness.

4. Neurons interact in complex ways, and visualizing combinations of neurons can reveal more complex features.

5. Preconditioned feature visualization (starting from natural images) can yield more interpretable results.  Can use L2, L inf, and decorrelated space (Fourier basis) constraints.

---

## Principles for Clinical Communication

### What makes visualizations interpretable?
<!-- Focus on: simplicity, biological relevance, avoiding jargon -->
- Use clear, simple visuals (e.g., heatmaps) that highlight key areas of interest.
- Relate findings to known biological concepts (e.g., protein domains, functional sites).
- Avoid technical jargon; use analogies and straightforward language.

### Common pitfalls to avoid:
<!-- What confuses non-ML audiences? -->
- Overloading visuals with too much information (e.g., all attention heads at once).
- Failing to connect results to biological significance (e.g., "this is just a heatmap").
- Using ML-specific terms without explanation (e.g., "self-attention", "transformer").



---

## Application to Attention Analysis

### How to visualize attention for clinicians:
<!-- Heatmaps? Highlighting? Overlays on protein structure? -->
- Use heatmaps to show attention across protein sequences.
- Highlight key regions of interest (e.g., active sites, binding domains).
- Overlay attention maps on 3D protein structures for spatial context.

### What language to use:
<!-- Replace "attention weights" with...? -->
- Instead of "attention weights" → "importance scores"
- Instead of "transformer layers" → "model layers"
- Instead of "softmax" → "probability distribution"

### What to show vs. hide:
**Show:**
- Key regions of interest (e.g., active sites, binding domains)

**Hide/Simplify:**
- Technical details about the model architecture
- Specifics of the attention mechanism (e.g., "self-attention", "multi-head")

---

## Concrete Ideas for Week 1 Analysis

### For pathogenic variant reports:
<!-- What visualizations will be most convincing? -->
- Heatmaps showing attention on the variant position and surrounding residues.
- Overlay attention maps on 3D structures highlighting the variant site.
- Annotated sequence logos indicating important residues.


### For comparing wild-type vs variant:
<!-- Side-by-side? Difference maps? Overlay? -->

- Side-by-side comparisons of attention maps for wild-type vs variant sequences.
- Difference maps highlighting changes in attention patterns.
- Overlay attention maps on 3D structures to show spatial differences.

---

## One Thing to Try This Week

<!-- One specific visualization technique from the paper to implement -->
- Implement heatmaps to show attention across protein sequences.

---

## Questions/Confusion

<!-- Anything unclear that needs follow-up? -->