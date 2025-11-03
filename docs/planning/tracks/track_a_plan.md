## TRACK A: INTERPRETABILITY ON YOUR MODELS (Nov-Dec 2025 - 2-3 weeks total)

### Phase 1: Attention Maps on ESM2 & Your Encoder-Decoder (Nov 2025 - 1.5 weeks)

**Goal:** Extract + visualize attention patterns; understand what your bottleneck learns

**Note on Genomic Foundation Models:** Original plan used Evo2, but due to usability issues, we're using ESM2 (protein-level) for interpretability learning in Track A. Genomic model selection (Enformer, Nucleotide Transformer, or alternatives) will be decided in Track B Phase 2 based on comparative evaluation.

**Minimal reading (you know transformers):**
1. **Avsec et al. 2021 (Enformer) Sections 3-4 only** - 1-2 hours (refresher on bio-focused attention)
2. **Olah et al. 2017 "Feature Visualization" blog** - 1 hour (making attention interpretable to non-ML)

**Hands-On (Core work):**

```python
# Project: Attention Visualization on Protein Models (ESM2)

# 1. ESM2 attention extraction (3-4 hours)
- Load ESM2 model from HuggingFace
- For 10 variants (5 pathogenic, 5 benign): protein sequence window
- Extract attention matrices from all layers
- Aggregate across heads: attention_scores[layer][position_i][position_j]
- Visualize: Heatmap showing which positions attend to functional domains?
- Validate: Compare to known active sites, binding domains, structural features

# 2. Your dual-llama encoder-decoder (4-5 hours)
- Encoder: Sequence attention patterns
- Decoder: Cross-attention to bottleneck embedding
- Key analysis: What does the bottleneck compress?
  - Does it capture protein folding signals?
  - Does it preserve phenotype-relevant information?
  - Can you reconstruct biological signal from bottleneck alone?

# 3. Comparative analysis (2 hours)
- Side-by-side: ESM2 vs your model attention on same variants
- Interpretation: Where do they agree/disagree?
- Insight: Why does your bottleneck work?
```

**Output:**
- Jupyter notebook: "Attention Analysis in Protein Foundation Models (ESM2)"
- 10-12 publication-quality heatmaps
- Document: "Bottleneck Architecture Analysis" (500 words)

---

### Phase 2: Variant-Level Ablation + Population Baseline (Dec 2025 - 1.5 weeks)

**Goal:** Build clinical-grade variant interpretation pipeline

**Reading:** None (you have this from Track B prep)

**Hands-On (20-25 hours):**

### 1. **Ablation + Population Baseline** (Core, 15-20 hrs)
**Most clinically usefulâ€”do this first.**

**Setup:**
- Pre-compute population variant effect catalog:
  - Encode reference CDS (no variants) â†’ baseline embedding `e_ref`
  - For each variant in gnomAD/dbSNP: encode CDS + variant individually â†’ `e_vi_pop`
  - Store population shifts: `Î”e_pop_i = e_vi_pop - e_ref`
  - Stratify by ancestry (EUR, AFR, EAS, SAS)

**Subject-level inference:**
- Subject has CDS variants {v1, v2, ..., vn}
- Encode full CDS + all variants â†’ `e_full`
- For each variant vi:
  - Encode CDS + all variants except vi â†’ `e_-vi`
  - Subject ablation effect: `Î”e_subj_i = e_full - e_-vi`
  - Compare: `Î”e_subj_i` vs population distribution `Î”e_pop_i`

**Output per variant:**
```
Variant: rs123 (p.Arg123His)
  Subject embedding shift: -0.8Ïƒ
  EUR carriers (n=450): -0.75Ïƒ Â± 0.2
  AFR carriers (n=12): -0.5Ïƒ Â± 0.4
  Interpretation: Effect matches EUR population; likely real
  
Variant: rs456 (p.Leu456Val) [NOVEL]
  Subject embedding shift: +0.3Ïƒ
  Population: No carriers in gnomAD
  Interpretation: Unique; no population baseline
```

**Advantages:**
- Deterministic (not stochastic permutations)
- Population-aware (catches ancestry confounding)
- Computationally cheap (O(n) encode steps + precomputed catalog lookup)
- Clinically interpretable (compare to known carriers)

---

### 2. **Integrated Gradients** (Optional enhancement, 6-8 hrs)
**If variants interact non-additively.**

**Setup:**
- Path integral from reference CDS â†’ subject CDS
- Interpolate: `seq(Î±) = ref + Î± Ã— (subject - ref)` for Î± âˆˆ [0,1]
- Integrate gradients: `attr_i = âˆ« âˆ‚prediction/âˆ‚seq(Î±) dÎ±`
- Attributes contribution of each position to final prediction

**When to use:**
- If ablation shows large combined effect but small individual effects â†’ epistasis signal
- Validate: Do variants in high-LD regions have correlated Integrated Gradients?

**Output:**
```
Variants v1, v2 both show weak individual effects (-0.1Ïƒ each)
But combined: -0.5Ïƒ (non-additive)
Integrated Gradients path: Shows interaction at embedding layer
Interpretation: v1 and v2 together enhance protein disruption
```

---

### 3. **Saliency Sanity Check** (4-6 hrs)
**Quick validation of ablation results.**

**Method:**
- Single backward pass: `âˆ‚prediction/âˆ‚input`
- Identify which CDS positions have high gradients
- Do those positions cluster around variant sites?

**Pass/fail criteria:**
- If variant at position i has high saliency â†’ âœ“ consistent
- If variant at position i has near-zero saliency â†’ âš ï¸ flag (possible spurious)

---

Key tasks:
1. Build population variant catalog (gnomAD + ancestry stratification) - 8-10 hrs
2. Implement ablation inference pipeline - 6-8 hrs
3. Saliency sanity checks - 4-6 hrs
4. Optional: Integrated Gradients - 2-4 hrs

**Output:**
- Ablation inference notebook
- Population variant catalog (structure + metadata)
- Clinical interpretation guide template
- GitHub code (reproducible)

---

### Track A Summary (By Dec 31, 2025):
- ✅ Understand what ESM2 and your model learn
- ✅ Can generate clinical explanations for variants
- ✅ Ready to integrate into Aim 1 methodology
- **Time investment: ~20-25 hours**
