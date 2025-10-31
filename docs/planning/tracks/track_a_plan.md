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

### Phase 2: SHAP + LIME + Counterfactual (Dec 2025 - 1.5 weeks)

**Goal:** Advanced explanations on your variant ranking system

**Reading (targeted, quick):**
1. **Lundberg & Lee 2017 SHAP** - 1.5 hours (you know gradients-based methods)
2. **Goyal et al. 2019 Counterfactual** - 1 hour (quick read on what-if scenarios)

**Hands-On (Core work):**

```python
# Project: Variant Ranking Interpretability

# 1. SHAP on pathogenicity prediction (3 hours)
- Model: Your encoder-decoder predicting pathogenic/benign
- Dataset: 100 variants (50 pathogenic ClinVar, 50 benign gnomAD)
- SHAP analysis per variant:
  - Which amino acid positions drive pathogenicity score?
  - Do top-ranked positions align with known functional domains?
  - SHAP visualization: force plots + bar plots
- Validation: Compare to known pathogenic regions in literature

# 2. LIME local explanations (2-3 hours)
- For 10 diverse variants (5 pathogenic, 5 benign)
- Generate LIME local surrogate models
- Output: "For this specific variant, these features matter most"
- Test with clinician (ask Eric Klee lab): Is this interpretable?

# 3. Counterfactual variants (3-4 hours)
- Pathogenic example: "BRCA1 p.R1699W"
- Query: "What single amino acid subs flip this to benign?"
- Generate top 20 counterfactuals ranked by confidence
- Validate: Are these realistic? What do they teach?
  - E.g., "R→K keeps charge, more likely benign" = mechanistic insight
```

**Output:**
- Notebook: "SHAP Analysis of Variant Pathogenicity"
- Notebook: "Counterfactual Variant Generation"
- Comparison document: SHAP vs LIME vs Attention (700 words)

---

### Track A Summary (By Dec 31, 2025):
- ✅ Understand what ESM2 and your model learn
- ✅ Can generate clinical explanations for variants
- ✅ Ready to integrate into Aim 1 methodology
- **Time investment: ~20-25 hours**
