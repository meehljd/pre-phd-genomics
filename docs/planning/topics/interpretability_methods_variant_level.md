# Interpretability Methods: Variant-Level Attribution for Rare Disease Diagnosis
**Pre-PhD Phase A/1 Integration**

---

## Problem Statement

Given a subject with multiple CDS variants in a gene, **which individual variants drive the diagnostic prediction**? Existing methods (SHAP, LIME) don't leverage population variant catalogs or catch ancestry confounding. We need clinical-grade interpretation that validates model predictions against known biology.

---

## Proposed Method Stack

### 1. **Ablation + Population Baseline** (Core, 15-20 hrs)
**Most clinically useful do this first.**

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

### 4. **Population Statistics** (4-6 hrs)
**Ground truth validation.**

**Method:**
- Case/control allele frequency difference (stratified by ancestry)
- Which variants are significantly enriched in diagnosed cohort?
- Chi-square test with ancestry correction

**Output:**
```
Variant rs123:
  Diagnosed cohort (n=1000): 8% allele freq
  Undiagnosed control (n=5000): 2% allele freq
  Fisher exact p-value: 0.001 (ancestry-adjusted)
  Interpretation: Real signal (not model artifact)
```

---

## Recommended Implementation Order

1. **Week 1-2:** Build population catalog (gnomAD variant encodings + ancestry stratification)
2. **Week 3:** Ablation + population comparison on toy examples (10 subjects, 3 genes)
3. **Week 4:** Saliency sanity checks on same cohort
4. **Week 5 (if time):** Integrated Gradients on high-interaction cases
5. **Week 6:** Validation against ACMG/ClinVar + case/control statistics

---

## Methods NOT Included (and Why)

| Method | Why skip (for now) |
|--------|-------------------|
| SHAP | Overkill for variant-level; doesn't leverage population data; O(2^n) permutations expensive |
| LIME | Local linear breaks with epistasis; unstable for rare variants |
| DeepLIFT | Redundant vs Integrated Gradients; more implementation overhead |
| TCAV | Requires pathway/phenotype annotations not yet available |
| Attribution Graphs | Defer to Year 1 PhD (post-model freeze) for deeper mechanistic analysis |

---

## Timeline & Scope for Phase A/1

**Total effort:** 30-40 hours (fits within Track A expansion)

**Deliverables:**
- âœ… Population variant catalog (encoded embeddings + metadata)
- âœ… Ablation inference pipeline (subject â†’ per-variant effects)
- âœ… Validation notebook: 20-30 subjects, ancestry-stratified comparison
- âœ… GitHub repo: reproducible saliency + Integrated Gradients code
- âœ… Clinical interpretation guide (for Paper 1 supplement)

**Integration into larger plan:**
- Track A Phase 2 (Dec 2025): Expand from generic SHAP/attention to variant-specific ablation
- Track B Phase 1 (Dec-Jan): Integrate population statistics + ancestry stratification
- Paper 1 validation (Jun-Jul 2027): Use all four methods for retrospective cohort

---

## Key Dependencies

- gnomAD VCF + ancestry labels (download ~Nov 2025)
- Mayo retrospective cohort (variant calls + phenotypes)
- Pre-computed reference CDS embeddings for genome (~2GB, one-time cost)
- GPU for batch variant encoding (~4 hrs for ~100k variants)

---

## Open Questions

1. How to handle multi-gene interactions? (Current approach: per-gene CDS only)
2. Should we weight population carriers by genetic distance to subject (ancestry PCA)?
3. For novel variants: fallback to saliency + conservation scores?
4. How to present results to clinicians (dashboard, PDF report, etc.)?

---

**Status:** Ready for Phase A/1 implementation  
**Next step:** Integrate into `pre-phd_intensive_study_plan.md` Track A Phase 2