# Debiasing Approaches: Handling Covariates in Rare Disease Models
**Pre-PhD Phase B Integration**

---

## Problem Statement

Models trained on genomic data inherit biases from:
- **Ancestry**: Training data skewed toward EUR, underrepresents AFR/EAS/SAS
- **Sex/X-linked variants**: Different biology for males vs females
- **Genetic structure confounding**: Variants linked to ancestry via LD, not causation

Goal: **Train fair models that generalize across ancestry/sex strata.**

---

## Debiasing Method Comparison

| Method | Approach | Cost | Best For | Trade-offs |
|--------|----------|------|----------|-----------|
| **Stratified Sampling** | Sample equal # per ancestry decile during training | 3-4 hrs | Baseline, prevents imbalance | Requires sufficient rare ancestry samples |
| **Subspace Removal** | Project out ancestry-correlated embedding dimensions | 15-20 hrs | Interpretable, linear bias | Assumes linear ancestry encoding |
| **INLP** | Iterative null-space projection (catches non-linear) | 20-30 hrs | Nonlinear ancestry signals | Slower; risk of over-removal |
| **Fairness Constraints** | Add equalized odds / demographic parity to loss | 15-20 hrs | Clinical fairness requirements | Multiple hyperparameters |
| **CORAL** | Align covariance across ancestry groups | 10-15 hrs | Lighter than INLP | Only matches 2nd-order stats |
| **Group DRO** | Minimize worst-case loss across ancestry groups | 20-25 hrs | Hard fairness guarantees | Can hurt overall accuracy |

---

## Recommended Stack for Phase B (25-35 hrs)

### **Tier 1: Core (do first)**

1. **Genetic Ancestry via PCA** (10-15 hrs)
   - Compute ancestry PCAs from training cohort genotypes (relatedness-pruned SNPs)
   - Use continuous PC1, PC2, PC3... (not discrete self-reported categories)
   - Bin by deciles for stratified sampling
   
   ```
   Stratified sampling:
     - Divide training data by PC1 decile — PC2 decile ~100 ancestry cells
     - Sample equal # subjects per cell per batch
     - Benefits: granular, captures admixture, objective
   ```

2. **Subspace Removal** (12-18 hrs)
   - Train regression head: `ancestry_PCAs = f_ancestry(encoder_output)`
   - Project out ancestry-correlated dimensions: `e_debiased = e - (W^T W) @ e`
   - Validate: MI(ancestry; e_debiased) should drop near-zero
   - Visualize: which embedding dimensions encode ancestry?

3. **Sex as Categorical Covariate** (2-3 hrs)
   - Do NOT subspace-remove sex (keep biological signal)
   - Instead: condition on sex during training/inference
   - X-chromosome handling:
     ```
     Males (XY): code X variants as 0/1 (hemizygous)
     Females (XX): code X variants as 0/1/2 (diploid)
     ```
   - Measure sex-specific variant effects in ablation output

### **Tier 2: Enhancement (if time)**

4. **INLP Refinement** (12-18 hrs)
   - If subspace removal doesn't fully decorrelate ancestry â†’ apply 1-2 INLP iterations
   - Iteratively project out non-linear ancestry signals
   - Validate: check that non-ancestry features still recover phenotype signal

5. **Fairness Constraints** (10-15 hrs)
   - Add to loss: `equalized_odds_loss = |TPR[EUR] - TPR[AFR]| + |FPR[EUR] - FPR[AFR]|`
   - Or demographic parity: `|P(pred+|EUR) - P(pred+|AFR)|`
   - Report: validate that fairness constraints are satisfied in validation

---

## Covariates: What to Include/Exclude

### **Include (genetic structure)**

| Covariate | Include? | How |
|-----------|----------|-----|
| **Ancestry (genetic)** | Yes | PCA debiasing + stratified sampling |
| **Sex** | Yes (conditional, not removed) | Categorical covariate; X-chromosome separate |
| **X-linked variants** | Yes | Code hemizygous for males; validate effect sizes differ |

### **Exclude (not causal for diagnosis)**

| Covariate | Include? | Why |
|-----------|----------|-----|
| **Age** | No | Diagnosis doesn't depend on age; only report age distribution |
| **BMI** | No | High missingness; consequence not cause of rare disease |
| **Smoking** | No | Independent of genetic diagnosis; adds deployment burden |
| **SES/environment** | No | Usually unavailable; high missingness; not causal |

**Exception:** If modeling age-of-onset (e.g., Huntington disease CAG repeats), include age. But for diagnostic classification: no.

---

## Validation Framework

**Report for each ancestry Ã— sex combination:**

```
Paper 1 Supplementary Table: Fairness Matrix

                EUR         AFR         EAS         SAS
           Male Female  Male Female  Male Female  Male Female
Accuracy   92%   91%    89%   88%    90%   89%    87%   86%
Sensitivity 85%   84%    82%   81%    84%   83%    80%   82%
Specificity 95%   96%    93%   94%    95%   94%    92%   93%

Pass criterion: All cells within 3-5% of best-performing strata
```

**Per-variant validation:**
```
Check: Does variant importance (from ablation) stay consistent across ancestry?

Variant rs123:
  EUR carriers (n=450): effect -0.8Ïƒ Â± 0.2
  AFR carriers (n=80):  effect -0.75Ïƒ Â± 0.3
  EAS carriers (n=120): effect -0.82Ïƒ Â± 0.25
  
  â†’ Consistent across ancestry? âœ“ Yes (within overlap of error bars)
     If no â†’ flag as potentially ancestry-confounded
```

---

## Implementation Timeline (Phase B Phase 1, Dec 2025)

**Week 1: Ancestry Setup**
- Pull Mayo retrospective cohort genotypes
- Compute ancestry PCAs (relatedness-pruned SNPs)
- Bin by deciles, wire into data loader

**Week 2: Subspace Removal + Visualization**
- Train regression head: ancestry_PCAs â† encoder_output
- Implement projection: `e_debiased = e - (W^T W) @ e`
- Plot: which embedding dimensions correlate with ancestry?
- Measure MI(ancestry; e_debiased)

**Week 3: Sex Handling + Validation**
- Code X-chromosome hemizygous for males
- Train conditional model (sex as covariate, not removed)
- Stratified validation across ancestry Ã— sex
- Ablation: measure variant effects by sex

**Week 4: Optionalâ€”INLP/Fairness**
- If subspace removal incomplete: 1-2 INLP iterations
- Optional: add fairness constraints to loss

---

## Key Dependencies

- Mayo retrospective cohort genotypes (VCF format)
- Ancestry labels from PCA (not self-reported)
- Sex/chromosome information (from phenotype data)
- Relatedness file for SNP pruning (~50k independent SNPs)

---

## Deliverables

- âœ… Population ancestry PCA embeddings + metadata
- âœ… Stratified data loader (balanced by PC deciles)
- âœ… Trained ancestry regression head + weight matrix
- âœ… Debiased encoder embeddings
- âœ… Fairness validation notebook (ancestry Ã— sex matrices)
- âœ… GitHub repo: reproducible debiasing pipeline
- âœ… Visualization: embedding dimension Ã— ancestry correlation heatmap

---

## Open Questions

1. **Post-hoc stratification vs training-time:** Should we also validate fairness on held-out ancestry groups not seen during stratified sampling?
2. **Prospective trial (Aim 2):** How to infer ancestry PCAs for new patients? (Use same PCA transform from training?)
3. **Cascade to Aim 2:** If fairness issues emerge in prospective trial, fall back to Group DRO or separate sex-stratified models?

---

**Status:** Ready for Phase B Phase 1 implementation  
**Integration:** Merge into `pre-phd_intensive_study_plan.md` Track B Phase 1