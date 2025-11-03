# Linkage Disequilibrium (LD) as Confounder & Fairness Tool
**Pre-PhD Phase B + Paper 1 Integration**

---

## Problem: LD Breaks Ancestry Fairness

**Why LD matters:**

Ancestry populations have different LD structures due to:
- Different demographic history (bottlenecks, admixture)
- Different generations since mutation origin
- Population-specific recombination rates

**Result:**

```
Causal variant: rs_true (AFR MAF 2%, EUR MAF 0.5%)
Tagged variant: rs_ld in strong LD with rs_true in EUR (rÂ²=0.8)
            but independent in AFR (rÂ²=0.1)

Model trained on mixed ancestry:
  - Learns rs_ld is predictive (EUR training data)
  - Assumes rs_ld works in AFR (but it doesn'tâ€”wrong LD structure)
  - Fails on AFR subjects with rs_true but not rs_ld
  
â†’ Model unfair across ancestry, confounded by LD structure differences
```

---

## Two-Part Strategy

### **PART 1: PROACTIVE** (Dec 2025, Phase B Phase 1, 8-10 hrs)
Reduce LD confounding upfront

### **PART 2: POST-HOC** (Jun-Jul 2027, Paper 1 validation, 30-40 hrs)
Explain and validate what model actually learned

---

## PART 1: PROACTIVE LD HANDLING (8-10 hrs)

### **1.1 Ancestry-Stratified LD Pruning** (6-8 hrs)

**During feature engineering (Dec 2025):**

```python
# For each ancestry in training cohort:
for ancestry in ['EUR', 'AFR', 'EAS', 'SAS']:
  # Compute LD matrix (rÂ² between SNPs)
  ld_matrix_ancestry = compute_ld(genotypes[ancestry], r2_threshold=0.1)
  
  # LD-prune independently in each ancestry
  pruned_variants_ancestry = ldprune(ld_matrix_ancestry)

# Take union: variants that pass pruning in ALL ancestries
final_variants = union(pruned_variants_EUR, 
                       pruned_variants_AFR, 
                       pruned_variants_EAS, 
                       pruned_variants_SAS)
```

**Effect:**
- Removes variants that are tagged by others in *any* ancestry
- Keeps "independent" variants that aren't LD-proxies
- Reduces feature dimension, removes LD leakage

**Output:**
```
Started with: 1,000 CDS variants across cohort
After ancestry-stratified LD pruning: 720 variants
  - Removed 280 variants (mostly LD-tagged across ancestries)
  - Retained variants: mix of low-frequency and ancestry-specific
```

**Parameters:**
- LD threshold: rÂ² < 0.1 (conservative; could use 0.2 if needed)
- Ancestry groupings: use 1000G superpopulations (EUR, AFR, EAS, SAS, AMR)
- Window size: 1 Mb (standard)

**Deliverables:**
- Variant list: final_variants (n=720) with ancestry-pruning metadata
- LD matrix per ancestry (for later fine-mapping)

---

### **1.2 Optional: LD-Corrected Loss During Training** (2-4 hrs, skip if time tight)

**If pursuing, add to phenotype loss:**

```
Standard loss:
  L_phenotype = cross_entropy(pred, true)

LD-aware loss:
  L_ld = Î£áµ¢â±¼ |corr_ancestry(varáµ¢, varâ±¼)| Ã— |effect_i| Ã— |effect_j|
  
Total loss:
  L = L_phenotype + Î» Ã— L_ld
  
where Î» ~ 0.01-0.1 (tuned on validation set)
```

**Effect:** Model penalizes learning from highly correlated variants; prefers "independent" effects

**Pros:** Single model; learns ancestry-aware importance  
**Cons:** Extra hyperparameter; modest benefit vs pruning

**Decision:** Include only if Paper 1 validation shows LD-related fairness issues; otherwise skip.

---

## PART 2: POST-HOC LD ANALYSIS (30-40 hrs)

**During Paper 1 manuscript preparation (Jun-Jul 2027):**

### **2.1 LD-Corrected Variant Effects** (12-18 hrs)

**After training, re-analyze ablation accounting for LD:**

```
For each subject variant v_i:
  1. Get ablation effect (existing method): Î”e_i
  2. Find LD partners in subject's ancestry:
     LD_partners = {v_j : rÂ²_ancestry(v_i, v_j) > 0.3}
  
  3. Conditional effect (regress out LD):
     Î”e_i_conditional = Î”e_i - Î£â±¼ Î²_j Ã— Î”e_j
     where Î²_j = regression coeff of LD partners on Î”e_i
  
  4. Compare to population baseline:
     - Does Î”e_i_conditional match gnomAD carriers?
     - If yes â†’ likely causal signal
     - If no â†’ confounded by LD in this subject
```

**Output per variant per subject:**

```
Subject 001, Gene BRCA1:

Variant rs123 (p.Arg123His):
  Raw ablation effect: -0.80Ïƒ
  LD partners in subject's EUR ancestry (rÂ²>0.3):
    - rs456 (rÂ²=0.6): ablation effect -0.50Ïƒ
    - rs789 (rÂ²=0.4): ablation effect -0.35Ïƒ
  LD-conditional effect: -0.75Ïƒ (regressed out LD partners)
  Population EUR carriers (n=450): -0.76Ïƒ Â± 0.2
  â†’ LD-corrected effect matches population âœ“ (likely causal)

Variant rs456 (p.Leu456Val):
  Raw ablation effect: +0.30Ïƒ
  LD partners: none (rÂ²<0.3 with all others)
  LD-conditional effect: +0.30Ïƒ (no change)
  Population EUR carriers (n=150): +0.31Ïƒ Â± 0.15
  â†’ Clean signal, no LD confounding âœ“
```

**Interpretation guide:**
- If LD-conditional >> raw â†’ variant was LD-tagged, not causal
- If LD-conditional â‰ˆ raw â†’ variant independent, likely causal
- If LD-conditional matches population â†’ causal signal validated

**Deliverables:**
- Supplementary Table: LD-corrected effects for all variants in Paper 1 cohort
- Visualization: raw vs LD-corrected effects (scatter plot)
- Flagged variants: high LD confounding (warn clinicians)

---

### **2.2 Ancestry-Stratified Fine-Mapping** (15-25 hrs)

**Which variants are likely causal vs just LD-tagged?**

**Method 1: FINEMAP/SuSiE integration** (if time)

```
Use model predictions as pseudo-GWAS summary statistics:
  1. For each variant: compute effect size from ablation
  2. Compute LD matrix per ancestry
  3. Run FINEMAP/SuSiE separately per ancestry:
     - EUR LD + EUR effects â†’ credible set EUR
     - AFR LD + AFR effects â†’ credible set AFR
     - EAS LD + EAS effects â†’ credible set EAS
  
  4. Compare across ancestry:
     - Variant in 95% credible set in ALL ancestries? â†’ strong causal signal
     - Variant in EUR set only? â†’ EUR-specific tag or confounding
```

**Method 2: Simpler Conditional Analysis** (if limited time, recommended)

```
For each gene:
  1. Rank variants by |ablation effect|
  2. For each variant v_i, condition on all higher-effect variants:
     v_i_conditional_effect = v_i effect after removing correlated higher-effects
  
  3. If v_i_conditional becomes near-zero â†’ likely tagged by higher-effect variant
     If v_i_conditional stays strong â†’ independent causal
```

**Output per gene:**

```
Gene TP53:

Variant rank | Variant | Raw Effect | Conditional Effect | 95% in EUR | 95% in AFR | ClinVar  |
1            | rs_path1| -1.2Ïƒ     | -1.2Ïƒ              | 96%        | 97%        | Path     |
2            | rs_ld1  | -0.5Ïƒ     | -0.1Ïƒ              | 5%         | <1%        | Benign   |
3            | rs_path2| -0.8Ïƒ     | -0.8Ïƒ              | 92%        | 95%        | Path     |
4            | rs_ld2  | -0.3Ïƒ     | -0.05Ïƒ             | 3%         | <1%        | Benign   |

Interpretation:
  - rs_path1, rs_path2: strong, independent causal signals (consistent across ancestry)
  - rs_ld1, rs_ld2: LD-tagged; conditional effect collapses (not causal)
```

**Validation:**
- Do fine-mapped variants match ACMG/ClinVar pathogenic classifications?
- Are LD-tagged variants benign/VUS?
- Do fine-mapped variants replicate across ancestry?

**Deliverables:**
- FINEMAP output per ancestry (or conditional ranking table)
- Supplementary Figure: 95% credible set overlap across ancestry
- Validated causal variant list (for Paper 1 Table)

---

### **2.3 LD Structure Ã— Fairness Correlation** (6-12 hrs)

**Does model fairness track LD differences?**

```
Hypothesis: LD structure mismatch causes ancestry fairness gaps

Analysis:
  For each subject Ã— ancestry pair:
    1. Compute LD profile for subject's variants:
       ld_profile_i = [rÂ²(var_i, var_j) for all j]
    
    2. Compute distance to training ancestry LD structures:
       ld_distance = ||ld_profile_i - ld_profile_train||
    
    3. Measure prediction error:
       error = |accuracy_subject_ancestry - accuracy_training_ancestry|
    
    4. Correlate:
       corr(ld_distance, error) = ?
```

**Expected result:**

```
If LD structure is major fairness driver:
  - AFR subjects with EUR-like LD patterns â†’ lower error
  - AFR subjects with different LD patterns â†’ higher error
  - Correlation: r > 0.4 (significant)

If LD is not the issue:
  - Correlation: r < 0.2 (noise)
  - Other factors dominating (rare variants, allele frequency effects)
```

**Output:**
- Scatter plot: LD distance vs prediction error (per ancestry)
- Correlation coefficient + p-value
- Report: "LD structure accounts for X% of fairness gap"

**Deliverables:**
- Figure: LD structure Ã— fairness correlation
- Table: which ancestries most affected by LD mismatch

---

## Integration into Timeline

### **Phase B Phase 1 (Dec 2025, 8-10 hrs)**
- Compute ancestry-stratified LD matrices (~3 hrs)
- Implement LD-pruning in feature selection (~4 hrs)
- Document: LD pruning metadata (which variants retained/removed) (~1-2 hrs)

### **Paper 1 Validation (Jun-Jul 2027, 30-40 hrs)**

**Week 1: LD-corrected effects** (~12-18 hrs)
- Reanalyze all subject variants post-hoc
- Compute conditional effects (regress LD partners)
- Compare to population baselines

**Week 2: Fine-mapping** (~12-18 hrs)
- Run FINEMAP per ancestry (or simpler conditional ranking)
- Generate credible sets
- Validate against ClinVar

**Week 3: Fairness analysis** (~6-12 hrs)
- LD distance Ã— prediction error correlation
- Quantify LD's contribution to fairness gaps
- Write interpretation section

---

## Deliverables by Phase

### **Phase B Phase 1 (Dec 2025)**
- âœ… Ancestry-stratified LD matrices (EUR, AFR, EAS, SAS)
- âœ… Final variant list (post-pruning, n~720)
- âœ… Pruning metadata: which variants removed/retained per ancestry
- âœ… LD correlation structure visualization

### **Paper 1 Validation (Jun-Jul 2027)**
- âœ… Supplementary Table: LD-corrected effects (all variants)
- âœ… Supplementary Table: Fine-mapped causal variants per gene per ancestry
- âœ… Supplementary Figure: LD credible set overlap across ancestry
- âœ… Supplementary Figure: LD structure Ã— fairness correlation
- âœ… Main text: "LD structure accounts for X% of fairness gaps; Y% due to rare variants/allele frequency"
- âœ… GitHub: reproducible LD analysis pipeline

---

## Validation Criteria (Paper 1 Gate)

**Pass if:**
1. LD-corrected variant effects match population baselines (r > 0.8)
2. Fine-mapped causal variants consistent across â‰¥2 ancestries (95% credible set overlap > 50%)
3. Fine-mapped variants match ACMG/ClinVar (precision > 85%)
4. LD structure accounts for â‰¤50% of fairness gaps (other factors = ancestry bias, allele freq, sample size)

**Fail if:**
- LD-corrected effects diverge from population â†’ model learned non-causal patterns
- Credible sets don't overlap across ancestry â†’ ancestry-specific confounding not resolved
- Fine-mapped variants contradict ClinVar â†’ model learning spurious signals

---

## Open Questions

1. **LD threshold:** Use rÂ² > 0.3 for fine-mapping? 0.5? Varies by ancestry sample size & MAF
2. **Multi-variant interactions:** How to handle epistasis (variant pairs with joint LD structure)?
3. **Rare variants:** LD estimation unstable for rare variants (MAF < 1%). Fall back to conservation scores?
4. **Prospective trial (Aim 2):** Report fine-mapped variants only, or include LD-tagged for robustness?
5. **Cross-ancestry LD:** Handle LD between variants in different genes? (usually ignorable)

---

## Key References / Tools

- **FINEMAP**: https://github.com/mattlee821/FINEMAP (fine-mapping)
- **SuSiE**: https://github.com/stephenslab/susieR (fine-mapping, handles LD directly)
- **gnomAD LD**: precomputed LD matrices available (saves computation)
- **Plink**: standard LD computation (`--r2`)

---

**Status:** Ready for Phase B Phase 1 implementation  
**Next step:** Integrate ancestry-stratified LD matrices into data pipeline (Dec 2025)  
**Paper 1 milestone:** LD-corrected interpretation + fairness report (Jun-Jul 2027)