# Week 3 Study Plan: Ancestry Debiasing Foundation

**Phase B1 - Week 3 Goals**  
**Total budget: 30-35 hours**

Establish ancestry stratification framework and baseline fairness metrics before implementing subspace removal debiasing.

---

## Task 1: Ancestry Clustering & Labeling (4-5 hrs)

**Deliverable:** Ancestry labels (EUR/AFR/EAS/SAS/AMR) for all 97k samples

### Subtasks:

1. **PCA visualization** (1 hr)
   - Plot PC1 vs PC2, PC1 vs PC3
   - Color by self-reported race (if available)
   - Identify cluster structure

2. **Cluster assignment** (2 hrs)
   - Option 1: k-means clustering (k=4 or 5)
   - Option 2: gnomAD reference projection (if you have reference PCs)
   - Generate `ancestry_labels.txt` (sample_id, ancestry)

3. **Validation** (1-2 hrs)
   - Check cluster sizes (expect EUR dominant ~95%)
   - Compute within-cluster PC variance
   - Flag ambiguous samples (multiethnic/outliers)

**Output:** `ancestry_labels.txt`, PC plots with cluster assignments

---

## Task 2: Ancestry Subspace Removal (6-8 hrs)

**Deliverable:** PC-debiased genotype matrices per ancestry group

### Subtasks:

1. **Load variant matrix** (1 hr)
   - Convert pgen → numpy array (samples × variants)
   - May need chunked processing (87 GB file)

2. **Implement projection** (2-3 hrs)
```python
   # X_debiased = X - X @ (PC @ PC^T)
```
   - Test with top 10 PCs first
   - Validate: PC1-10 should be ~0 in debiased data

3. **Stratified debiasing** (2-3 hrs)
   - Create separate debiased matrices per ancestry
   - Save as: `genome_qc_debiased_{ancestry}.pgen`

4. **Sanity checks** (1 hr)
   - Verify variant counts unchanged
   - Check MAF distributions pre/post
   - Confirm PC correlations removed

**Output:** Debiased pgen files (4 ancestry groups + pooled)

---

## Task 3: Baseline Model Training (10-12 hrs)

**Deliverable:** Fairness metrics for 3 conditions × 4 ancestries

### Subtasks:

1. **Data preparation** (2 hrs)
   - Load ClinVar pathogenic/benign labels
   - Split train/test stratified by ancestry
   - Create 3 datasets: raw, LD-pruned, PC-debiased

2. **Model training** (4-5 hrs)
   - XGBoost with default hyperparameters
   - Train on: (a) all variants, (b) pruned only, (c) debiased
   - 5-fold CV within each ancestry group

3. **Evaluation** (3-4 hrs)
   - Compute per-ancestry: accuracy, sensitivity, specificity, AUROC
   - Generate confusion matrices
   - Calculate fairness metrics (max accuracy gap across groups)

4. **Visualization** (1-2 hrs)
   - Heatmap: 3 conditions × 4 ancestries × 4 metrics
   - Bar plot: accuracy gaps (EUR vs others)
   - Save results: `baseline_fairness_metrics.csv`

**Output:** Baseline performance table, fairness gap quantification

---

## Task 4: Weekly Report (2-3 hrs)

**Deliverable:** Jupyter notebook summarizing Week 3

### Sections:

1. Ancestry clustering results (PC plots, cluster sizes)
2. Subspace removal validation (PC variance pre/post)
3. Baseline fairness analysis (performance gaps)
4. Key insights for Paper 1 (LD vs ancestry confounding)

---

## Week 3 Timeline

- **Mon-Tue:** Task 1 (ancestry labels)
- **Wed-Thu:** Task 2 (subspace removal)
- **Fri-Sun:** Task 3 (baseline models)
- **Mon (Week 4):** Task 4 (report)

---

## Critical Dependencies

- Ancestry labels needed before stratified debiasing
- Baseline metrics establish fairness gap to close
- Week 3 output feeds directly into Paper 1 (LD confounding analysis)

---

## Potential Blockers

1. **Memory:** 87 GB pgen may not fit in RAM → use chunked processing
2. **Runtime:** Model training on 28M variants may be slow → subsample to 1M for prototyping
3. **Label quality:** If Tapestry is 95% EUR, non-EUR groups may be underpowered

---

## Success Criteria

- [ ] Ancestry labels assigned to all samples with >95% confidence
- [ ] PC-debiased genotype files created and validated
- [ ] Baseline fairness metrics computed across 3 conditions
- [ ] Accuracy gap quantified (target: identify >10% disparity to motivate debiasing)
- [ ] Week 3 report notebook completed with key visualizations

---

**Next Steps:** Begin Task 1 with PCA visualization notebook