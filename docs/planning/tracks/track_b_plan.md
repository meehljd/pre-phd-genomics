## TRACK B: GENOMICS + ANCESTRY ROBUSTNESS + NETWORK MEDICINE (Dec 2025 - Apr 2026 - 10-12 weeks)

### Phase 1: ACMG + Variant Interpretation + Ancestry Robustness (Dec 2025 - 3 weeks)

**Goal:** Fluency in how clinicians classify variants; understand population genetics and ancestry confounding

**Reading (Week 1: ACMG fundamentals):**
1. **Richards et al. 2015 (ACMG classification guidelines)** - Nature Genetics
   - Focus: 5 pathogenic criteria (PVS, PS, PM, PP, BA) + 4 benign (BS, BP)
   - Time: 2 hours

2. **Lek et al. 2016 (gnomAD)** - Nature
   - Focus: Allele frequency interpretation; constraint metrics (pLI, Z-scores)
   - Time: 1.5 hours

3. **Oren et al. 2020** - "Sparse Modeling of Cell-Type-Specific Gene Expression" (Nature Genetics)
   - Focus: Tissue-specific variant effects
   - Time: 1.5 hours

**Reading (Week 2: Ancestry & Robustness) - NEW:**
4. **Martin et al. 2019** - "Clinical use of current polygenic risk scores may exacerbate health disparities" (Nature Genetics)
   - Focus: Training data bias, performance disparities across populations
   - Time: 1.5 hours

5. **Popejoy & Fullerton 2016** - "Genomics is failing on diversity" (Nature)
   - Focus: Ancestry representation in genomics research, clinical implications
   - Time: 1 hour

**Hands-On (Week 1: ACMG basics, 8-10 hours):**

```python
# Exercise 1: ACMG classification practice (3-4 hours)
- Dataset: 20 variants from ClinVar (known classifications)
- For each variant, manually apply ACMG criteria:
  * PVS1 (null variants in critical genes)
  * PS1 (same amino acid change as established pathogenic)
  * PM1 (mutational hotspot)
  * PM2 (absent/rare in population databases)
  * PP3 (computational evidence)
  * BS1, BS2, BP4 (benign criteria)
- Compare your classification to ClinVar gold standard
- Identify: Where do you disagree? Why?

# Exercise 2: gnomAD constraint metrics (2-3 hours)
- Download gnomAD constraint scores for 100 genes
- Questions:
  * What's the distribution of pLI scores?
  * Which genes are loss-of-function intolerant (pLI > 0.9)?
  * How does this inform variant interpretation?
- Build classifier: Can you predict haploinsufficiency from pLI alone?

# Exercise 3: ACMG automation baseline (3-4 hours)
- Implement simple rule-based ACMG classifier in Python
- Input: Variant (gene, position, ref, alt)
- Output: ACMG classification + confidence
- Test on 50 ClinVar variants
- Benchmark: Accuracy vs. ClinVar gold standard
```

### Optional: Ancestry-Stratified LD Pruning (6-8 hours)

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


**Why:** Remove LD-tagged variants before training to reduce ancestry-specific confounding

**Implementation:**
- Compute LD matrices per ancestry (EUR, AFR, EAS, SAS)
- LD-prune independently in each ancestry (r² < 0.1)
- Take union: variants passing all ancestries
- Output: Final variant set (~70% of input after pruning)

**Deliverables:**
- ✅ Ancestry-stratified LD matrices
- ✅ Final variant list (post-pruning)
- ✅ Pruning metadata table
- ✅ LD correlation heatmap

**Note:** Optional if time tight; can defer to early Aim 1 (Jan 2027)

## Hands-On Week 2: Ancestry Robustness (8-12 hours) - EXPANDED

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

```python
### Exercise 4: Genetic Ancestry PCA (2-3 hours)
[See debiasing doc Section "1. Genetic Ancestry via PCA"]
- Compute PCAs on training cohort
- Visualize PC1 vs PC2
- Bin by deciles for stratified sampling

### Exercise 5: Subspace Removal (2-3 hours)
[See debiasing doc Section "2. Subspace Removal"]
- Train ancestry regression head
- Implement: e_debiased = e - (W^T W) @ e
- Validate: MI(ancestry; e_debiased) drop

### Exercise 6: Sex Handling (1.5 hours)
[See debiasing doc Section "3. Sex as Categorical Covariate"]
- Hemizygous males (0/1), diploid females (0/1/2)
- Condition on sex, don't remove signal

### Exercise 7: Fairness Matrix (1.5 hours)
[See debiasing doc Section "VALIDATION FRAMEWORK"]
- Compute metrics for ancestry × sex
- Success: All cells within 3-5% of best group
```

**Deliverables:**
- ✅ Ancestry PCA embeddings
- ✅ Stratified data loader
- ✅ Debiased embeddings
- ✅ Fairness validation notebook

---

### Embedding Debiasing & Harmonization (10-12 hours)

**Goal:** Systematic batch correction for Evo2 gene embeddings with quantitative evaluation

#### Batch Correction Methods

**Primary: InMoose (ComBat)** (3-4 hours)
- Empirical Bayes shrinkage for continuous embeddings
- Preserves biological covariates (ancestry) via `mod` parameter
- Reference batch normalization option

```python
import inmoose.pycombat as pycombat

# Concatenate gene embeddings (20 genes)
X_all = np.concatenate([gene_embeddings[g] for g in genes], axis=1)

# Apply InMoose correction
corrected = pycombat.pycombat(
    pd.DataFrame(X_all.T),
    batch=metadata['assay_version'],
    mod=pd.get_dummies(metadata['ancestry']),  # preserve ancestry
    ref_batch='v3'  # normalize to v3
)
```

**Alternative: Harmony** (2-3 hours)
- Fast, non-parametric correction
- Comparison baseline for InMoose

```python
from harmonypy import run_harmony

corrected = run_harmony(
    X_all,
    metadata,
    vars_use=['assay_version'],
    max_iter_harmony=20
)
```

#### Evaluation Metrics (3-4 hours)

**Batch Mixing (Lower = Better)**
- Silhouette score (batch labels): Quantify batch separation
- kBET rejection rate: k-nearest neighbor batch effect test  
- LISI (batch): Local Inverse Simpson's Index

**Biological Preservation (Maintain/Increase)**
- Silhouette score (ancestry): Verify structure retained
- LISI (biology): Local diversity of biological labels

```python
# Example evaluation
from sklearn.metrics import silhouette_score

batch_sep_before = silhouette_score(X_all, batch)
batch_sep_after = silhouette_score(X_corrected, batch)

ancestry_sep_before = silhouette_score(X_all, ancestry)
ancestry_sep_after = silhouette_score(X_corrected, ancestry)

print(f"Batch separation: {batch_sep_before:.3f} → {batch_sep_after:.3f}")
print(f"Ancestry signal: {ancestry_sep_before:.3f} → {ancestry_sep_after:.3f}")
```

#### Cross-Batch Validation (2-3 hours)

**Strategy:** Leave-one-batch-out cross-validation
- Harmonize training batches only
- Test on held-out batch (uncorrected)
- Assess generalization to new batches

```python
# For each batch
for held_out_batch in unique_batches:
    train_mask = batch != held_out_batch
    test_mask = batch == held_out_batch
    
    # Correct training only
    X_train_corrected = inmoose_correct(X_train, batch_train)
    
    # Train MLP, test on uncorrected held-out
    clf.fit(X_train_corrected, y_train)
    auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
```

#### Gene-Specific Analysis (2-3 hours)

**Sensitivity Analysis:** Per-gene performance impact
- Compare uncorrected vs corrected for each of 20 genes
- Identify genes where correction degrades performance
- Flag for gene-specific strategies

```python
for gene in genes:
    scores_before = cross_val_score(clf, gene_embeddings[gene], y, scoring='roc_auc')
    scores_after = cross_val_score(clf, corrected_genes[gene], y, scoring='roc_auc')
    delta = scores_after.mean() - scores_before.mean()
```

**Success Criteria:**
- Batch silhouette reduction >50%
- Ancestry silhouette maintained within 10%
- Cross-batch AUROC >0.70
- <3 genes with Δ AUROC < -0.02

**Deliverables:**
- ✅ Corrected embeddings (20 T1D genes)
- ✅ Metrics report (before/after)
- ✅ Cross-batch validation results
- ✅ Per-gene sensitivity analysis + visualization
- ✅ Method comparison (uncorrected, InMoose, Harmony)

---

**Hands-On (Week 3: HPO + Phenotype Prediction, 6-8 hours):**

```python
# Exercise 7: HPO exploration (2-3 hours)
- Download HPO database (free)
- For 3 rare diseases (e.g., BRCA1-associated cancer, Marfan, cystic fibrosis):
  * Map disease → phenotype terms
  * Build HPO hierarchy visualization
  * Identify: Are phenotypes clustered? Predictable from genotype?

# Exercise 8: Gene-phenotype linking baseline (2-3 hours)
- Dataset: 100 rare disease patients (genotype + phenotype)
- Model baseline: Simple logistic regression
  * Input: One-hot encoded gene variants
  * Output: HPO term (binary classification)
- Evaluate: AUC, precision@5
- Compare to random baseline
- Ancestry-aware evaluation: Stratify by population (EUR, AFR, etc.)
  * Report performance separately per ancestry group
  * Identify: Does model perform worse on underrepresented populations?

# Exercise 9: Understand IMPPROVE approach (2-3 hours)
- Read their methodology for isoform-level phenotype prediction
- Replicate their simple baseline on public data
- Document: Key insights for your Aim 1
```

**Output:**
- ACMG classification notebook
- gnomAD constraint analysis
- Ancestry stratification analysis document (1000 words)
- PCA ancestry correction implementation
- Training data bias mitigation plan for Aim 1
- HPO exploration notebook
- Gene-phenotype baseline model with ancestry-stratified validation

---

### Phase 2: Genomic Foundation Model Selection + Network Medicine (Jan-Feb 2026 - 3-4 weeks)

**DECISION POINT: Genomic Foundation Model Selection (Week 1 of Phase 2)**

**Goal:** Evaluate and select genomic foundation model (Enformer, Nucleotide Transformer, or alternatives) based on usability, performance, and integration with your thesis goals.

**Evaluation Criteria (2-3 hours per model, test 2-3 candidates):**

```python
# Model Evaluation Framework

# Test each candidate on:
1. Installation & setup (30 min)
   - Can you install locally?
   - GPU requirements?
   - Documentation quality?

2. Basic functionality (1 hour)
   - Load model
   - Test on 5 genomic sequences (500bp, 2kb, 10kb)
   - Extract embeddings
   - Performance (inference time, memory)

3. Attention extraction (1 hour)
   - Can you extract attention matrices?
   - Layer-wise access?
   - Visualization feasibility?

4. Variant analysis (1 hour)
   - Test on 5 pathogenic variants
   - Does model provide useful signal for variant interpretation?
   - Compare to ESM2 attention patterns (from Track A)

# Candidates to evaluate:
- Enformer (DeepMind) - 100kb context, strong on regulatory
- Nucleotide Transformer (InstaDeep) - Up to 1000bp, general purpose
- Hyena-DNA - 1M context, efficient
- Others: DNABERT-2, Genomic-FM, your dual-llama if genomic

# Decision rubric:
- Usability: ★★★★★
- Performance: ★★★★★
- Documentation: ★★★★★
- Interpretability: ★★★★★
- Maintenance/support: ★★★★★

# Final selection documented in: docs/genomic_model_selection.md
```

**Reading (Network medicine & systems biology - Weeks 2-3):**

1. **Guney et al. 2016** - "Network-based in silico drug efficacy screening" (Nature Communications)
   - Focus: Network proximity kernel; shortest-path metrics for disease gene prioritization
   - Time: 2 hours

2. **Menche et al. 2015** - "Disease networks. Uncovering disease-disease relationships through the incomplete human interactome" (Nature Communications)
   - Focus: Network modularity and disease modules; how network architecture predicts phenotypic pleiotropy
   - Time: 1.5 hours

3. **Barabási et al. 2011** - "Network medicine: a network-based approach to human disease" (Nature Reviews Genetics)
   - Focus: Hub genes vs. peripheral genes; network degree predicts phenotypic breadth; scale-free topology implications
   - Time: 1.5 hours

**Hands-On (Network-based phenotype prediction - Weeks 2-4, 18-22 hours):**

```python
# Exercise 10: Patient-scale heterogeneous graph construction (5-6 hours)
# ** Core for Project 3: Patient-scale GNN **
- Dataset: 100-150 rare disease patients (genotype + phenotype + network)
- For each patient:
  * Extract variant genes from VCF
  * Query STRING/Reactome/ENCODE for: PPI, TF regulation, pathways, co-expression
  * Build heterogeneous subgraph: K=50 genes (mutated + 1-2 hop neighbors)
  * Assign node features: pathogenicity (your encoder-decoder) + expression + conservation + centrality
  * Create edge tensors per type: PPI, TF, pathway, co-expr, homology
  * Verify: ~100-150 PyTorch Geometric heterogeneous graphs ready for GNN training

# Exercise 11: Network propagation baseline (4-5 hours)
# ** Baseline comparison for Project 3 **
- Implement heat diffusion kernel on PPI network (Guney et al. approach)
- Input: Variant in gene X on patient with phenotype profile P
- Algorithm: Propagate variant "signal" through network starting at X
- Output: Top 20 genes by diffusion score (most likely to contribute to P)
- Test on 5 known Mendelian disease cases: does network proximity improve gene ranking?
- Comparison: Network + ACMG vs. ACMG alone (benchmark against baseline)
- Save this as Baseline 2 for Project 3 evaluation

# Exercise 12: Simple GNN on global PPI (3-4 hours)
# ** Baseline comparison for Project 3 **
- Train single-layer GCN on shared PPI network (no edge type distinction)
- Compare performance to: ACMG alone, heat diffusion kernel, heterogeneous GNN (later)
- Save as Baseline 3 for Project 3

# Exercise 13: Multi-layer network & pathway logic (3-4 hours)
- For 5-10 genes from Exercise 7:
  * Map upstream regulators (TFs that control them)
  * Map downstream targets (genes they regulate)
  * Identify regulatory motifs (feedforward loops, feedback, etc.)
  * Hypothesis: Do genes in coherent FFLs exhibit coordinated phenotypes?
  * Example: Gene disruption in coherent FFL → oscillation loss → specific phenotype signature
- Validate: Does regulatory logic align with disease mechanism in literature?
```

**Output:**
- Genomic foundation model selection report (`docs/genomic_model_selection.md`)
- PPI network analysis notebook with degree vs. phenotype analysis
- Network propagation implementation (Python module) + validation on 5 cases
- Multi-layer network visualization + mechanistic interpretation document (1500 words)
- Comparison table: ACMG alone vs. ACMG + network propagation on test cases

---

### Phase 3: Rare Disease Diagnostic Workflows & Clinical Context (Mar 2026 - 2 weeks)

**Goal:** Deep understanding of diagnostic odyssey; how Mayo's system works; integration points

**Reading (Mix of papers + internal Mayo resources):**

1. **Search PubMed "Undiagnosed Diseases Network" (recent papers 2023-2025)**
   - Read 2-3 representative papers on diagnostic outcomes
   - Focus: Patient journeys, diagnostic success rates, barriers
   - Time: 2 hours

2. **Eric Klee lab publications on rare disease diagnosis**
   - Download 2-3 recent Mayo papers
   - Focus: How Mayo currently approaches diagnosis
   - Time: 1.5 hours

3. **Clinical workflow papers:**
   - **Rajkomar et al. 2018** - "Scalable and Accurate Deep Learning for Electronic Health Records" (NPJ)
     * Focus: EHR integration, clinical deployment challenges
     * Time: 1 hour

   - **Schieppati et al. 2008** - "Why rare diseases are an important medical and social issue" (Lancet)
     * Focus: Diagnostic odyssey, patient impact
     * Time: 1 hour

**Hands-On:**

```python
# Exercise 14: Diagnostic odyssey case studies (3-4 hours)
- Find 5 published rare disease case reports (varied complexity)
- For each case:
  * Timeline: How long from symptoms → diagnosis?
  * Tests ordered: WES, WGS, gene panels?
  * Barriers: What delayed diagnosis?
  * Resolution: How was diagnosis achieved?
- Document: Common patterns across diagnostic odysseys

# Exercise 15: Mayo workflow mapping (2-3 hours)
- Interview Eric Klee lab members (or read Mayo documentation)
- Map current diagnostic workflow:
  * Patient intake
  * Phenotype collection (HPO terms?)
  * Sequencing (WES vs WGS?)
  * Variant prioritization pipeline
  * Clinical interpretation meeting
  * Report generation
- Identify: Where does AI/ML integration make sense?

# Exercise 16: Clinical integration design (3-4 hours)
- Design document: How would your AI agent fit into Mayo workflow?
- Key questions:
  * Input format: EHR data? VCF? Phenotype list?
  * Output format: Ranked gene list? ACMG report?
  * Clinician interface: Dashboard? Report PDF?
  * Explainability requirements: What do clinicians need to see?
- Mockup: Sketch clinician-facing interface (can be hand-drawn)
```

**Output:**
- Diagnostic odyssey case study document (1500 words)
- Mayo workflow map (visual diagram)
- Clinical integration design document (2000 words) + mockup

---

### Track B Summary (By Apr 30, 2026):
- ✅ Fluent in ACMG classification
- ✅ Understand ancestry confounding and mitigation strategies
- ✅ Embedding debiasing pipeline with InMoose/Harmony
- ✅ Genomic foundation model selected and validated
- ✅ Can build network-based gene prioritization systems
- ✅ Understand Mayo clinical workflows
- ✅ Ready to design Aim 1 retrospective validation
- **Time investment: ~115-142 hours** (expanded from 105-130 to include embedding debiasing)