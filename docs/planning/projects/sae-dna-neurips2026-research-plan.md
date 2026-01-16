# Sparse Autoencoders for DNA Foundation Models: NeurIPS 2026 Research Plan

**Project Lead:** Josh Meehl
**Target Venue:** NeurIPS 2026 Main Conference (Submission deadline: May 2026)
**Project Duration:** January - May 2026 (4.5 months)
**Status:** Planning (Draft v1.0)
**Date Created:** 2026-01-15

---

## Executive Summary

This research plan outlines a path to a NeurIPS 2026 submission on mechanistic interpretability for DNA foundation models using sparse autoencoders (SAEs). The core contribution is extending SAE methodology from protein language models (InterPLM, Nature Methods 2025) to the DNA domain, with specific application to inflammatory bowel disease (IBD) phenotype prediction. The project leverages Mayo Clinic's infrastructure (Helix 100k, Regeneron 80k cohorts), existing gfm-platform capabilities (two-stage embedding pipeline, layer hunting, Nucleotide Transformer implementation), and planned GoodFire AI collaboration.

**Primary Hypothesis:** Sparse autoencoders trained on DNA foundation model representations will discover interpretable features corresponding to regulatory elements and disease mechanisms, enabling faithful variant-level attribution for phenotype prediction.

**Success Criteria:** Identify 100+ biologically interpretable features with validated correspondence to known IBD-associated regulatory elements (JASPAR motifs, GWAS loci, ClinVar variants), demonstrate computational necessity/sufficiency, and achieve method paper acceptance at NeurIPS 2026.

---

## 1. Research Questions and Hypotheses

### 1.1 Primary Research Questions

**RQ1 (Feasibility):** Can sparse autoencoders discover interpretable features in DNA foundation model representations, despite the challenges of longer context (6kb-131kb vs <1kb proteins) and less characterized sequence features compared to protein structure?

**RQ2 (Biological Validity):** Do SAE-derived features correspond to known biological entities (transcription factor binding sites, regulatory motifs, variant effect patterns) in a faithful (computationally necessary) manner, or are they merely plausible post-hoc explanations?

**RQ3 (Phenotype Specificity):** Do different layers of decoder models (Nucleotide Transformer) encode phenotype-relevant information at different biological scales (motifs → variant effects → gene functions → disease pathways), and can SAEs recover these hierarchical features?

**RQ4 (Clinical Utility):** Can SAE-based variant attribution provide mechanistic evidence suitable for ACMG-AMP clinical variant interpretation (PP3/BP4 computational evidence), moving beyond black-box pathogenicity scores?

### 1.2 Hypotheses

**H1 (Feature Sparsity):** DNA foundation model representations exhibit superposition (multiple features encoded in same neuron directions), requiring SAEs with sparsity levels 5-10x higher than protein models due to longer context windows.

**H2 (Layer Specialization):** Early layers (0-10) will encode local motifs (JASPAR TF binding sites), middle layers (11-20) will encode variant effects and regulatory logic, and late layers (21-32) will encode gene-level functional concepts.

**H3 (IBD Feature Circuits):** SAE features will form interpretable circuits for IBD risk prediction, with specific features activating for NOD2 LRR domains, IL23R protective variants, ATG16L1 autophagy pathway elements, and HNF4A barrier function motifs.

**H4 (Computational Necessity):** Ablating high-activation SAE features will cause measurable prediction degradation (AUROC drop ≥0.05), demonstrating faithfulness beyond plausibility.

**H5 (Cross-Model Universality):** Core regulatory features (e.g., CTCF, GATA, HNF4A binding sites) will align across different DNA foundation models (NT, DNABERT-2, Evo-2), indicating model-independent biological signal recovery.

### 1.3 Success Definitions

**Minimum Success (NeurIPS Workshop):**
- Train SAEs on Nucleotide Transformer representations
- Identify 50+ features with plausible biological interpretation (JASPAR motif matches, p < 0.01)
- Demonstrate computational necessity for 10+ features (ablation AUROC drop ≥0.03)
- Write methodology paper documenting DNA SAE training challenges

**Target Success (NeurIPS Main Conference):**
- Train SAEs on 2+ DNA models (NT + Evo-2 or DNABERT-2)
- Identify 100+ interpretable features with validated biological correspondence
- Demonstrate computational necessity/sufficiency for 25+ features
- Show phenotype-specific feature circuits for IBD (NOD2/IL23R/ATG16L1)
- Cross-validate with GWAS effect sizes, ClinVar pathogenicity, deep mutational scanning data
- Achieve novel scientific claims about DNA foundation model representations

**Aspirational Success (Nature Methods trajectory):**
- All target success criteria plus:
- Cross-model feature alignment demonstrating universal regulatory grammar
- Clinical validation: SAE features improve ACMG-AMP variant classification (compare to VEP scores)
- Prospective validation: SAE-based variant ranking predicts held-out IBD cases (AUROC ≥0.70)
- Public release: SAE catalog for NT (similar to InterPLM protein features)

---

## 2. Technical Challenges (Specific and Honest)

### 2.1 DNA-Specific Challenges vs. Protein SAEs

| Challenge | Protein (InterPLM) | DNA (This Work) | Mitigation Strategy |
|-----------|-------------------|-----------------|---------------------|
| **Context Length** | <1024 tokens | 6kb-131kb (1,500-32,768 tokens) | Hierarchical SAE training; focus on 6kb windows initially |
| **Feature Catalog** | UniProt, PDB, GO terms | JASPAR (limited), GWAS (sparse), ClinVar (variant-level) | Build custom DNA feature catalog; integrate RegulomeDB, ENCODE |
| **Ground Truth** | Protein structure, domains, binding sites | Regulatory elements (less characterized) | Multi-source validation: JASPAR + GWAS + MPRA + DMS |
| **Training Data** | ESM-2 (98M proteins) | NT trained on human genome (3Gb, repetitive) | Use diverse genomic regions; balance gene-rich vs gene-desert |
| **Validation Gold Standard** | Mutagenesis, structure | ClinVar (limited), GWAS (population-level) | Combine computational necessity + biological validation |

### 2.2 SAE Training at Scale

**Challenge 2.2.1: Compute Requirements**
- **Estimate:** Training SAEs on NT (86M parameters) for 1M sequences at layer 15 (1024-dim embeddings)
  - Forward passes: 1M sequences × 32 layers = 32M embeddings to cache (if training SAEs per layer)
  - SAE training: ~500k steps per SAE (based on InterPLM), 10-20 hours on single GPU per layer
  - **Total:** ~640 GPU-hours for 32 SAEs (one per layer)
- **Mitigation:**
  - Prioritize layers based on layer hunting results (train SAEs for top 5 layers first)
  - Leverage Mayo HPC GPU nodes (already available)
  - Use embedding cache from two-stage pipeline (ADR-002) to avoid re-extraction

**Challenge 2.2.2: Sparsity Tuning**
- InterPLM used L1 penalty coefficient λ = 0.001-0.01 for protein features
- Longer DNA contexts may require higher sparsity (more features, sparser activation)
- **Mitigation:** Hyperparameter sweep (λ ∈ [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])
- Evaluate sparsity vs. reconstruction error tradeoff using Pareto frontier

**Challenge 2.2.3: SAE Architecture Selection**
- Standard autoencoder: Linear encoder/decoder with ReLU
- Alternative: Gated SAEs (used in recent LLM interpretability work)
- **Decision Point:** Week 2 - compare architectures on small scale before full training

### 2.3 Feature Interpretability in DNA (Less Characterized Than Proteins)

**Challenge 2.3.1: Limited Ground Truth Catalogs**
- JASPAR: ~700 TF motifs (vs. 20,000+ protein domains in UniProt)
- ClinVar: ~2M variants but dominated by coding variants, sparse in regulatory
- GWAS: ~300k associations but population-level, not mechanistic

**Mitigation:**
- Build multi-source validation framework:
  1. JASPAR motif scanning (PWM matches, p < 0.001)
  2. ENCODE cCRE overlap (candidate cis-regulatory elements)
  3. GWAS catalog LD-proxy enrichment
  4. ClinVar pathogenic variant overlap
  5. RegulomeDB functional scores
- Define "interpretability score" combining all sources
- Negative control: SAE trained on shuffled embeddings should show random interpretation scores

**Challenge 2.3.2: Distinguishing Plausible from Faithful**
- Risk: SAE features may match JASPAR motifs by chance or due to training data artifacts
- **Validation Hierarchy:**
  1. **Plausibility:** Feature activates on sequences with JASPAR motif (p < 0.01)
  2. **Computational Necessity:** Ablating feature reduces prediction (AUROC drop ≥0.03)
  3. **Computational Sufficiency:** Inserting feature in new context increases prediction
  4. **Biological Necessity:** Feature overlaps with ClinVar pathogenic variants or GWAS hits

**Challenge 2.3.3: Multi-Gene Aggregation Interpretability**
- IBD prediction uses 15-50 genes per patient
- Current platform aggregates with ConcatPCA or HierarchicalAttention
- **Problem:** How to trace SAE feature activations through aggregation to patient-level prediction?
- **Approach:**
  - Train SAEs on gene-level embeddings (post-aggregation within gene, pre-aggregation across genes)
  - Use integrated gradients to attribute patient prediction to specific genes → sequences → SAE features
  - Visualize feature activation profiles across all genes for case vs. control

### 2.4 Computational Requirements (Detailed Estimates)

**Phase 1: SAE Training (Weeks 1-6)**
- Embedding extraction: Already cached in gfm-platform (two-stage pipeline)
- SAE training: 32 layers × 20 GPU-hours = 640 GPU-hours
- Hyperparameter search: 5 sparsity levels × 3 architectures × 5 layers (top from layer hunting) = 75 SAE training runs × 5 GPU-hours (smaller scale) = 375 GPU-hours
- **Total Phase 1:** ~1,000 GPU-hours (Mayo HPC: 10 A100 GPUs × 100 hours, feasible)

**Phase 2: Feature Characterization (Weeks 7-10)**
- Feature activation profiling: 1M sequences × 32 SAEs (CPU-bound, embarrassingly parallel)
- JASPAR motif scanning: 1M sequences × 700 motifs (existing tools: FIMO, ~500 CPU-hours)
- ENCODE overlap: Genomic coordinate operations (CPU, ~50 hours)
- **Total Phase 2:** ~600 CPU-hours (trivial on HPC)

**Phase 3: Biological Validation (Weeks 11-14)**
- Ablation experiments: 100 features × 1k test samples × 1 forward pass = 100k forward passes (~10 GPU-hours)
- Sufficiency experiments: 50 features × 500 insertion contexts = 25k forward passes (~5 GPU-hours)
- GWAS enrichment: Statistical analysis (CPU, ~20 hours)
- **Total Phase 3:** ~15 GPU-hours + 20 CPU-hours

**Total Compute Budget:**
- **GPU:** ~1,015 GPU-hours (equivalent to 1 A100 for 42 days, or 10 A100s for 4.2 days)
- **CPU:** ~670 CPU-hours (trivial on HPC cluster)
- **Storage:** Embeddings already cached (~500 GB), SAE models (~5 GB per SAE × 32 = 160 GB), feature catalogs (~10 GB)

**Risk Assessment:** Compute is feasible within Mayo HPC allocation. Storage is manageable.

### 2.5 Data Access and PHI Considerations

**Challenge 2.5.1: Real Data Access Timeline**
- Helix 100k: Available (IBD cohort in gfm-platform)
- Regeneron 80k: Pending data use agreement (target: Feb 2026)
- UKBioBank 500k: Pending application (may not arrive in time for NeurIPS)

**Mitigation:**
- Start with Helix 100k (sufficient for method development)
- Use 1000 Genomes + synthetic IBD cohort for early prototyping
- If Regeneron delayed, use public biobanks (TOPMed, gnomAD) for external validation

**Challenge 2.5.2: PHI Compliance**
- Cannot share patient-level embeddings or SAE activations
- Can share: Aggregate feature statistics, motif catalogs, SAE model weights
- **Solution:** All results reported as aggregates; release SAE models with de-identified feature annotations

### 2.6 Baseline Comparisons Needed

**Required Baselines:**
1. **Random Features:** SAE trained on shuffled embeddings (plausibility control)
2. **Individual Neurons:** Raw NT neurons vs. SAE features (does SAE improve interpretability?)
3. **Existing Interpretability Methods:** ISM, Integrated Gradients, Attention (from ADR-034/035/039)
4. **Domain Baselines:** GWAS effect sizes, CADD scores, ClinVar stars
5. **InterPLM Protocol:** Apply InterPLM's protein SAE methodology exactly as described, then compare DNA results

**Comparison Metrics:**
- Interpretability: % features with JASPAR match (p < 0.01)
- Faithfulness: Mean AUROC drop on ablation
- Sparsity: Average % active features per sequence
- Coverage: % of phenotype variance explained by top 10/50/100 features

---

## 3. Phase-by-Phase Plan

### Phase 1: Infrastructure and SAE Training (Weeks 1-6, Jan 15 - Feb 26)

**Week 1-2: Environment Setup and Baseline Establishment**
- **Deliverables:**
  - SAE training codebase (adapt InterPLM implementation or use GoodFire SDK)
  - Embedding extraction for IBD cohort (15 genes × 2,000 patients = 30k sequences)
  - Layer hunting results for NT on IBD phenotype (identify top 5 layers)
  - Baseline interpretability: ISM, IG, Attention on 100 test sequences
- **Key Decisions:**
  - SAE architecture: Standard vs. Gated
  - Sparsity range for hyperparameter sweep
  - GoodFire partnership status (code access, collaboration model)
- **Go/No-Go Criterion:** Embeddings cached, layer hunting shows clear best layers (AUROC spread ≥0.05)

**Week 3-4: SAE Training (Initial Scale)**
- **Deliverables:**
  - Train 5 SAEs (one per top layer from layer hunting)
  - Hyperparameter sweep: 5 sparsity levels × 3 layer choices = 15 SAE training runs
  - Reconstruction error analysis: Validate SAEs recover original embeddings
  - Sparsity analysis: % active features per sequence, feature activation distributions
- **Validation:**
  - Reconstruction R² ≥ 0.95 (SAE faithfully represents embeddings)
  - Sparsity L0 = 5-20 active features per 6kb sequence (interpretable)
- **Go/No-Go Criterion:** At least 1 SAE achieves R² ≥ 0.95 with L0 ≤ 20

**Week 5-6: Full-Scale SAE Training**
- **Deliverables:**
  - Train SAEs for all 32 NT layers using best hyperparameters from Week 3-4
  - Extract feature activations for full IBD cohort (30k sequences)
  - Generate feature activation matrix: [sequences × features × layers]
  - Initial feature visualization: Top 10 activating sequences per feature
- **Quality Checks:**
  - Feature diversity: Features not dominated by GC content or repeat elements
  - Sanity check: Shuffle test (SAEs trained on shuffled embeddings show random patterns)
- **Go/No-Go Criterion:** ≥50% of features show non-trivial activation patterns (not constant/GC-only)

**Milestone 1 (End of Week 6):** SAEs trained for all NT layers, feature activations extracted, ready for biological annotation.

---

### Phase 2: Feature Characterization (Weeks 7-10, Feb 27 - Mar 26)

**Week 7: Biological Annotation Pipeline**
- **Deliverables:**
  - JASPAR motif scanning: For each feature, scan top 100 activating sequences for TF motifs
  - ENCODE cCRE overlap: Map sequences to genomic coordinates, compute overlap with regulatory elements
  - ClinVar variant overlap: Check if features activate on ClinVar pathogenic variants
  - Interpretability score: Combine all sources into single metric per feature
- **Metrics:**
  - % features with JASPAR match (p < 0.01): Target ≥30%
  - % features with ENCODE overlap: Target ≥40%
  - % features with ClinVar enrichment: Target ≥10%
- **Deliverable:** Feature catalog CSV with columns: [feature_id, layer, top_motif, jaspar_pvalue, encode_overlap, clinvar_enrichment, interpretation_score]

**Week 8: IBD-Specific Feature Analysis**
- **Deliverables:**
  - Identify features enriched in IBD cases vs. controls (t-test, FDR < 0.05)
  - Map features to IBD-associated genes: NOD2, IL23R, ATG16L1, HNF4A
  - Literature cross-reference: Do features overlap with known IBD GWAS loci?
  - Pathway enrichment: Do features cluster by biological pathway (autophagy, innate immunity, barrier function)?
- **Key Questions:**
  - Are there features specific to NOD2 LRR domains (known IBD mechanism)?
  - Do any features capture IL23R R381Q protective variant effect?
  - Can we identify regulatory features in HNF4A (barrier function)?
- **Deliverable:** IBD feature report with biological interpretations, literature citations

**Week 9: Cross-Layer Feature Analysis**
- **Deliverables:**
  - Layer specialization analysis: What feature types dominate early vs. middle vs. late layers?
  - Feature similarity across layers: Do similar features appear in multiple layers? (cosine similarity of activation profiles)
  - Hierarchical feature organization: Cluster features by co-activation patterns
- **Hypotheses to Test:**
  - H2 (Layer Specialization): Early layers → motifs, middle → variant effects, late → gene functions
- **Deliverable:** Layer specialization report with visualizations (UMAP of features colored by layer, motif type)

**Week 10: Comparison to Baselines**
- **Deliverables:**
  - Compare SAE features to individual NT neurons: Do SAEs improve interpretability?
  - Compare to ISM/IG/Attention: Do SAEs identify different important positions?
  - Compare to InterPLM protocol: Apply protein SAE methodology to DNA, report differences
  - Random baseline: SAE trained on shuffled embeddings, measure interpretability score
- **Metrics:**
  - Interpretability gain: SAE JASPAR match rate - individual neuron match rate
  - Concordance: Spearman correlation between SAE feature importance and ISM/IG scores
- **Deliverable:** Baseline comparison table for paper Methods section

**Milestone 2 (End of Week 10):** 100+ features biologically annotated, IBD-specific features identified, baseline comparisons complete.

---

### Phase 3: Validation (Computational Necessity/Sufficiency) (Weeks 11-14, Mar 27 - Apr 23)

**Week 11-12: Ablation Experiments (Computational Necessity)**
- **Approach:**
  1. Identify 100 candidate features (top by IBD case-control difference)
  2. For each feature: Ablate (set to 0) in test set embeddings, recompute predictions
  3. Measure AUROC drop: AUROC_original - AUROC_ablated
  4. Threshold: Features with AUROC drop ≥0.03 are computationally necessary
- **Deliverables:**
  - Ablation results table: [feature_id, AUROC_drop, p_value, biological_annotation]
  - Identify 25+ features with significant necessity (AUROC drop ≥0.03, p < 0.05)
  - Analyze ablation patterns: Do NOD2/IL23R/ATG16L1 features show highest necessity?
- **Expected Results:**
  - Target: 25-50% of tested features show computational necessity
  - Negative control: Random features (from shuffled SAE) should show AUROC drop ~0
- **Go/No-Go Criterion:** ≥25 features with AUROC drop ≥0.03 (sufficient for paper claims)

**Week 13: Sufficiency Experiments**
- **Approach:**
  1. Identify 50 necessary features from Week 11-12
  2. For each feature: Insert (set to max activation) in control sequences
  3. Measure AUROC gain: AUROC_inserted - AUROC_original
  4. Sufficiency: Features that increase IBD prediction when inserted
- **Deliverables:**
  - Sufficiency results: [feature_id, AUROC_gain, insertion_context_specificity]
  - Causal feature set: Features showing both necessity AND sufficiency
- **Expected Challenge:**
  - Sufficiency is harder to demonstrate than necessity (context-dependent effects)
  - May need to select specific insertion contexts (e.g., insert NOD2 feature only in NOD2 gene sequences)

**Week 14: Biological Validation**
- **Deliverables:**
  - GWAS enrichment: Do necessary features overlap with IBD GWAS loci? (hypergeometric test)
  - ClinVar validation: Do features predict ClinVar pathogenicity? (AUROC for pathogenic vs. benign)
  - Deep mutational scanning: If available, correlate feature activations with DMS fitness scores
  - Literature validation: Map features to experimentally validated regulatory elements (ENCODE, RegulomeDB)
- **Metrics:**
  - GWAS enrichment OR ≥3 (features overlap IBD GWAS loci more than expected by chance)
  - ClinVar prediction AUROC ≥0.65 (features distinguish pathogenic vs. benign)
- **Deliverable:** Biological validation report for paper Results section

**Milestone 3 (End of Week 14):** 25+ features validated as computationally necessary, biological validation complete, ready for paper writing.

---

### Phase 4: Paper Writing (Weeks 15-18, Apr 24 - May 22)

**Week 15: Results Synthesis and Figure Generation**
- **Deliverables:**
  - Main figures (6-8 publication-quality):
    1. SAE training overview (architecture, reconstruction error, sparsity)
    2. Feature catalog summary (JASPAR match rates, ENCODE overlap)
    3. Layer specialization analysis (early vs. middle vs. late features)
    4. IBD feature circuits (NOD2/IL23R/ATG16L1 feature activations)
    5. Ablation results (necessity heatmap, AUROC drops)
    6. Biological validation (GWAS enrichment, ClinVar prediction)
  - Supplementary figures (10-15):
    - Individual feature examples (top activating sequences, motif logos)
    - Baseline comparisons (SAE vs. neurons, SAE vs. ISM/IG/Attention)
    - Hyperparameter sensitivity (sparsity vs. interpretability)
    - Negative controls (shuffled SAE results)

**Week 16-17: Manuscript Drafting**
- **Deliverables:**
  - Abstract (250 words)
  - Introduction (1.5 pages): Problem, prior work (InterPLM), contribution, significance
  - Methods (2.5 pages): SAE training, feature annotation, validation experiments
  - Results (3 pages): Feature catalog, IBD circuits, necessity/sufficiency, biological validation
  - Discussion (1.5 pages): DNA vs. protein SAE differences, clinical implications, limitations
  - Conclusion (0.5 pages): Summary, future work
  - **Target:** 8-9 pages main text (NeurIPS limit: 9 pages + unlimited appendix)

**Week 18: Revision and Submission Preparation**
- **Deliverables:**
  - Internal review: Share with Carl Molnar, Eric Klee (advisor), GoodFire collaborators
  - Incorporate feedback
  - Code release preparation: GitHub repo with SAE training code, feature catalog
  - Reproducibility checklist: NeurIPS requires code/data availability statement
  - Submit by NeurIPS deadline (typically early May)

**Milestone 4 (End of Week 18):** Paper submitted to NeurIPS 2026.

---

## 4. Validation Strategy: Plausible vs. Faithful

### 4.1 Validation Hierarchy (Aligned with gfm-book Ch25)

The distinction between plausible and faithful interpretability (gfm-book sec-ch25-interpretability) structures our entire validation approach.

**Level 1: Plausibility (Necessary but Not Sufficient)**
- **Definition:** Feature activation correlates with known biological entities
- **Methods:**
  - JASPAR motif enrichment (hypergeometric test, p < 0.01)
  - ENCODE cCRE overlap (Fisher's exact test)
  - ClinVar pathogenic variant enrichment
- **Negative Control:** Shuffled SAE (trained on random embeddings) should show no enrichment
- **Interpretation:** Plausibility establishes that features are biologically interpretable, but does NOT prove the model uses them

**Level 2: Computational Necessity (Sufficient for NeurIPS)**
- **Definition:** Removing feature degrades model performance
- **Methods:**
  - Feature ablation: Set feature activation to 0, measure AUROC drop
  - Threshold: AUROC drop ≥0.03 (clinically meaningful)
  - Statistical test: Bootstrap CI (1000 iterations), p < 0.05
- **Negative Control:** Ablating random features should show AUROC drop ~0
- **Interpretation:** Necessity proves the model computationally relies on the feature

**Level 3: Computational Sufficiency (Strong Evidence)**
- **Definition:** Adding feature in new context improves prediction
- **Methods:**
  - Feature insertion: Set feature to max activation in control sequences
  - Measure AUROC gain (should increase IBD prediction)
  - Context specificity: Insertion effect should depend on genomic context
- **Challenge:** Sufficiency is context-dependent (inserting NOD2 feature in non-NOD2 gene may not matter)
- **Interpretation:** Sufficiency + necessity = causal feature

**Level 4: Biological Necessity (Aspirational, Beyond NeurIPS Scope)**
- **Definition:** Experimental perturbation validates computational prediction
- **Methods:**
  - CRISPR: Delete motif, measure phenotype change
  - Reporter assay: Insert motif in heterologous context, measure activity
  - Deep mutational scanning: Correlate SAE activations with fitness
- **Timeline:** Experimental validation requires 6-12 months (not feasible for May submission)
- **Future Work:** Propose in paper Discussion, pursue for Nature Methods follow-up

### 4.2 Comparison to JASPAR Motifs, ClinVar, GWAS

**JASPAR Motif Validation**
- **Approach:** For each SAE feature, scan top 100 activating sequences with FIMO (motif scanner)
- **Metrics:**
  - Motif match rate: % features with JASPAR motif (p < 0.001)
  - InterPLM benchmark: 60% of protein SAE features matched GO terms
  - **Target:** ≥30% of DNA SAE features match JASPAR motifs (DNA is less characterized than proteins)
- **Interpretation:**
  - High match rate → features recover known regulatory grammar
  - Low match rate → features may represent novel patterns (publishable if validated)

**ClinVar Pathogenicity Prediction**
- **Approach:**
  1. Extract ClinVar variants overlapping IBD genes (NOD2, IL23R, ATG16L1, HNF4A)
  2. Compute SAE feature activations for pathogenic vs. benign variants
  3. Train logistic regression: SAE activations → pathogenicity classification
  4. Compare to baseline (CADD, VEP scores)
- **Metrics:**
  - AUROC ≥0.65 (better than random)
  - Feature importance: Do IBD-specific features drive pathogenicity prediction?
- **Clinical Implication:** SAE features could provide PP3/BP4 evidence for ACMG-AMP classification

**GWAS Enrichment**
- **Approach:**
  1. IBD GWAS catalog: ~200 loci from International IBD Genetics Consortium
  2. Map SAE features to genomic coordinates (top activating sequences)
  3. Hypergeometric test: Do necessary features overlap GWAS loci more than expected?
- **Metrics:**
  - Enrichment OR ≥3 (features 3x more likely to overlap GWAS loci)
  - Specific loci: Do NOD2/IL23R/ATG16L1 features overlap their respective GWAS signals?
- **Interpretation:** GWAS enrichment validates that features capture population-level disease associations

### 4.3 Ablation Study Design

**Experimental Setup**
- **Test Set:** 400 IBD cases + 400 controls (held-out from SAE training)
- **Baseline Model:** NT embeddings → layer 15 (from layer hunting) → ConcatPCA aggregator → logistic regression classifier
- **Baseline AUROC:** Assume 0.72 (based on gfm-platform IBD study results)

**Ablation Procedure**
1. Identify 100 candidate features (top by case-control t-test, FDR < 0.05)
2. For each feature f:
   - Extract test set embeddings
   - Set SAE feature f activation to 0 across all test samples
   - Reconstruct embeddings from ablated SAE activations
   - Re-run classifier on reconstructed embeddings
   - Compute AUROC_ablated
   - Effect size: ΔAUROC = AUROC_baseline - AUROC_ablated
3. Statistical significance: Bootstrap 1000 iterations, compute 95% CI
4. Threshold: ΔAUROC ≥0.03 (clinically meaningful) AND p < 0.05

**Expected Results**
- **Hypothesis H4:** 25-50% of tested features show ΔAUROC ≥0.03
- **Specific predictions:**
  - NOD2 LRR features: ΔAUROC ≥0.05 (strongest signal)
  - IL23R protective variant features: ΔAUROC ≥0.04
  - HNF4A barrier function features: ΔAUROC ≥0.03
  - GC content features (artifact): ΔAUROC ~0 (negative control)

**Interpretation Challenges**
- **Redundancy:** Multiple features may encode same information (ablating one has small effect, but ablating all has large effect)
- **Mitigation:** Group ablation (ablate all NOD2 features together)
- **Distributed Encoding:** Important information may be spread across many weak features
- **Mitigation:** Cumulative ablation curves (ablate top 1, top 5, top 10, top 25 features)

### 4.4 Faithfulness vs. Plausibility Distinction

**Case Study 1: GATA Motif Feature**
- **Plausible Scenario:**
  - SAE feature activates on GATA motif sequences (JASPAR match p < 0.001)
  - Biological interpretation: GATA transcription factor binding
  - **BUT:** Ablation shows ΔAUROC = 0.01 (not significant)
  - **Conclusion:** Plausible but not faithful (model doesn't rely on this feature)

- **Faithful Scenario:**
  - Same GATA feature, but ablation shows ΔAUROC = 0.05 (p < 0.001)
  - Sufficiency: Inserting GATA feature in control sequences increases IBD prediction
  - GWAS: Feature overlaps IBD GWAS locus in GATA-binding region
  - **Conclusion:** Plausible AND faithful (model relies on GATA for prediction)

**Case Study 2: GC Content Artifact**
- **Plausible Scenario:**
  - SAE feature activates on GC-rich sequences
  - JASPAR match: SP1 motif (GC-rich binding site, p < 0.01)
  - **BUT:** Ablation shows ΔAUROC = 0.00
  - GWAS: No enrichment
  - Negative control: Shuffled SAE also shows SP1 match
  - **Conclusion:** Plausible but spurious (training data artifact, not biological signal)

**Validation Protocol**
- **Rule 1:** Never claim biological interpretation without computational necessity
- **Rule 2:** Use multiple validation sources (JASPAR + GWAS + ClinVar)
- **Rule 3:** Always report negative controls (shuffled SAE results)
- **Rule 4:** Distinguish "model uses X" from "X is biologically plausible"

---

## 5. Risk Assessment and Mitigation

### 5.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation | Go/No-Go Trigger |
|------|-----------|--------|------------|------------------|
| **SAEs don't find interpretable features in DNA** | Medium | High | Negative result is publishable; focus on methodology differences vs. proteins | End of Week 10: If <10% features have JASPAR match, pivot to methods paper |
| **Features are plausible but not faithful** | Medium | Medium | Emphasize computational necessity validation; compare to baselines | End of Week 12: If <25 features show necessity, broaden to multi-model SAEs |
| **Compute resources insufficient** | Low | High | Prioritize top 5 layers; use Mayo HPC GPU allocation; request additional compute if needed | Week 4: If SAE training >40 hours per layer, reduce to top 3 layers |
| **GoodFire partnership doesn't materialize** | Medium | Low | Implement SAEs in-house using InterPLM code (open source) | Week 2: If no GoodFire response, proceed independently |
| **Real data access delayed** | Medium | Medium | Start with 1000 Genomes + synthetic; publish methods paper with promise of follow-up | Week 6: If no Helix access, use public data only |

### 5.2 Timeline Risks

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|-----------|--------|------------|-------------|
| **NeurIPS deadline too aggressive** | Medium | High | Front-load critical path (SAE training, feature annotation); defer aspirational goals | Target ICLR 2027 (Oct deadline) or ICML 2026 (Feb deadline) as backup |
| **Validation takes longer than expected** | Medium | Medium | Parallelize ablation experiments; use HPC job arrays | Reduce ablation scope to 50 features instead of 100 |
| **Paper writing takes 4+ weeks** | Medium | Low | Start Methods section during Phase 2; draft Introduction in parallel | Request deadline extension if NeurIPS allows |

### 5.3 Scientific Risks

| Risk | Likelihood | Impact | Mitigation | Pivot Strategy |
|------|-----------|--------|------------|----------------|
| **DNA SAEs show fundamentally different properties than protein SAEs** | High | Low (could be positive) | Frame as scientific contribution; "DNA vs. protein interpretability" | Emphasize novel findings about DNA representations |
| **No phenotype-specific features for IBD** | Medium | High | Expand to multi-phenotype (RA, T1D from gfm-platform) | Broader paper: "SAE interpretability for genomic FMs" |
| **Features don't align with known biology** | Low | High | Deep-dive validation; may discover novel regulatory patterns | If validated as faithful but unknown, publish as discovery |
| **Results scooped by concurrent work** | Low | High | Monitor arXiv/bioRxiv for DNA interpretability papers | Emphasize unique aspects (IBD, clinical validation, Mayo data) |

### 5.4 Go/No-Go Decision Points

**Decision Point 1 (End of Week 2): SAE Architecture and Hyperparameters**
- **Criteria:** Reconstruction R² ≥0.90 on test set, sparsity L0 ≤ 30
- **Go:** Proceed to full-scale training (Week 3-6)
- **No-Go:** If R² < 0.90, troubleshoot architecture (try gated SAEs, adjust hidden dim)

**Decision Point 2 (End of Week 6): Feature Quality**
- **Criteria:** ≥50% features show non-trivial activation (not GC-only), shuffle control shows random patterns
- **Go:** Proceed to biological annotation (Week 7-10)
- **No-Go:** If features dominated by artifacts, revisit training data (balance gene-rich vs. gene-desert) or SAE architecture

**Decision Point 3 (End of Week 10): Interpretability**
- **Criteria:** ≥30% features match JASPAR motifs (p < 0.01) OR ≥50 IBD-enriched features (FDR < 0.05)
- **Go:** Proceed to validation (Week 11-14)
- **No-Go:** If <10% JASPAR match AND no IBD enrichment, pivot to methods paper ("Challenges of DNA SAE interpretability") or multi-model approach

**Decision Point 4 (End of Week 12): Computational Necessity**
- **Criteria:** ≥25 features show ablation ΔAUROC ≥0.03 (p < 0.05)
- **Go:** Proceed to paper writing (Week 15-18)
- **No-Go:** If <10 necessary features, pivot to workshop paper or ICLR 2027 with extended validation

---

## 6. Resource Requirements

### 6.1 Compute Resources

**GPU Requirements**
- **SAE Training:** ~1,000 GPU-hours (A100 or equivalent)
  - Mayo HPC allocation: 10 A100 GPUs available
  - Timeline: 100 hours of wall-clock time (parallelized across 10 GPUs)
  - **Status:** Feasible within existing allocation

**CPU Requirements**
- **Feature Annotation:** ~670 CPU-hours (embarrassingly parallel)
  - JASPAR motif scanning: ~500 hours
  - ENCODE overlap: ~50 hours
  - Statistical analysis: ~120 hours
  - **Status:** Trivial on Mayo HPC cluster (100+ CPU nodes)

**Storage Requirements**
- **Embeddings:** ~500 GB (already cached in gfm-platform two-stage pipeline)
- **SAE Models:** ~160 GB (32 SAEs × 5 GB each)
- **Feature Activations:** ~50 GB (activation matrices)
- **Feature Catalog:** ~10 GB (annotations, validation results)
- **Total:** ~720 GB
- **Status:** Feasible within Mayo storage allocation (~5 TB available)

### 6.2 Data Access

| Dataset | Status | Timeline | Use Case |
|---------|--------|----------|----------|
| **Helix 100k (IBD cohort)** | Available | Now | Primary training/validation data |
| **Regeneron 80k** | Pending DUA | Target: Feb 2026 | External validation (if available) |
| **1000 Genomes** | Public | Now | Prototyping, negative controls |
| **Synthetic IBD Cohort** | Available (gfm-platform) | Now | Early method development |
| **UKBioBank 500k** | Application pending | Unlikely by May | Future work |
| **GWAS Catalog (IBD)** | Public | Now | Biological validation |
| **ClinVar** | Public | Now | Pathogenicity prediction |
| **JASPAR** | Public | Now | Motif annotation |
| **ENCODE cCREs** | Public | Now | Regulatory element overlap |

**Risk Mitigation:**
- If Regeneron delayed: Use TOPMed or gnomAD for external validation
- If Helix access issues: Proceed with 1000 Genomes + synthetic (sufficient for methods paper)

### 6.3 External Collaborations

**GoodFire AI Partnership**
- **Status:** Planned but not started (per gfm-platform roadmap Week 23-26)
- **Potential Contributions:**
  - SAE training code (if proprietary methods)
  - Interpretability best practices from LLM work
  - Co-authorship discussions
- **Timeline:** Engage by Week 2 (Jan 22)
- **Contingency:** If no response, proceed with InterPLM open-source code

**Mayo Clinic Internal**
- **Eric Klee Lab (Genomics):** Clinical context, IBD domain expertise
- **Carl Molnar (ML):** ML methodology review, compute resources
- **Panos Korfiatis (CSO):** Strategic guidance, Mayo publication support

**External (Aspirational)**
- **Stanford (Zou Lab - InterPLM authors):** Methodology consultation, cross-validation
- **Broad Institute (Avanti Shrikumar - DeepLIFT):** Interpretability methodology

### 6.4 Software and Tools

**Existing Infrastructure (gfm-platform)**
- Two-stage embedding pipeline (ADR-002): Embeddings already cached
- Layer hunting implementation (ADR-003): Identify top layers
- Nucleotide Transformer adapter: Model loading, inference
- Evaluation harness: AUROC, calibration metrics
- W&B integration: Experiment tracking

**New Development Required**
- SAE training pipeline (~1 week, Week 1-2)
- Feature activation extraction (~3 days, Week 6)
- Biological annotation pipeline (~1 week, Week 7)
- Ablation experiment framework (~1 week, Week 11)

**External Tools**
- FIMO (JASPAR motif scanning): Already installed
- BEDTools (genomic overlap): Already installed
- InterPLM code (SAE reference implementation): Open source, needs adaptation

---

## 7. Milestones and Success Criteria

### 7.1 Phase Milestones

| Phase | End Date | Milestone | Success Criteria | Deliverables |
|-------|----------|-----------|------------------|--------------|
| **Phase 1** | Feb 26 | SAE Training Complete | ✓ 32 SAEs trained (R² ≥0.95, L0 ≤20)<br>✓ Feature activations extracted | - SAE models (32)<br>- Feature activation matrices<br>- Training report |
| **Phase 2** | Mar 26 | Feature Characterization | ✓ ≥30% features with JASPAR match<br>✓ ≥50 IBD-enriched features<br>✓ Baseline comparisons complete | - Feature catalog (annotated)<br>- IBD feature report<br>- Comparison tables |
| **Phase 3** | Apr 23 | Validation Complete | ✓ ≥25 features with necessity (ΔAUROC ≥0.03)<br>✓ GWAS enrichment (OR ≥3)<br>✓ ClinVar prediction (AUROC ≥0.65) | - Ablation results<br>- Biological validation report<br>- Final feature set |
| **Phase 4** | May 22 | Paper Submitted | ✓ 8-9 page manuscript<br>✓ 6-8 main figures<br>✓ Code released<br>✓ NeurIPS submission | - Manuscript<br>- Supplementary materials<br>- GitHub repo |

### 7.2 Publication Targets

**Primary Target: NeurIPS 2026 Main Conference**
- **Deadline:** Early May 2026 (typical NeurIPS deadline)
- **Acceptance Rate:** ~25%
- **Fit:** Mechanistic interpretability is core NeurIPS topic; genomics application is novel
- **Requirements:**
  - Technical novelty: SAEs for DNA (first application)
  - Empirical rigor: Multi-source validation, necessity/sufficiency
  - Reproducibility: Code/data release, clear methodology
  - Impact: Clinical utility (ACMG-AMP evidence), biological insights

**Backup Target 1: NeurIPS 2026 Workshop (Computational Biology)**
- **Deadline:** Typically Sep 2026 (after main conference decisions)
- **Acceptance Rate:** ~50%
- **Fit:** If main conference rejects due to "insufficient novelty" or "needs more validation"
- **Strategy:** Revise based on reviews, emphasize domain-specific contributions

**Backup Target 2: ICLR 2027**
- **Deadline:** Oct 2026
- **Acceptance Rate:** ~30%
- **Fit:** If NeurIPS rejects, use extra time for extended validation (sufficiency experiments, multi-model SAEs)
- **Strategy:** Incorporate reviewer feedback, add Evo-2 or DNABERT-2 results

**Aspirational Target: Nature Methods (Follow-up)**
- **Timeline:** Submit Dec 2026 - Jan 2027 (after NeurIPS results)
- **Requirements:**
  - Experimental validation (CRISPR, reporter assays): 6-12 months
  - Cross-model universality (NT + Evo-2 + DNABERT-2)
  - Clinical validation: Prospective IBD case prediction
  - Public resource: SAE catalog for community use
- **Strategy:** Position NeurIPS as methods foundation, Nature Methods as biological discovery

### 7.3 Code and Data Release

**Code Release (GitHub)**
- **Repository:** `gfm-interpretability-sae` (new repo, linked from gfm-platform)
- **Contents:**
  - SAE training code (PyTorch, documented)
  - Feature annotation pipeline (JASPAR, ENCODE integration)
  - Ablation experiment framework
  - Notebooks reproducing paper figures
  - Pre-trained SAE models (32 models, ~160 GB)
- **License:** Apache 2.0 (Mayo-approved open source)
- **Timing:** Public release upon NeurIPS acceptance (Aug 2026)

**Data Release**
- **Feature Catalog:** CSV with all features, annotations, validation results (public)
- **Embeddings:** Cannot release (PHI concerns), but provide code to generate from public data (1000 Genomes)
- **SAE Models:** Weights released (trained on Helix 100k, but models themselves are not PHI)
- **Reproducibility:** Full pipeline runnable on 1000 Genomes data (public)

---

## 8. What Could Make This Unpublishable?

### 8.1 Showstopper Scenarios

**Scenario 1: Features Are Not Interpretable**
- **What Happens:** <10% of features match JASPAR motifs, ENCODE overlap is random, no IBD enrichment
- **Why This Kills Paper:** Core claim is "SAEs discover interpretable features" - if features are not interpretable, no contribution
- **Likelihood:** Low (InterPLM worked for proteins; DNA should have some interpretable patterns)
- **Pivot:** Methods paper - "Why DNA SAEs Are Harder Than Protein SAEs" (still publishable at workshop level)

**Scenario 2: Features Are Plausible But Not Faithful**
- **What Happens:** Features match JASPAR (plausible) but show zero ablation effect (not faithful)
- **Why This Problematic:** Reviewers will ask "does the model actually use these features?" - if answer is no, not a scientific contribution
- **Likelihood:** Medium (saturation, redundancy, or distributed encoding could cause this)
- **Pivot:** Focus on subset of faithful features; frame as "some features are faithful, others are spurious - here's how to distinguish"

**Scenario 3: Insufficient Novelty vs. InterPLM**
- **What Happens:** Reviewers say "this is just InterPLM applied to DNA, no new methodology"
- **Why This Kills Paper:** NeurIPS requires methodological contribution, not just application
- **Likelihood:** Medium (main risk for main conference acceptance)
- **Pivot:** Emphasize DNA-specific challenges (longer context, weaker ground truth, multi-gene aggregation), add multi-model analysis (NT + Evo-2)

**Scenario 4: Real Data Access Blocked**
- **What Happens:** Helix 100k access delayed due to PHI/IRB issues
- **Why This Problematic:** Synthetic data is not convincing for biological claims
- **Likelihood:** Low (data already available in gfm-platform)
- **Pivot:** Use 1000 Genomes + public GWAS, publish methods paper with promise of follow-up on real cohorts

**Scenario 5: Concurrent Work Scoops Results**
- **What Happens:** Another group publishes DNA SAEs before NeurIPS submission
- **Why This Kills Paper:** Novelty gone
- **Likelihood:** Low (monitoring arXiv/bioRxiv shows no current DNA SAE work)
- **Pivot:** Emphasize unique aspects (IBD phenotype, clinical validation, Mayo data scale, multi-layer analysis)

### 8.2 Reviewer Concerns and Preemptive Responses

**Concern 1: "This is just engineering, not science"**
- **Response:** Distinguish publishable findings (H1-H5 about DNA representations) from engineering work (layer hunting, attention viz)
- **Evidence:** Novel claims about layer specialization, superposition in DNA vs. proteins, phenotype-specific circuits

**Concern 2: "Validation is weak - no experimental ground truth"**
- **Response:** Multi-source computational validation (JASPAR + GWAS + ClinVar + ablation) is standard in interpretability field
- **Evidence:** InterPLM used similar approach; experimental validation is future work (already proposed in Discussion)

**Concern 3: "Results are specific to IBD and Nucleotide Transformer"**
- **Response:** IBD is proof-of-concept; methodology generalizes
- **Evidence:** If time permits, add one additional phenotype (RA) and/or one additional model (DNABERT-2) in supplementary

**Concern 4: "Interpretability methods are known to be unreliable"**
- **Response:** This is precisely why we emphasize faithful (necessity) over plausible (motif matches)
- **Evidence:** Ablation experiments, negative controls (shuffled SAE), comparison to baselines

### 8.3 Mitigation Strategies

**Strategy 1: Front-Load Critical Path**
- Prioritize SAE training (Week 1-6) and feature annotation (Week 7-10)
- Early go/no-go decisions (Week 2, 6, 10, 12)
- Contingency time built into Phase 4 (paper writing)

**Strategy 2: Build in Scientific Optionality**
- Core contribution: SAEs for DNA (must work)
- Optional extensions: Multi-model (NT + Evo-2), multi-phenotype (IBD + RA), sufficiency experiments
- If timeline tight, cut optional extensions and focus on core

**Strategy 3: Over-Validate Core Claims**
- Never claim interpretability without necessity
- Always report negative controls
- Multiple validation sources for every feature
- Statistical rigor (bootstrap CIs, FDR correction)

**Strategy 4: Transparent Limitations**
- Paper Discussion explicitly addresses: synthetic vs. real data, single phenotype, computational vs. experimental validation
- Frame as "first step" in DNA interpretability, not comprehensive solution
- Propose clear future work (experimental validation, multi-model alignment, clinical deployment)

---

## 9. Conclusion and Next Steps

### 9.1 Summary

This research plan outlines a 4.5-month sprint (Jan 15 - May 22, 2026) to develop and validate sparse autoencoders for DNA foundation model interpretability, targeting NeurIPS 2026 submission. The project extends SAE methodology from proteins (InterPLM) to DNA, with specific application to IBD phenotype prediction using Mayo Clinic's Helix 100k cohort and existing gfm-platform infrastructure.

**Core Hypothesis:** SAEs will discover interpretable regulatory features in Nucleotide Transformer representations that correspond to IBD disease mechanisms (NOD2, IL23R, ATG16L1), and these features will be computationally necessary for phenotype prediction.

**Success Criteria:**
- **Minimum (Workshop):** 50+ interpretable features, 10+ necessary features
- **Target (Main Conference):** 100+ interpretable features, 25+ necessary features, biological validation (GWAS, ClinVar)
- **Aspirational (Nature Methods):** Cross-model universality, experimental validation, clinical deployment

**Feasibility:** Project is technically feasible within timeline (compute available, data accessible, methods validated in protein domain) but scientifically risky (DNA may not yield interpretable features, or features may be plausible but not faithful).

### 9.2 Immediate Next Steps (Week 1: Jan 15-22)

**Day 1-2 (Jan 15-16): Project Kickoff**
1. Create project repository structure in gfm-discovery: `/docs/planning/projects/sae-dna-neurips2026/`
2. Set up tracking: Create GitHub project board with milestones from Section 7.1
3. Contact GoodFire AI: Initiate partnership discussion (code access, collaboration model)
4. Review InterPLM code: Clone repo, understand SAE architecture and training procedure

**Day 3-5 (Jan 17-19): Environment Setup**
5. Set up SAE training environment: PyTorch, HuggingFace, gfm-platform integration
6. Extract IBD embeddings: Use gfm-platform two-stage pipeline to cache NT embeddings for 15 genes × 2000 patients
7. Run layer hunting: Identify top 5 layers for IBD phenotype prediction
8. Baseline interpretability: Run ISM, IG, Attention on 100 test sequences (ADR-034/035/039 implementations)

**Day 6-7 (Jan 20-21): SAE Architecture Testing**
9. Implement SAE training code: Adapt InterPLM or use GoodFire SDK (depending on partnership status)
10. Small-scale SAE training: Train on 1000 sequences, one layer, test reconstruction R² and sparsity L0
11. Hyperparameter sweep design: Define grid for Week 3-4 (sparsity λ, hidden dim, architecture variants)
12. Go/No-Go Decision 1 (Jan 21): If R² < 0.90, troubleshoot architecture; if R² ≥ 0.90, proceed to Week 2

### 9.3 Open Questions for Resolution (Week 1)

1. **GoodFire Partnership:** What is current status? Code access? Co-authorship model?
2. **Helix Data Access:** Confirm IRB approval for interpretability research use case
3. **Compute Allocation:** Verify Mayo HPC GPU availability (10 A100s for 100 hours in Jan-Feb)
4. **InterPLM Code:** License compatible with Mayo usage? Need to adapt vs. use as-is?
5. **NeurIPS Deadline:** Confirm exact deadline (typically early May, but verify 2026 date)

### 9.4 Risk Monitoring

**Weekly Risk Review:**
- Track progress against milestones (Table in Section 7.1)
- Evaluate go/no-go criteria at decision points (Section 5.4)
- Monitor arXiv/bioRxiv for concurrent DNA interpretability work
- Adjust timeline if critical path delays (front-load core work, defer optional extensions)

**Pivot Triggers:**
- If Week 6 feature quality poor: Pivot to methods paper
- If Week 10 interpretability low: Add multi-model analysis (NT + Evo-2)
- If Week 12 necessity weak: Target workshop vs. main conference
- If Week 16 timeline tight: Request ICLR 2027 extension

### 9.5 Long-Term Vision (Beyond NeurIPS)

**Phase 2 (Post-NeurIPS, Summer 2026):**
- Experimental validation: CRISPR, reporter assays (6-12 months, collaboration with Eric Klee lab)
- Cross-model analysis: Evo-2, DNABERT-2 SAEs for feature universality
- Multi-phenotype: RA, T1D, prostate cancer (expand from IBD)

**Phase 3 (Nature Methods, 2027):**
- Public SAE catalog: Community resource (like InterPLM for proteins)
- Clinical integration: ACMG-AMP variant interpretation pipeline with SAE evidence
- Prospective validation: Predict held-out IBD cases from Mayo Biobank

**PhD Thesis Integration (Sep 2026 onwards):**
- NeurIPS paper becomes Chapter 2: Interpretability methodology
- PhD Aim 1: Network-aware + ancestry-robust phenotype prediction with SAE-based variant attribution
- Differentiator: Interpretable, fair, prospectively validated genomic foundation models

---

## Appendix A: Project File Structure

```
gfm-discovery/
├── docs/
│   └── planning/
│       └── projects/
│           └── sae-dna-neurips2026/
│               ├── research-plan.md (this document)
│               ├── weekly-progress-log.md
│               ├── go-no-go-decisions.md
│               ├── literature-review.md (InterPLM, SAE methods)
│               └── neurips-submission/
│                   ├── manuscript-draft.md
│                   ├── figures/
│                   ├── supplementary/
│                   └── reviews-responses/ (if needed)
│
├── 08_sae_interpretability/  (NEW)
│   ├── 01_sae_training/
│   │   ├── train_sae.py
│   │   ├── config/ (hyperparameter YAMLs)
│   │   └── models/ (saved SAE checkpoints)
│   ├── 02_feature_extraction/
│   │   ├── extract_activations.py
│   │   └── feature_activation_matrices/ (HDF5)
│   ├── 03_biological_annotation/
│   │   ├── jaspar_scanning.py
│   │   ├── encode_overlap.py
│   │   ├── clinvar_enrichment.py
│   │   └── feature_catalog.csv (annotated features)
│   ├── 04_validation/
│   │   ├── ablation_experiments.py
│   │   ├── sufficiency_experiments.py
│   │   ├── gwas_enrichment.R
│   │   └── results/ (ablation tables, enrichment stats)
│   ├── 05_analysis_notebooks/
│   │   ├── 01_sae_training_qc.ipynb
│   │   ├── 02_feature_catalog_overview.ipynb
│   │   ├── 03_ibd_feature_analysis.ipynb
│   │   ├── 04_layer_specialization.ipynb
│   │   ├── 05_ablation_results.ipynb
│   │   └── 06_biological_validation.ipynb
│   └── 06_paper_figures/
│       ├── fig1_sae_overview.py
│       ├── fig2_feature_catalog.py
│       ├── fig3_layer_specialization.py
│       ├── fig4_ibd_circuits.py
│       ├── fig5_ablation_results.py
│       └── fig6_biological_validation.py
```

---

## Appendix B: Key References

**SAE Methodology**
- InterPLM (Simon & Zou, Nature Methods 2025): Protein SAE reference
- Anthropic SAE work (LLM interpretability): Gated SAEs, training best practices
- GoodFire AI blog posts: Practical SAE training tips

**DNA Interpretability**
- gfm-book Chapter 25: Plausible vs. faithful, attribution methods, validation
- ADR-034 (ISM), ADR-035 (IG), ADR-039 (Attention): Platform baselines
- DeepLIFT, Integrated Gradients papers: Gradient-based attribution

**IBD Biology**
- NOD2 LRR variants: Ogura et al. (Nature 2001)
- IL23R R381Q protective: Duerr et al. (Science 2006)
- ATG16L1 T300A: Hampe et al. (Nat Genet 2007)
- IBD GWAS: International IBD Genetics Consortium (Nature 2012, 2017)

**Clinical Validation**
- ACMG-AMP Variant Interpretation: Richards et al. (Genet Med 2015)
- PP3/BP4 Computational Evidence: ClinGen guidelines
- gfm-book Chapter 29: Rare disease diagnostic workflows

---

**Document Version:** 1.0
**Last Updated:** 2026-01-15
**Next Review:** 2026-01-22 (end of Week 1)
**Owner:** Josh Meehl (meehl.joshua@mayo.edu)
