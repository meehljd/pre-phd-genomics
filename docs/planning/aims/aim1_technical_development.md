# Aim 1: Technical Development & Retrospective Validation
## Gene-Scale Interpretable AI for Rare Disease Variant Interpretation

**Timeline:** Months 1-18 (Sep 2026 - May 2027)  
**Phase:** Model development and retrospective validation  
**Output:** Paper 1 - Technical methodology (target: *Nature Genetics* / *Genome Medicine*, 2027)

---

## I. OBJECTIVE

**Develop and retrospectively validate an interpretable AI agent system for rare disease variant interpretation that integrates gene-scale foundation models, isoform-specific phenotype prediction, and multi-omics data to improve diagnostic accuracy on Mayo Clinic retrospective cohorts.**

---

## II. RATIONALE

### Current Limitations in Rare Disease Diagnosis

**Standard clinical workflow:**
1. Patient undergoes whole exome/genome sequencing (WES/WGS)
2. Genetic counselor + geneticist manually review variants
3. Filter by population frequency, in silico predictions, phenotype match
4. Interpret ~20-100 candidate variants per patient
5. Apply ACMG guidelines to classify pathogenicity
6. **Result:** 25-35% diagnostic yield, 4-6 weeks turnaround time

**Bottlenecks:**
- **Variant overload:** WES generates 20,000+ variants per patient
- **Phenotype matching:** Manual HPO term → gene mapping is time-consuming
- **Isoform complexity:** Most tools ignore tissue-specific isoform expression
- **Interpretability gap:** AI models lack biological explanations
- **Training data bias:** Most models trained predominantly on European ancestry populations

### Existing AI Methods (Competitors)

**Recent publications (2024-2025):**

1. **DeepRare** (Jun 2025) - multi-agent rare disease system
   - Strengths: Multi-agent architecture, phenotype integration
   - Limitations: Black box (no interpretability), retrospective only

2. **AlphaGenome** (Jun 2025) - regulatory variant foundation model
   - Strengths: Regulatory variant prediction
   - Limitations: Coding variants less accurate, no phenotype integration

3. **GenoMAS** (Jul 2025) - multi-agent genomic analysis
   - Strengths: Agent orchestration, literature search
   - Limitations: No foundation model backbone, limited ancestry diversity

4. **AI-MARRVEL** (2024) - variant prioritization with model organisms
   - Strengths: Functional evidence integration
   - Limitations: Requires model organism data (not always available)

**Gap in literature:**
- **No prospectively validated systems**
- **Limited interpretability** (most are black boxes)
- **Poor ancestry representation** in training data
- **No isoform-level resolution** (most operate at gene level only)

### Innovation of This Approach

**Key differentiators:**

1. **Gene-scale interpretability**
   - Visible neural networks (GenNet architecture) at gene level
   - Attention mechanisms show which genomic regions drive predictions
   - Counterfactual explanations reveal causal relationships

2. **Isoform-specific phenotype prediction**
   - ESM1b (protein foundation model) + IMPPROVE (variant effect prediction)
   - Tissue-specific isoform expression integration
   - Predict phenotypic consequences at isoform resolution

3. **Multi-ancestry training**
   - Explicit focus on diverse populations (not just European)
   - Population stratification correction in training
   - Fairness metrics reported for each ancestry group

4. **Multi-omics integration**
   - Genomics (WES/WGS variants)
   - Transcriptomics (RNA-seq for isoform expression)
   - Phenomics (HPO terms)
   - Network topology (protein-protein interactions)

5. **Clinical workflow integration**
   - Microsoft Healthcare Agent Orchestrator (HAO) for multi-agent coordination
   - NVIDIA BioNeMo for efficient model training/deployment
   - Sub-48-hour turnaround time (vs 4-6 weeks standard)

---

## III. SYSTEM ARCHITECTURE

### High-Level Overview

**AI System Name:** GenoInsight (tentative - can change)

**Three-tier architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER 1: Data Integration                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  WES/WGS VCF │  │  RNA-seq     │  │  HPO Terms   │      │
│  │  + Annotation│  │  Expression  │  │  (Phenotype) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│          │                  │                  │             │
│          └──────────────────┴──────────────────┘             │
│                           │                                   │
└───────────────────────────┼───────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              TIER 2: Foundation Model Layer                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  A. Genomic Foundation Model (TBD - Jan 2026)        │  │
│  │     - Candidates: Enformer, NT, Hyena-DNA, DNABERT-2│  │
│  │     - Input: Genomic sequence context (~10kb)        │  │
│  │     - Output: Variant pathogenicity embeddings       │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  B. Protein Foundation Model (ESM2)                  │  │
│  │     - Input: Protein sequence + variant              │  │
│  │     - Output: Variant effect prediction              │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  C. Gene-Scale Interpretable Network (GenNet)        │  │
│  │     - Visible neural network with gene-level nodes   │  │
│  │     - Attention mechanism over genomic features      │  │
│  │     - Output: Gene-level pathogenicity scores        │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           TIER 3: Phenotype & Network Integration            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  D. Isoform-Specific Phenotype Prediction           │  │
│  │     - ESM1b + IMPPROVE for variant effect           │  │
│  │     - Tissue-specific isoform expression            │  │
│  │     - HPO term → isoform mapping                     │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  E. Network Medicine Integration                     │  │
│  │     - Protein-protein interaction networks           │  │
│  │     - Disease module identification                  │  │
│  │     - Network topology → phenotype breadth           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  F. Multi-Agent Orchestration (HAO)                  │  │
│  │     - Literature search agent                        │  │
│  │     - ACMG classification agent                      │  │
│  │     - Evidence aggregation agent                     │  │
│  │     - Report generation agent                        │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT: Clinical Report                   │
│  • Ranked candidate genes (top 5-10)                        │
│  • Risk scores (0-100) per gene                             │
│  • Variant-level ACMG classification                        │
│  • Interpretability features (attention, counterfactuals)   │
│  • Evidence summary (papers, functional studies)            │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

#### A. Genomic Foundation Model (Decision Point: Jan 2026)

**Purpose:** Encode genomic sequence context around variants

**Model selection process:**
- Evaluate 4 candidates (see genomic_model_selection_framework.md)
- Criteria: Performance (30%), Interpretability (25%), Usability (20%), Context length (15%), Support (10%)
- Decision timeline: 6-8 hours evaluation in Jan 2026

**Top candidates:**
1. **Enformer** (DeepMind/Google)
   - Pros: Strong regulatory prediction, well-validated
   - Cons: 200kb context limit, attention may not expose well
   
2. **Nucleotide Transformer** (InstaDeep)
   - Pros: 1000bp context, transformer architecture (attention visualization)
   - Cons: Less validated for variant interpretation
   
3. **Hyena-DNA** (Stanford)
   - Pros: 1M bp context (massive), efficient
   - Cons: Novel architecture (less interpretable?)
   
4. **DNABERT-2** (Zhihan Zhou lab)
   - Pros: BERT-style, good fine-tuning, manageable context
   - Cons: Limited context (512bp)

**Integration approach:**
- Freeze foundation model weights (no retraining on Mayo data - too expensive)
- Fine-tune small adapter layers on Mayo retrospective cohorts
- Extract embeddings for variant neighborhoods (±5kb)
- Feed embeddings to downstream interpretable network

#### B. Protein Foundation Model (ESM2)

**Fixed choice:** ESM2 (Meta AI) - 650M parameter model

**Rationale:**
- Pre-trained on 250M protein sequences
- Strong performance on variant effect prediction
- Well-documented attention extraction
- Used successfully in pre-PhD Track A work

**Integration:**
- Input: Wildtype protein sequence + mutant sequence
- Compute: ΔEmbedding = Embed(mutant) - Embed(wildtype)
- Predict: Pathogenicity score from ΔEmbedding
- Attention: Visualize which residues are most affected

**Isoform handling:**
- Run ESM2 on each annotated isoform separately
- Weight by tissue-specific expression (GTEx data)
- Aggregate: Isoform scores → gene-level score

#### C. Gene-Scale Interpretable Network (GenNet)

**Architecture:** Visible neural network with biological hierarchy

**Inspiration:** GenNet (Schubach et al., 2022) + attention mechanisms

**Layer structure:**
```
Input Layer (Variant Features)
    ↓
Gene Layer (20,000 nodes = 20,000 genes)
    ↓ [Gene-level attention]
Pathway Layer (1,000 nodes = pathways/networks)
    ↓ [Pathway-level attention]
Disease Category Layer (10 nodes = neurological, metabolic, etc.)
    ↓
Output Layer (Binary: Pathogenic / Benign)
```

**Key feature:** Each node = biological entity (gene, pathway)
- Weights can be inspected (which genes contribute to prediction?)
- Attention scores show importance of each gene
- Biologically grounded (not black box)

**Training:**
- Supervised learning on ClinVar pathogenic/benign variants
- Regularization to enforce biological structure (pathway coherence)
- Multi-ancestry training data (see below)

#### D. Isoform-Specific Phenotype Prediction

**Problem:** Most tools predict gene-level effects, but isoforms vary by tissue

**Approach:**
1. **Variant effect prediction at isoform level:**
   - ESM1b (smaller, faster ESM model) for protein embedding
   - IMPPROVE framework (Rao et al., 2023) for variant impact
   - Predict: Which isoform(s) are affected by variant?

2. **Tissue-specific expression integration:**
   - GTEx v8 data: Isoform expression by tissue
   - Weight variant effect by expression in disease-relevant tissues
   - Example: Neurological phenotype → prioritize brain-expressed isoforms

3. **HPO term → isoform mapping:**
   - HPOlib (phenotype ontology)
   - Map each HPO term to relevant tissues
   - Score: How well does affected isoform explain patient's HPO terms?

**Formula:**
```
Phenotype Match Score = Σ (HPO_weight_i × Isoform_expression_tissue × Variant_effect_isoform)
```

**Example:**
- Patient has HPO term "Intellectual disability" (brain-related)
- Variant affects Gene A, Isoform 2
- Isoform 2 highly expressed in brain (GTEx)
- ESM1b predicts Isoform 2 function disrupted
- **High phenotype match score** → Gene A prioritized

#### E. Network Medicine Integration

**Hypothesis:** Disease genes cluster in networks; topology predicts phenotype breadth

**Data sources:**
1. **Protein-protein interactions (PPI):**
   - STRING database (high-confidence interactions)
   - BioGRID (experimental interactions)
   - Combine: Weighted PPI network

2. **Disease modules:**
   - DIAMOnD algorithm (Ghiassian et al., 2015) to identify disease neighborhoods
   - Seeds: Known disease genes for patient's phenotype
   - Expand: Neighbors in PPI network

3. **Network topology features:**
   - Degree centrality (hub genes → pleiotropic effects)
   - Betweenness centrality (connector genes → multi-system disease)
   - Clustering coefficient (local network density)

**Integration into scoring:**
- Candidate gene's network proximity to known disease genes
- If gene is hub → expect broad phenotype
- If gene is peripheral → expect narrow phenotype
- Match observed phenotype breadth to predicted breadth

#### F. Multi-Agent Orchestration (Microsoft HAO)

**Purpose:** Coordinate multiple AI agents for comprehensive diagnostic workup

**Agents:**

1. **Literature Search Agent:**
   - Query PubMed for gene-disease associations
   - Extract relevant case reports, functional studies
   - Summarize evidence strength

2. **ACMG Classification Agent:**
   - Apply ACMG/AMP 2015 guidelines programmatically
   - Assign evidence codes (PVS1, PS1-4, PM1-6, PP1-5)
   - Aggregate: Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign

3. **Population Frequency Agent:**
   - Query gnomAD, TOPMed, UK Biobank
   - Check allele frequency in disease-relevant populations
   - Flag if too common for rare disease

4. **Segregation Analysis Agent:**
   - Check if family data available
   - Evaluate co-segregation with disease
   - Assign PP1 or BS4 evidence codes

5. **Functional Evidence Agent:**
   - Query ClinVar, HGMD for known pathogenic variants
   - In silico predictions (REVEL, CADD, AlphaMissense)
   - Aggregate functional evidence

6. **Evidence Aggregation Agent:**
   - Combine outputs from all agents
   - Weight evidence (literature > in silico predictions)
   - Generate final rank-ordered candidate list

7. **Report Generation Agent:**
   - Create human-readable clinical report
   - Include interpretability visualizations
   - Format for clinical geneticist review

**Orchestration logic (HAO framework):**
```python
def diagnostic_pipeline(patient_vcf, hpo_terms):
    # Stage 1: Parallel variant analysis
    genomic_scores = genomic_foundation_model(patient_vcf)
    protein_scores = esm2_variant_effect(patient_vcf)
    gene_scores = gennet_prediction(patient_vcf)
    
    # Stage 2: Phenotype matching
    phenotype_scores = isoform_phenotype_match(patient_vcf, hpo_terms)
    network_scores = network_proximity(patient_vcf, hpo_terms)
    
    # Stage 3: Evidence gathering (parallel agents)
    literature = literature_agent.search(candidate_genes)
    acmg = acmg_agent.classify(candidate_variants)
    frequency = frequency_agent.check(candidate_variants)
    
    # Stage 4: Aggregation
    final_scores = aggregate(
        genomic_scores, protein_scores, gene_scores,
        phenotype_scores, network_scores,
        literature, acmg, frequency
    )
    
    # Stage 5: Interpretability
    attention = extract_attention(gennet, genomic_model)
    counterfactuals = generate_counterfactuals(top_candidates)
    
    # Stage 6: Report
    report = report_agent.generate(
        ranked_genes=final_scores,
        evidence=literature,
        interpretability=(attention, counterfactuals)
    )
    
    return report
```

---

## IV. TRAINING DATA

### Retrospective Cohorts

**Primary source:** Mayo Clinic rare disease program (Eric Klee's lab)

**Datasets:**

1. **Solved cases (training set):**
   - Patients with confirmed molecular diagnosis (ClinVar P/LP variant)
   - Time period: 2015-2023 (before trial launch)
   - Expected N: 2,000-3,000 cases
   - Data: WES/WGS VCF, HPO terms, causal variant, diagnosis

2. **Unsolved cases (negative examples):**
   - Patients without molecular diagnosis after extensive workup
   - Used to train model on "no pathogenic variant" class
   - Expected N: 3,000-5,000 cases

3. **External validation sets:**
   - **Baylor Genetics:** Clinical lab cases (request collaboration)
   - **Broad Institute:** Rare disease cohorts (public datasets)
   - **GeneDx:** Commercial lab cases (if data sharing agreement possible)

**Data split:**
- Training: 70% of Mayo cases (randomly selected)
- Validation: 15% (hyperparameter tuning)
- Test: 15% (held-out, final evaluation)
- External: Baylor/Broad cohorts (completely independent)

### Multi-Ancestry Training Strategy

**Problem:** Most genomic AI models trained predominantly on European ancestry
- Leads to worse performance in underrepresented populations
- Violates health equity principles

**Approach:**

1. **Ancestry composition in training data:**
   - Measure: Genetic ancestry PCA on training set
   - Target: ≥20% non-European ancestry (if available at Mayo)
   - Oversampling strategy: If <20%, oversample non-European cases

2. **Ancestry-stratified performance:**
   - Report model performance separately for each ancestry group
   - Ensure no group has AUC >0.1 below European performance
   - If disparity detected → investigate and mitigate

3. **Population stratification correction:**
   - Include genetic ancestry principal components (PCs) as covariates
   - Prevents model from learning ancestry-correlated confounders
   - Forces model to learn biology, not population structure

4. **Ancestry-specific calibration:**
   - Calibrate risk scores separately per ancestry group
   - Ensures 70% risk score → 70% true pathogenic rate in all groups

**Fairness metrics (to be reported in Paper 1):**
- Sensitivity by ancestry
- Specificity by ancestry
- PPV by ancestry
- Calibration curves by ancestry
- Disparate impact ratio (min/max AUC across groups)

---

## V. INTERPRETABILITY METHODS

### Why Interpretability Matters

**Clinical requirements:**
- Geneticists need to understand *why* AI flagged a gene
- Regulatory agencies (FDA) may require explainability
- Trust: Black box predictions are hard to trust
- Scientific value: Interpretability reveals biological insights

**Competitor gap:** Most AI systems are black boxes (DeepRare, AlphaGenome)

### Method 1: Attention Visualization

**For transformer-based models (genomic foundation model, ESM2):**

**Process:**
1. Extract attention weights from each layer
2. Aggregate: Attention rollout (sum across layers)
3. Map: Attention scores to genomic positions

**Visualization:**
- Heatmap showing which genomic regions were most attended to
- Example: If variant in exon gets high attention → model focusing on coding impact
- Example: If variant in promoter gets high attention → model considering regulatory impact

**Clinical utility:**
- Geneticist can see if model is looking at biologically relevant regions
- Red flag if model attends to non-coding regions for coding variant (possible error)

### Method 2: SHAP (SHapley Additive exPlanations)

**Purpose:** Quantify contribution of each feature to prediction

**Features to explain:**
- Variant type (missense, nonsense, frameshift, splice, etc.)
- Population frequency (gnomAD AF)
- In silico scores (CADD, REVEL, AlphaMissense)
- Conservation (PhyloP, GERP)
- Gene constraint (pLI, LOEUF)
- Phenotype match score
- Network proximity score

**Output:**
- SHAP values: Positive = pushes toward pathogenic, Negative = pushes toward benign
- Rank features by |SHAP value|
- Report top 5 features driving prediction

**Example interpretation:**
```
Variant: Gene A p.Arg123Ter (nonsense)
SHAP values:
  +0.8: Variant type (nonsense)
  +0.5: Gene constraint (high pLI = loss-of-function intolerant)
  +0.3: Phenotype match (HPO terms align with Gene A)
  -0.2: Population frequency (absent in gnomAD)
  +0.1: Conservation (moderately conserved)
  
→ Model prediction: Pathogenic (driven by nonsense + pLI + phenotype match)
```

### Method 3: Counterfactual Explanations

**Inspiration:** MrVI approach (variational inference for mechanistic explanations)

**Question:** What would need to change about this variant for the model to predict differently?

**Process:**
1. Take predicted pathogenic variant
2. Perturb features one at a time (in silico)
3. Re-run model to see if prediction flips
4. Identify minimal changes that flip prediction

**Example:**
```
Original: Gene A missense variant, CADD=25, gnomAD AF=0, pLI=0.99
Prediction: Pathogenic (score 85)

Counterfactual 1: If CADD=15 (instead of 25)
→ Prediction: VUS (score 55) [FLIPPED]
Interpretation: CADD score is critical for this prediction

Counterfactual 2: If gnomAD AF=0.001 (instead of 0)
→ Prediction: Likely benign (score 35) [FLIPPED]
Interpretation: Even rare presence in gnomAD strongly argues against pathogenicity
```

**Clinical utility:**
- Reveals which features are most critical
- Helps geneticist understand model reasoning
- Identifies brittle predictions (flip with small changes → low confidence)

### Method 4: Gene-Level Attribution (GenNet)

**Unique to visible neural network architecture:**

**Process:**
1. Trace prediction through network layers
2. Identify which gene nodes had highest activation
3. Identify which pathway nodes connected to those genes

**Visualization:**
- Network graph: Nodes sized by activation strength
- Edges colored by weight strength
- Highlight path from input variant → gene node → pathway → output

**Example:**
```
Variant in Gene A (exon 5)
  ↓ [high activation]
Gene A node (layer 1)
  ↓ [strong edge weight]
"DNA repair pathway" node (layer 2)
  ↓ [strong edge weight]
"Cancer predisposition" category (layer 3)
  ↓
Output: Pathogenic

Interpretation: Model learned that Gene A participates in DNA repair, 
disruption causes cancer predisposition phenotype
```

**Scientific value:** Can discover novel gene-pathway associations

### Method 5: Saliency Maps

**For sequence-based models:**

**Process:**
1. Compute gradient of output with respect to input sequence
2. Gradient magnitude = "saliency" (how much each nucleotide matters)
3. Visualize as heatmap over genomic sequence

**Use case:**
- Identify critical nucleotides for variant interpretation
- Example: Splice variant → high saliency at splice donor/acceptor sites
- Example: Regulatory variant → high saliency at transcription factor binding motifs

---

## VI. VALIDATION STRATEGY

### Retrospective Performance Metrics

**Primary metrics:**

1. **Diagnostic yield (gene-level):**
   - Definition: % of cases where true causal gene is in top-K predictions
   - Report for K=1, K=5, K=10
   - Gold standard: Known causal gene from medical record

2. **Variant-level accuracy:**
   - AUC-ROC: Area under receiver operating characteristic curve
   - AUC-PR: Area under precision-recall curve (better for imbalanced data)
   - Sensitivity, specificity, PPV, NPV at clinical threshold (e.g., 75% risk score)

3. **Calibration:**
   - Do predicted probabilities match observed frequencies?
   - Plot: Predicted risk vs observed pathogenic rate
   - Brier score: Lower = better calibration

**Secondary metrics:**

4. **Time-to-diagnosis (simulated):**
   - Rank genes by AI score
   - Measure: How many genes geneticist would need to review before finding true causal gene
   - Compare to standard prioritization (ACMG criteria alone)

5. **Phenotype resolution:**
   - For correctly identified genes, what % of HPO terms are explained?
   - Mean phenotype resolution score across test set

6. **Ranking metrics:**
   - Mean reciprocal rank (MRR): 1 / (rank of true gene)
   - Normalized discounted cumulative gain (NDCG)

### Comparison to Existing Methods

**Baselines to compare:**

1. **Standard clinical workflow:**
   - ACMG criteria applied manually
   - Phenotype-driven filtering (HPO → gene lists)
   - Performance = diagnostic yield in Mayo historical data

2. **Individual tools:**
   - CADD + REVEL + phenotype matching (simple ensemble)
   - Exomiser (popular clinical tool)
   - LIRICAL (likelihood ratio-based approach)

3. **Recent AI methods:**
   - **If reproducible:** DeepRare, AlphaGenome, GenoMAS
   - **Challenge:** May not have access to their models or data
   - **Alternative:** Implement simplified versions based on papers

**Comparison table (for Paper 1):**

| Method | Gene in Top-1 | Gene in Top-5 | AUC-ROC | AUC-PR |
|--------|---------------|---------------|---------|--------|
| Standard clinical | X% | Y% | - | - |
| CADD+REVEL+HPO | A% | B% | 0.XX | 0.XX |
| Exomiser | C% | D% | 0.XX | 0.XX |
| DeepRare (repro) | E% | F% | 0.XX | 0.XX |
| **GenoInsight (ours)** | **G%** | **H%** | **0.XX** | **0.XX** |

**Goal:** Outperform baselines by ≥5% in top-5 yield

### External Validation

**Purpose:** Ensure model generalizes beyond Mayo

**Datasets:**
1. **Baylor Genetics** (clinical lab cases)
   - Independent patient population
   - Different sequencing platform
   - Test for overfitting to Mayo data

2. **Broad Institute** (rare disease cohorts)
   - Publicly available datasets (if IRB-approved access)
   - Different clinical workflows

**Metrics:**
- Same as primary validation (yield, AUC, calibration)
- Report separately for each external site
- Flag if performance drops >10% (generalization issue)

### Ablation Studies

**Question:** Which components contribute most to performance?

**Experiments:**
1. Remove genomic foundation model → quantify impact
2. Remove protein foundation model (ESM2) → quantify impact
3. Remove isoform-specific prediction → use gene-level only
4. Remove network integration → quantify impact
5. Remove multi-ancestry training → train on European only

**Report:**
- Performance difference for each ablation
- Identify critical vs dispensable components

### Failure Analysis

**For misclassified cases:**
- Categorize errors:
  - True gene not in top-10 (missed diagnosis)
  - False positive gene ranked high (incorrect diagnosis)
  - VUS incorrectly classified as pathogenic
- Identify patterns:
  - Specific gene families that fail?
  - Specific phenotypes that fail?
  - Ancestry groups with worse performance?
- Lessons: What can be improved?

---

## VII. IMPLEMENTATION DETAILS

### Computing Infrastructure

**Hardware:**
- **Training:** NVIDIA A100 GPUs (Mayo HPC cluster or cloud)
- **Inference:** Mayo secure servers (HIPAA-compliant)
- Expected training time: 2-4 weeks for full pipeline

**Software stack:**
- Python 3.10+
- PyTorch 2.0+ (deep learning framework)
- Hugging Face Transformers (for ESM2, genomic models)
- NVIDIA BioNeMo SDK (optimized for bio foundation models)
- Microsoft Healthcare Agent Orchestrator (HAO)

### Data Pipeline

**Input processing:**
1. **VCF annotation:**
   - VEP (Variant Effect Predictor) or SnpEff
   - Annotate: Gene, transcript, variant type, population frequency
   
2. **HPO term extraction:**
   - From clinical notes (NLP extraction) or structured data
   - Normalize to standard HPO IDs

3. **RNA-seq processing (if available):**
   - Align with STAR or HISAT2
   - Quantify isoforms with Salmon or RSEM
   - Normalize expression (TPM)

4. **Quality control:**
   - Filter low-quality variants (QUAL < 30)
   - Remove variants in low-complexity regions
   - Exclude common variants (gnomAD AF > 0.01 for most cases)

**Feature engineering:**
- Variant-level features: CADD, REVEL, AlphaMissense, conservation, constraint
- Gene-level features: pLI, LOEUF, expression, network centrality
- Phenotype features: HPO term embeddings, semantic similarity

### Model Training

**Training procedure:**

1. **Pre-training (foundation models):**
   - Use pre-trained weights (ESM2, genomic model)
   - Do NOT retrain from scratch (too expensive)

2. **Fine-tuning (adapter layers):**
   - Add small adapter layers on top of foundation models
   - Train only adapters on Mayo data (freeze foundation model weights)
   - Epochs: 10-20
   - Batch size: 32-64
   - Learning rate: 1e-4 to 1e-5
   - Optimizer: AdamW

3. **GenNet training:**
   - Initialize with biological priors (gene-pathway connections)
   - Train end-to-end on ClinVar + Mayo data
   - Regularization: L1 penalty on gene weights (sparsity)
   - Early stopping: Monitor validation AUC

4. **Multi-ancestry balancing:**
   - Sample batches with balanced ancestry representation
   - Weight loss function by inverse ancestry frequency

**Hyperparameter tuning:**
- Grid search or Bayesian optimization
- Hyperparameters: Learning rate, batch size, dropout rate, regularization strength
- Optimize on validation set (not test set)

### Model Deployment

**For Aim 2 trial (prospective use):**
- Deploy on Mayo secure servers
- API endpoint: Submit VCF + HPO terms → receive prediction
- Turnaround: <48 hours (target)
- Monitoring: Log all inputs/outputs for audit trail

**Model versioning:**
- Git repo with version control
- Docker container for reproducibility
- Model weights stored with commit hash

**Performance monitoring:**
- Track inference time (should be <2 hours per patient)
- Track GPU memory usage
- Set up alerts for failures

---

## VIII. TIMELINE & MILESTONES

### Month-by-Month Plan (Sep 2026 - May 2027)

**Sep-Oct 2026 (Months 1-2): Setup & Data Preparation**
- [ ] Set up Mayo computing accounts, GPU access
- [ ] Obtain IRB approval for retrospective data use
- [ ] Pull Mayo retrospective cohorts (solved + unsolved cases)
- [ ] Annotate VCFs (VEP pipeline)
- [ ] Extract HPO terms from medical records
- [ ] Create train/val/test splits
- [ ] Ancestry PCA on training data
- **Milestone:** Dataset ready (n=5,000+ cases)

**Nov-Dec 2026 (Months 3-4): Foundation Model Selection**
- [ ] Implement genomic model evaluation framework (from Track B prep)
- [ ] Test 4 candidate genomic models on 10 variants
- [ ] Decision: Select genomic foundation model (Jan 2026)
- [ ] Set up model inference pipelines (ESM2 + genomic model)
- [ ] Extract embeddings for all variants in training set
- **Milestone:** Foundation models operational

**Jan-Feb 2027 (Months 5-6): Core Model Development**
- [ ] Implement GenNet architecture
- [ ] Train GenNet on ClinVar + Mayo training data
- [ ] Implement isoform-specific phenotype prediction
- [ ] Integrate network medicine features
- [ ] Hyperparameter tuning on validation set
- **Milestone:** Core model trained (AUC > 0.85 on validation)

**Mar 2027 (Month 7): Multi-Agent Orchestration**
- [ ] Set up Microsoft Healthcare Agent Orchestrator (HAO)
- [ ] Implement literature search agent
- [ ] Implement ACMG classification agent
- [ ] Implement evidence aggregation logic
- [ ] Test on 10 example patients
- **Milestone:** Full pipeline operational

**Apr 2027 (Month 8): Interpretability Implementation**
- [ ] Implement attention visualization (genomic + ESM2)
- [ ] Implement SHAP value calculation
- [ ] Implement counterfactual generation
- [ ] Implement gene-level attribution (GenNet)
- [ ] Create visualization scripts (heatmaps, network graphs)
- **Milestone:** Interpretability methods working

**May 2027 (Month 9): Validation & Analysis**
- [ ] Evaluate on held-out test set (Mayo)
- [ ] Evaluate on external datasets (Baylor, Broad)
- [ ] Ancestry-stratified performance analysis
- [ ] Ablation studies (remove components, measure impact)
- [ ] Failure analysis (characterize misclassified cases)
- [ ] Statistical significance tests (vs baselines)
- **Milestone:** Validation complete, results ready

**Jun 2027 (Month 10): Paper Writing**
- [ ] Draft Introduction
- [ ] Draft Methods (detailed)
- [ ] Draft Results (tables, figures)
- [ ] Draft Discussion (interpret findings, limitations)
- [ ] Create main figures (4-5 figures)
- [ ] Create supplementary materials
- [ ] Internal review by Eric Klee and lab
- **Milestone:** Paper draft complete

**Jul 2027 (Month 11): Revision & Submission**
- [ ] Address co-author feedback
- [ ] Polish figures and tables
- [ ] Finalize supplementary materials
- [ ] Prepare code repository (GitHub) for reproducibility
- [ ] Write cover letter
- **Milestone:** Paper 1 submitted to *Nature Genetics* (target: Jul 2027, allows buffer before trial)

**Aug 2027 (Month 12): Model Freeze for Aim 2**
- [ ] Freeze model weights (version control commit)
- [ ] Set decision thresholds (risk score cutoffs)
- [ ] Deploy model on Mayo servers for trial
- [ ] Silent-run testing (1 month before trial launch)
- **Milestone:** Model locked for Aim 2 trial (Sep 2027 start)

---

## IX. DELIVERABLES

### Paper 1: Technical Methodology

**Target journals:**
1. **Nature Genetics** (IF ~30) - top choice
2. **Genome Medicine** (IF ~12) - backup
3. **Nature Communications** (IF ~17) - backup if above reject

**Authorship:**
- **First author:** [Your Name] (PhD candidate, led development)
- **Senior/corresponding:** Eric Klee (PI, provided data + guidance)
- **Co-authors:**
  - Bioinformatician (data processing)
  - Genetic counselors (phenotype curation)
  - Microsoft collaborators (HAO integration, if substantial contribution)
  - NVIDIA collaborators (BioNeMo optimization, if substantial contribution)

**Title (draft):**
"Interpretable Multi-Omics AI for Rare Disease Diagnosis: A Retrospective Validation Study"

**Abstract structure:**
- Background: Rare disease diagnostic challenge, limitations of existing methods
- Methods: Gene-scale interpretable AI, isoform-specific prediction, multi-ancestry training
- Results: Performance on Mayo + external cohorts, comparison to baselines, interpretability examples
- Conclusions: AI improves diagnostic yield while maintaining interpretability, ready for prospective validation

**Main figures (5-6):**

**Figure 1: System architecture**
- Panel A: Overall workflow (data → foundation models → interpretable network → output)
- Panel B: GenNet architecture (visible neural network with gene/pathway layers)
- Panel C: Multi-agent orchestration schematic

**Figure 2: Performance on retrospective cohorts**
- Panel A: Gene in top-K yield (K=1, 5, 10) - bar chart comparing methods
- Panel B: ROC curve (our method vs baselines)
- Panel C: Precision-recall curve
- Panel D: Calibration plot (predicted vs observed pathogenic rate)

**Figure 3: Multi-ancestry performance**
- Panel A: Training data ancestry composition (pie chart)
- Panel B: Performance by ancestry group (bar chart: AUC per group)
- Panel C: Calibration curves by ancestry (separate lines per group)
- Panel D: Disparate impact analysis

**Figure 4: Interpretability examples**
- Panel A: Attention heatmap (genomic model)
- Panel B: SHAP values (top features for example variant)
- Panel C: Gene-level attribution (GenNet network graph)
- Panel D: Counterfactual explanation (what changes flip prediction?)

**Figure 5: Clinical case studies**
- 3-4 example patients with diagnostic odyssey
- Show: AI top-ranked gene, attention visualization, evidence summary
- Compare: Standard workflow vs AI-assisted workflow

**Figure 6: Comparison to competitors**
- Panel A: Performance comparison table (our method vs DeepRare, AlphaGenome, etc.)
- Panel B: Interpretability comparison (which methods provide explanations?)
- Panel C: Computational cost comparison (inference time, GPU memory)

**Supplementary materials:**
- Full methods details
- Hyperparameter tuning results
- Ablation study results
- All ancestry-stratified metrics
- Failure analysis
- Code availability (GitHub repo)
- Data availability statement

### Software Release

**GitHub repository:** `your-username/genoinsight` (or similar)

**Contents:**
- Model code (PyTorch implementations)
- Training scripts
- Inference pipeline
- Interpretability visualization tools
- Example notebooks
- Documentation (README, installation guide)
- Pre-trained model weights (if possible to share)

**License:** Open source (MIT or Apache 2.0) - check Mayo IP policy

**Purpose:**
- Reproducibility (other labs can validate)
- Community adoption
- Citations (software papers get cited)

### Conference Presentations

**Abstract submissions:**
- **ASHG 2027** (American Society of Human Genetics) - Oct 2027
  - Submit retrospective validation results
  - Platform talk (15 min) or poster
- **RECOMB 2028** (Research in Computational Molecular Biology)
  - Submit methods paper (ML focus)

---

## X. RISK MITIGATION

### Technical Risks

**Risk 1: Foundation models underperform**
- **Probability:** Low-Medium
- **Impact:** Model accuracy insufficient for clinical use
- **Mitigation:**
  - Pre-PhD evaluation framework (Jan 2026) catches this early
  - Backup plan: Use ensemble of simpler models (CADD + phenotype matching)
  - If genomic model fails, rely more heavily on ESM2 (protein-level)

**Risk 2: Multi-ancestry data insufficient**
- **Probability:** Medium
- **Impact:** Model performs poorly on non-European ancestry
- **Mitigation:**
  - Check training data ancestry composition early (Month 1)
  - If <10% non-European, flag to Eric Klee
  - Request data augmentation (partner with other sites for diverse data)
  - Worst case: Report limitation, plan Aim 2 enrollment to oversample underrepresented groups

**Risk 3: Interpretability methods not convincing to clinicians**
- **Probability:** Low
- **Impact:** Model seen as black box despite interpretability features
- **Mitigation:**
  - User testing with Mayo geneticists (Month 7-8)
  - Iterate on visualization based on feedback
  - Focus groups: "Is this explanation helpful?"

**Risk 4: Model doesn't outperform baselines**
- **Probability:** Low
- **Impact:** Paper 1 rejected, delays timeline
- **Mitigation:**
  - Target ≥5% improvement in top-5 yield (conservative)
  - If only marginal improvement, emphasize interpretability as differentiator
  - Pivot to method paper (emphasizing novel architecture) vs performance paper

### Timeline Risks

**Risk 5: IRB delays data access**
- **Probability:** Low (Mayo IRB typically fast for retrospective studies)
- **Impact:** 1-2 month delay in starting
- **Mitigation:**
  - Submit IRB protocol ASAP (Aug 2026, before PhD start)
  - Work on pre-PhD projects (Track A/B/C) while waiting

**Risk 6: Computing resources insufficient**
- **Probability:** Low
- **Impact:** Training takes too long
- **Mitigation:**
  - Test on small dataset first (Month 3-4)
  - If Mayo GPUs slow, request cloud credits (AWS, Google Cloud for research)
  - Collaborate with NVIDIA for BioNeMo compute access

**Risk 7: Paper 1 rejected from top-tier journal**
- **Probability:** Medium (Nature Genetics ~10% acceptance rate)
- **Impact:** 2-3 month delay in resubmission
- **Mitigation:**
  - Have backup journal ready (Genome Medicine)
  - Revise quickly based on reviews
  - Buffer: Submit by Jul 2027 (allows 1-2 months for revisions before Aim 2 trial)

**Risk 8: Model freeze delayed (affects Aim 2)**
- **Probability:** Low
- **Impact:** Aim 2 trial launch delayed
- **Mitigation:**
  - Strict deadline: Model must be frozen by Aug 2027
  - If validation not complete, freeze anyway based on validation set performance
  - Can always report updated results in Paper 1 revision

---

## XI. SUCCESS CRITERIA

**Minimum viable product (MVP):**
- [ ] Model achieves ≥40% gene in top-5 yield (vs ~30% standard clinical)
- [ ] AUC-ROC ≥ 0.85 on held-out test set
- [ ] Interpretability methods operational (attention, SHAP, counterfactuals)
- [ ] Multi-ancestry performance gap <0.1 AUC across groups
- [ ] Model deployed and frozen by Aug 2027 for Aim 2 trial

**Stretch goals:**
- [ ] Model achieves ≥50% gene in top-5 yield
- [ ] AUC-ROC ≥ 0.90
- [ ] External validation (Baylor, Broad) within 5% of Mayo performance
- [ ] Paper 1 accepted at *Nature Genetics* (vs backup journal)
- [ ] Software package with >100 GitHub stars (community adoption)

---

## XII. QUESTIONS FOR ADVISOR (ERIC KLEE)

**Before starting Aim 1:**

1. **Data access:**
   - How many solved rare disease cases at Mayo (2015-2023)?
   - How many with RNA-seq data available?
   - What's ancestry composition? (need ≥20% non-European ideally)
   - IRB process for retrospective data use? (timeline estimate)

2. **Infrastructure:**
   - Can I get Mayo HPC/GPU access for model training?
   - If not, should I use cloud (AWS/Google Cloud)?
   - HIPAA-compliant compute for protected health data?

3. **Collaborations:**
   - Existing partnerships with Microsoft (HAO), NVIDIA (BioNeMo)?
   - Can we access these tools, or need to negotiate?
   - Baylor/Broad collaborations for external validation data?

4. **Clinical workflow:**
   - Which features would geneticists find most useful in AI output?
   - How much detail in reports? (1-page summary vs 10-page deep dive)
   - Current pain points in variant interpretation? (what should AI prioritize)

5. **Model requirements for Aim 2:**
   - Must model be FDA-cleared for trial use? (or research exemption?)
   - What accuracy threshold is "good enough" to deploy in trial?
   - Risk tolerance for false positives/negatives in prospective trial?

6. **Publication strategy:**
   - Is *Nature Genetics* realistic target, or aim lower initially?
   - Should I pursue conference papers (NeurIPS, ICML, RECOMB) in parallel?
   - Preprint on bioRxiv before journal submission?

7. **Intellectual property:**
   - Who owns AI model IP? (Mayo, me, joint?)
   - Restrictions on open-sourcing code/model weights?
   - Patent considerations?

8. **Authorship:**
   - Confirm: Me as first author, you as senior?
   - Other co-authors to include early (data contributors, collaborators)?
   - Order of co-authors?

---

## XIII. INTEGRATION WITH AIMS 2 & 3

### Thesis Narrative

**Aim 1:** Develop interpretable AI system
- Output: Paper 1 (technical methods)
- Establishes: Technical validity on retrospective data

**Aim 2:** Prospectively validate in RCT
- Output: Paper 2 (clinical utility)
- Establishes: Real-world efficacy

**Aim 3:** Long-term outcomes & economics
- Output: Paper 3 (health economics)
- Establishes: Cost-effectiveness and impact

**Story arc:** Build it → Test it → Prove it matters

### Dependencies

**Aim 2 depends on Aim 1:**
- Model must be trained and validated before trial starts (Sep 2027)
- Buffer: Aim 1 paper submitted Jul 2027, model frozen Aug 2027
- 1-month silent-run (Aug 2027) before trial launch

**Aim 3 depends on Aim 2:**
- Needs patients from Aim 2 trial
- But data collection can start as soon as first patients reach 12-month mark (Jul 2028)

**Timeline flow:**
```
Sep 2026 ─────────────┬───────── May 2027 ─────────── Jul 2027 ─────────── Aug 2027
          Aim 1        │         Paper 1              Paper 1              Model
        Development    │        Complete            Submitted             Frozen
                       │                                                      │
                       └──────────────────────────────────────────────────────┘
                                                                              │
                                                                              ▼
Aug 2027 ────────────────────────────────────────────────────────────── Aug 2029
                            Aim 2 Trial Enrollment                            │
                                                                              │
                       Jul 2028 (first patients reach 12-month mark)         │
                              │                                               │
                              ▼                                               │
Jul 2028 ──────────────────────────────────────────────────────────────── Apr 2030
                          Aim 3 Data Collection                                
                               & Analysis                                      
                                                                              │
                                                                              ▼
                                                                         Apr 2030
                                                                      PhD Defense
```

---

**Status:** Draft for discussion  
**Next step:** Review with Eric Klee to confirm data availability, feasibility, and timeline

**Document Control:**
- Version: 1.0 Draft
- Date: October 31, 2025
- Author: [Your Name]
- Reviewer: Eric Klee (pending)
