# Project 3: Patient-Scale Heterogeneous GNN for Rare Disease Diagnosis

## Implementation Plan (Jan-Apr 2026)

**Target Publication:** Bioinformatics or NPJ Digital Medicine
**Submission Target:** May 2026
**Differentiator:** Network-aware + ancestry-robust + interpretable

---

## 1. Executive Summary

### The Problem
Current rare disease diagnostic tools treat variants in isolation, missing critical network context. Genes don't function alone—they operate in regulatory networks, protein complexes, and metabolic pathways. A variant's pathogenicity depends on its network position.

### Our Solution
Patient-scale heterogeneous graph neural networks (HetGNN) that:
1. Represent each patient as a multi-relational graph (PPI, regulatory, pathway, co-expression)
2. Learn patient-level embeddings via message passing across edge types
3. Predict diagnostic category with interpretable subgraph explanations
4. Maintain fairness across ancestry groups

### Key Innovation
- **Per-patient graphs**: Unlike global network approaches, each patient gets a personalized network centered on their variants
- **Heterogeneous edges**: Different relationship types (PPI vs regulatory vs pathway) are modeled separately
- **Ancestry robustness**: Explicit fairness constraints and stratified evaluation

---

## 2. Technical Architecture

### 2.1 Graph Construction Pipeline

```
Input: Patient VCF + HPO terms
           │
           ▼
┌─────────────────────────────┐
│ 1. VARIANT EXTRACTION       │
│    - Filter: coding + splice│
│    - Annotate: VEP + AF     │
│    - Output: gene set G     │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 2. NETWORK EXPANSION        │
│    - 1-2 hop neighbors      │
│    - K=50-100 genes total   │
│    - Sources: STRING,       │
│      Reactome, ENCODE       │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 3. EDGE CONSTRUCTION        │
│    - PPI (STRING ≥700)      │
│    - TF regulation (ENCODE) │
│    - Pathway (Reactome)     │
│    - Co-expression (GTEx)   │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 4. NODE FEATURES            │
│    - Pathogenicity scores   │
│    - Expression (GTEx)      │
│    - Conservation (PhyloP)  │
│    - Network centrality     │
│    - Ancestry (optional)    │
└─────────────────────────────┘
           │
           ▼
Output: PyTorch Geometric HeteroData
```

### 2.2 Model Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    HetGNN Architecture                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input: HeteroData(                                        │
│    gene: [N, d_node],     # Node features                  │
│    ppi: [2, E_ppi],       # PPI edges                      │
│    regulatory: [2, E_tf], # TF-target edges                │
│    pathway: [2, E_pw],    # Pathway edges                  │
│    coexpr: [2, E_cx]      # Co-expression edges            │
│  )                                                         │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ Option A: R-GCN (Relational GCN)                 │     │
│  │ - Separate weight matrices per edge type         │     │
│  │ - h_i^(l+1) = σ(Σ_r Σ_j W_r h_j + W_0 h_i)      │     │
│  │ - Pro: Explicit edge type modeling               │     │
│  │ - Con: Parameter explosion with many relations   │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │ Option B: HAN (Heterogeneous Attention Network)  │     │
│  │ - Hierarchical attention: node + semantic        │     │
│  │ - Learns importance of edge types automatically  │     │
│  │ - Pro: Interpretable attention weights           │     │
│  │ - Con: More complex training                     │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  Layers: 2-3 message passing layers                        │
│  Pooling: Mean/attention pooling → graph embedding         │
│  Head: MLP → diagnostic category prediction                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 2.3 Ancestry Robustness Design

```
Fairness Integration Points:

1. NODE FEATURES
   - Include ancestry PC1-PC3 as optional features
   - Ablation: with vs without ancestry features

2. TRAINING
   - Stratified batching by ancestry deciles
   - Group DRO loss: optimize worst-group performance
   - Regularization: penalize ancestry-predictive embeddings

3. EVALUATION
   - Stratified metrics: AUROC per ancestry group
   - Fairness gap: max - min group performance
   - Target: all groups within 5% of best
```

---

## 3. Data Requirements

### 3.1 Patient Cohort

| Source | N Patients | Phenotype | Ancestry | Access |
|--------|------------|-----------|----------|--------|
| ClinVar submissions | ~500 | Disease category | Mixed | Public |
| Decipher | ~300 | HPO terms | EUR-heavy | Application |
| 100K Genomes | ~1000 | ICD10 + HPO | UK diverse | Application |
| Mayo UDP (future) | ~50 | Full clinical | US diverse | Internal |

**Minimum viable:** 200-300 patients with VCF + phenotype + known diagnosis
**Target:** 500+ patients for robust training

### 3.2 Network Databases

| Database | Content | Version | Edges |
|----------|---------|---------|-------|
| STRING | PPI | v12.0 | ~12M (human, ≥700 score) |
| Reactome | Pathways | 2024 | ~250K gene-pathway |
| ENCODE | TF-target | Phase 3 | ~2M regulatory |
| GTEx | Co-expression | v8 | Top 1% correlations |
| gnomAD | Allele freq | v4 | Constraint scores |

### 3.3 Node Feature Sources

| Feature | Source | Dimension |
|---------|--------|-----------|
| Pathogenicity | CADD, AlphaMissense | 2 |
| Expression | GTEx (49 tissues) | 49 or PCA → 10 |
| Conservation | PhyloP, PhastCons | 2 |
| Constraint | gnomAD pLI, LOEUF | 2 |
| Network | Degree, betweenness, PageRank | 4 |
| **Total** | | **~20-70 dims** |

---

## 4. Implementation Phases

### Phase 1: Data Infrastructure (Jan 1-21, 2026) - 3 weeks

#### Week 1: Network Database Setup
| Task | Hours | Deliverable |
|------|-------|-------------|
| Download STRING v12, filter to human ≥700 | 2 | `data/networks/string_human_700.parquet` |
| Download Reactome gene-pathway mappings | 2 | `data/networks/reactome_genes.parquet` |
| Download ENCODE TF-target (ChIP-seq peaks) | 3 | `data/networks/encode_tf_targets.parquet` |
| Compute GTEx co-expression (top 1%) | 4 | `data/networks/gtex_coexpr.parquet` |
| Build unified gene ID mapping (Ensembl) | 2 | `data/mappings/gene_id_map.json` |
| **Subtotal** | **13** | |

#### Week 2: Patient Data Pipeline
| Task | Hours | Deliverable |
|------|-------|-------------|
| ClinVar pathogenic variants download | 2 | `data/patients/clinvar_pathogenic.vcf` |
| Parse patient-variant-phenotype links | 4 | `data/patients/patient_variants.parquet` |
| HPO term extraction and mapping | 3 | `data/patients/patient_hpo.parquet` |
| Ancestry inference pipeline (1KG projection) | 4 | `data/patients/patient_ancestry.parquet` |
| Train/val/test split (stratified by ancestry) | 2 | `data/splits/` |
| **Subtotal** | **15** | |

#### Week 3: Graph Construction
| Task | Hours | Deliverable |
|------|-------|-------------|
| Implement `PatientGraphBuilder` class | 6 | `src/graph/builder.py` |
| Variant → gene extraction | 2 | Integrated in builder |
| K-hop neighbor expansion | 3 | Integrated in builder |
| Multi-relation edge construction | 4 | Integrated in builder |
| Node feature assembly | 3 | `src/graph/features.py` |
| Unit tests for graph construction | 2 | `tests/test_graph_builder.py` |
| Build 100-150 patient graphs | 2 | `data/graphs/patient_graphs.pt` |
| **Subtotal** | **22** | |

**Phase 1 Total: ~50 hours**

---

### Phase 2: Model Development (Jan 22 - Feb 18, 2026) - 4 weeks

#### Week 4: Baseline Implementations
| Task | Hours | Deliverable |
|------|-------|-------------|
| Baseline 1: ACMG-only ranking | 3 | `src/baselines/acmg_baseline.py` |
| Baseline 2: Network propagation (Guney et al.) | 5 | `src/baselines/network_propagation.py` |
| Baseline 3: Simple GCN (no edge types) | 4 | `src/baselines/simple_gcn.py` |
| Baseline evaluation harness | 3 | `src/evaluation/baseline_eval.py` |
| **Subtotal** | **15** | |

#### Week 5-6: HetGNN Implementation
| Task | Hours | Deliverable |
|------|-------|-------------|
| R-GCN layer implementation (PyG) | 6 | `src/models/rgcn.py` |
| HAN layer implementation (PyG) | 8 | `src/models/han.py` |
| Graph pooling strategies | 4 | `src/models/pooling.py` |
| Classification head | 2 | `src/models/heads.py` |
| Training loop with early stopping | 4 | `src/training/trainer.py` |
| Hyperparameter config system | 2 | `configs/model_configs.yaml` |
| **Subtotal** | **26** | |

#### Week 7: Ancestry Robustness Integration
| Task | Hours | Deliverable |
|------|-------|-------------|
| Stratified batch sampler | 3 | `src/training/samplers.py` |
| Group DRO loss implementation | 4 | `src/training/losses.py` |
| Fairness metrics (per-group AUROC) | 3 | `src/evaluation/fairness.py` |
| Ancestry ablation experiments | 4 | `notebooks/ancestry_ablation.ipynb` |
| **Subtotal** | **14** | |

**Phase 2 Total: ~55 hours**

---

### Phase 3: Experiments & Evaluation (Feb 19 - Mar 18, 2026) - 4 weeks

#### Week 8-9: Main Experiments
| Task | Hours | Deliverable |
|------|-------|-------------|
| Train R-GCN (5 seeds) | 4 | `results/rgcn/` |
| Train HAN (5 seeds) | 4 | `results/han/` |
| Hyperparameter tuning (Optuna) | 6 | `results/hparam_search/` |
| Compare: HetGNN vs baselines | 4 | `results/comparison/` |
| Statistical significance tests | 2 | Integrated in results |
| **Subtotal** | **20** | |

#### Week 10: Ablation Studies
| Task | Hours | Deliverable |
|------|-------|-------------|
| Edge type ablation (remove each type) | 6 | `results/ablations/edge_types/` |
| Node feature ablation | 4 | `results/ablations/node_features/` |
| Graph size sensitivity (K=25,50,100) | 3 | `results/ablations/graph_size/` |
| Layer depth ablation (1,2,3 layers) | 2 | `results/ablations/depth/` |
| **Subtotal** | **15** | |

#### Week 11: Interpretability Analysis
| Task | Hours | Deliverable |
|------|-------|-------------|
| Attention weight extraction (HAN) | 3 | `src/interpretation/attention.py` |
| Subgraph explanation (GNNExplainer) | 5 | `src/interpretation/explainer.py` |
| Case study: 5 patients deep dive | 6 | `notebooks/case_studies.ipynb` |
| Visualize disease-driving subgraphs | 4 | `figures/subgraph_viz/` |
| **Subtotal** | **18** | |

**Phase 3 Total: ~53 hours**

---

### Phase 4: Paper Writing (Mar 19 - Apr 30, 2026) - 6 weeks

#### Week 12-13: Results & Figures
| Task | Hours | Deliverable |
|------|-------|-------------|
| Main comparison figure (bar/box plot) | 3 | `figures/fig1_comparison.pdf` |
| Ablation heatmap | 2 | `figures/fig2_ablation.pdf` |
| Fairness stratification plot | 2 | `figures/fig3_fairness.pdf` |
| Case study figure (network + attention) | 4 | `figures/fig4_case_study.pdf` |
| Supplementary figures | 4 | `figures/supp/` |
| **Subtotal** | **15** | |

#### Week 14-15: Manuscript Draft
| Task | Hours | Deliverable |
|------|-------|-------------|
| Abstract (250 words) | 2 | `paper/abstract.md` |
| Introduction (1.5 pages) | 6 | `paper/introduction.md` |
| Methods (3 pages) | 8 | `paper/methods.md` |
| Results (2.5 pages) | 6 | `paper/results.md` |
| Discussion (1.5 pages) | 4 | `paper/discussion.md` |
| **Subtotal** | **26** | |

#### Week 16-17: Review & Submission
| Task | Hours | Deliverable |
|------|-------|-------------|
| Internal review (Eric Klee lab) | 4 | Feedback incorporated |
| Revisions based on feedback | 8 | Updated manuscript |
| Code cleanup & documentation | 4 | GitHub release |
| Submission preparation | 2 | Submitted manuscript |
| **Subtotal** | **18** | |

**Phase 4 Total: ~59 hours**

---

## 5. Evaluation Framework

### 5.1 Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **AUROC** | Diagnostic category prediction | >0.80 |
| **AUPRC** | Important for class imbalance | >0.60 |
| **Top-5 Recall** | Correct gene in top 5 | >0.70 |
| **Fairness Gap** | Max - min ancestry group AUROC | <0.05 |

### 5.2 Baseline Comparison Matrix

| Method | Network | Heterogeneous | Ancestry-Aware | Interpretable |
|--------|---------|---------------|----------------|---------------|
| ACMG alone | ❌ | ❌ | ❌ | ✅ |
| Network propagation | ✅ | ❌ | ❌ | ⚠️ |
| Simple GCN | ✅ | ❌ | ❌ | ⚠️ |
| **HetGNN (ours)** | ✅ | ✅ | ✅ | ✅ |

### 5.3 Ablation Hypotheses

| Ablation | Hypothesis | Expected Impact |
|----------|------------|-----------------|
| Remove PPI edges | Network topology matters | -5-10% AUROC |
| Remove regulatory edges | TF context helps | -3-5% AUROC |
| Remove pathway edges | Biological processes help | -2-4% AUROC |
| Remove co-expression | Tissue context helps | -2-3% AUROC |
| Remove ancestry features | Fairness degrades | +2-3% fairness gap |

---

## 6. Paper Outline

### Title Options
1. "Patient-Scale Heterogeneous Graph Networks for Ancestry-Robust Rare Disease Diagnosis"
2. "Network Medicine Meets Foundation Models: Heterogeneous GNNs for Diagnostic Prediction"
3. "Beyond Variant Scores: Learning Disease-Driving Network Modules with Graph Neural Networks"

### Abstract Structure (250 words)
- **Problem:** Rare disease diagnosis misses 50%+ of cases; variants evaluated in isolation
- **Gap:** Network context matters but existing tools don't leverage heterogeneous relationships
- **Method:** Patient-scale HetGNN with PPI, regulatory, pathway, co-expression edges
- **Results:** X% improvement over baselines; Y% fairness gap reduction
- **Impact:** Interpretable, ancestry-robust diagnostic prioritization

### Section Outline

**1. Introduction (1.5 pages)**
- Rare disease diagnostic challenge (25-50% yield)
- Network medicine hypothesis: genes function in context
- Limitations of current approaches (ACMG, single-variant)
- Our contribution: heterogeneous GNN + ancestry robustness

**2. Methods (3 pages)**
- 2.1 Patient graph construction
- 2.2 Heterogeneous GNN architecture (R-GCN, HAN)
- 2.3 Ancestry-robust training (stratification, Group DRO)
- 2.4 Interpretability (attention, GNNExplainer)
- 2.5 Evaluation metrics and baselines

**3. Results (2.5 pages)**
- 3.1 HetGNN outperforms baselines
- 3.2 Edge type ablation: PPI most important
- 3.3 Ancestry stratification: fairness maintained
- 3.4 Case studies: interpretable disease modules

**4. Discussion (1.5 pages)**
- Comparison to DeepRare, AlphaGenome (multi-agent vs graph)
- Limitations: data availability, network completeness
- Clinical translation pathway
- Future: prospective validation at Mayo

---

## 7. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient patient data | Medium | High | Use ClinVar + Decipher; synthetic augmentation |
| Network databases incomplete | High | Medium | Multiple sources; report coverage |
| HetGNN doesn't beat baselines | Low | Critical | Multiple architectures; ablation to understand |
| Ancestry groups too small | High | Medium | Focus on EUR/AFR/EAS; report uncertainty |
| Compute limitations | Low | Medium | Cloud burst; efficient batching |

---

## 8. Success Criteria

### Minimum Viable Paper
- [ ] HetGNN beats ACMG baseline by ≥5% AUROC
- [ ] At least one edge type ablation shows significant contribution
- [ ] Fairness gap <10% across ancestry groups
- [ ] 3+ interpretable case studies

### Stretch Goals
- [ ] HetGNN beats all baselines by ≥10%
- [ ] Fairness gap <5%
- [ ] External validation on held-out cohort
- [ ] Accepted to ASHG 2026 as poster/talk

---

## 9. Timeline Summary

```
Jan 2026
├── Week 1-3: Data infrastructure (Phase 1)
│   ├── Network databases
│   ├── Patient data pipeline
│   └── Graph construction

Feb 2026
├── Week 4-7: Model development (Phase 2)
│   ├── Baselines
│   ├── HetGNN implementation
│   └── Ancestry robustness

Mar 2026
├── Week 8-11: Experiments (Phase 3)
│   ├── Main experiments
│   ├── Ablations
│   └── Interpretability

Apr 2026
├── Week 12-17: Paper writing (Phase 4)
│   ├── Figures
│   ├── Manuscript
│   └── Submission (May 2026)
```

---

## 10. Resource Requirements

### Compute
- GPU: 1x A100 (or 2x V100) for training
- Storage: ~100GB for networks + patient data
- Estimated training time: 2-4 hours per model

### Software Stack
- PyTorch 2.x
- PyTorch Geometric 2.4+
- Optuna (hyperparameter tuning)
- Weights & Biases (experiment tracking)
- NetworkX (graph utilities)

### Collaboration
- Eric Klee lab: Clinical feedback, case study review
- Mayo UDP: Future prospective validation data

---

## Appendix A: Directory Structure

```
gfm-discovery/
├── 03_network_medicine/
│   └── project3_hetgnn/
│       ├── README.md
│       ├── configs/
│       │   ├── model_configs.yaml
│       │   └── data_configs.yaml
│       ├── data/
│       │   ├── networks/
│       │   ├── patients/
│       │   ├── graphs/
│       │   └── splits/
│       ├── src/
│       │   ├── graph/
│       │   │   ├── builder.py
│       │   │   └── features.py
│       │   ├── models/
│       │   │   ├── rgcn.py
│       │   │   ├── han.py
│       │   │   └── pooling.py
│       │   ├── baselines/
│       │   │   ├── acmg_baseline.py
│       │   │   ├── network_propagation.py
│       │   │   └── simple_gcn.py
│       │   ├── training/
│       │   │   ├── trainer.py
│       │   │   ├── samplers.py
│       │   │   └── losses.py
│       │   ├── evaluation/
│       │   │   ├── metrics.py
│       │   │   └── fairness.py
│       │   └── interpretation/
│       │       ├── attention.py
│       │       └── explainer.py
│       ├── notebooks/
│       │   ├── 01_data_exploration.ipynb
│       │   ├── 02_graph_construction.ipynb
│       │   ├── 03_model_training.ipynb
│       │   ├── 04_ablation_analysis.ipynb
│       │   ├── 05_case_studies.ipynb
│       │   └── ancestry_ablation.ipynb
│       ├── results/
│       │   ├── rgcn/
│       │   ├── han/
│       │   ├── comparison/
│       │   └── ablations/
│       ├── figures/
│       └── paper/
│           ├── abstract.md
│           ├── introduction.md
│           ├── methods.md
│           ├── results.md
│           └── discussion.md
```

---

## Appendix B: Key References

### Network Medicine
1. Barabási et al. 2011 - Network medicine framework (Nature Reviews Genetics)
2. Menche et al. 2015 - Disease module detection (Science)
3. Guney et al. 2016 - Network propagation for drug efficacy (Nature Communications)

### Graph Neural Networks
4. Kipf & Welling 2017 - GCN (ICLR)
5. Schlichtkrull et al. 2018 - R-GCN for knowledge graphs (ESWC)
6. Wang et al. 2019 - HAN for heterogeneous graphs (WWW)
7. Ying et al. 2019 - GNNExplainer (NeurIPS)

### Rare Disease Genomics
8. Richards et al. 2015 - ACMG guidelines (Genetics in Medicine)
9. Smedley et al. 2015 - Exomiser (Nature Protocols)
10. Martin et al. 2019 - Ancestry bias in PRS (Nature Genetics)

### Competitive
11. DeepRare (2025) - Multi-agent rare disease diagnosis
12. AlphaGenome (2025) - Google DeepMind regulatory variants
13. GenoMAS (2025) - Multi-agent genomic reasoning
