# Pre-PhD Intensive Study Plan v4 (REVISED - WITH ANCESTRY ROBUSTNESS)
## Nov 2025 - Sep 2026 (10 Months)
### Optimized for your advanced ML background: Master genomics + clinical context + network medicine + competitive landscape + robustness

---

## REVISION NOTES (v4)

**Changes from v3:**
1. **Genomic Foundation Model (GLM) Decision Point:** Removed Evo2-specific implementation. Track A now uses ESM2 (proteins) for learning interpretability fundamentals. Genomic model selection (Enformer, Nucleotide Transformer, or alternatives) deferred to decision point in Track B Phase 2 (Jan 2026) based on usability and performance evaluation.
2. **Ancestry & Confounder Robustness:** Added comprehensive coverage in Track B Phase 1 (3-4 hours exercises + 2.5 hours reading). Includes population stratification, allele frequency confounding, training data bias analysis, and fairness metrics.
3. **Updated time estimates:** Track B now 105-130 hours (was 100-120) to accommodate ancestry work.

---

## YOUR CURRENT STATE (Recalibrated)

**You already know:**
- ✅ Transformers, BERT fundamentals
- ✅ ESM papers, Enformer
- ✅ Working with ESM2 + proprietary dual-llama encoder-decoder (last token bottleneck architecture)
- ✅ Saliency maps (DL course 2 years ago)
- ✅ Network science fundamentals (Georgia Tech course, Barabasi textbook, Uri Alon regulatory motifs)

**You need to master (Priority order):**
1. ❌ **Genomics domain** - ACMG classification, HPO, phenotype prediction, rare disease workflows, **ancestry robustness**
2. ❌ **Network medicine & systems biology** - PPI networks, network propagation, pathway logic, genotype→phenotype via topology
3. ❌ **Clinical context** - Diagnostic odyssey, Mayo workflows, IRB/regulatory
4. ❌ **Competitive landscape** - DeepRare, AlphaGenome, GenoMAS (June-July 2025)
5. ❌ **HAO multi-agent systems** - Microsoft orchestrator platform
6. ❌ **Prospective study design** - Clinical trial methodology
7. ⚠️ **Attention maps + interpretability** - Apply to your models (can do in parallel)

**This plan structure:**
- **Track A (2-3 weeks):** Interpretability quick-wins (attention maps + SHAP on ESM2 and your encoder-decoder)
- **Track B (10-12 weeks):** Genomics domain + ancestry robustness + network medicine + clinical (heavy focus) **[EXPANDED]**
- **Track C (6-8 weeks):** Competitive analysis + HAO + study design
- **Concurrent:** Weekly competitive papers + integrated projects

---

## TRACK A: INTERPRETABILITY ON YOUR MODELS (Nov-Dec 2025 - 2-3 weeks total)
[track_a_plan.md](tracks/track_a_plan.md)
### Phase 1: Attention Maps on ESM2 & Your Encoder-Decoder (Nov 2025 - 1.5 weeks)
### Phase 2: SHAP + LIME + Counterfactual (Dec 2025 - 1.5 weeks)

---

## TRACK B: GENOMICS + ANCESTRY ROBUSTNESS + NETWORK MEDICINE (Dec 2025 - Apr 2026 - 10-12 weeks)
[track_b_plan.md](tracks/track_b_plan.md)
### Phase 1: ACMG + Variant Interpretation + Ancestry Robustness (Dec 2025 - 3 weeks)
### Phase 2: Genomic Foundation Model Selection + Network Medicine (Jan-Feb 2026 - 3-4 weeks)
### Phase 3: Rare Disease Diagnostic Workflows & Clinical Context (Mar 2026 - 2 weeks)

---

## TRACK C: COMPETITIVE ANALYSIS + HAO + STUDY DESIGN (Apr-Jun 2026 - 6-8 weeks)
[track_c_plan.md](tracks/track_c_plan.md)
### Phase 1: Competitive Paper Deep Dive (Apr-May 2026 - 3-4 weeks)
### Phase 2: HAO Multi-Agent System Design (May 2026 - 2-3 weeks)
### Phase 3: Prospective Study Design & IRB Preparation (Jun 2026 - 2-3 weeks)

---

## INTEGRATED PROJECTS (Concurrent: Nov 2025 - Sep 2026)
[project_plan.md](projects/project_plan.md)
### Project 1: Variant Effect Prediction System (Nov-Jan 2026)
### Project 2: Phenotype-Driven Variant Ranking (Jan-Feb 2026)
### Project 3: Patient-Scale Heterogeneous Graph Networks for Diagnosis (Jan-Apr 2026) **[MAJOR]**
### Project 4: Multi-Agent HAO Workflow (Feb-Apr 2026)
### Project 5: Competitive Paper Reproduction (Mar-Apr 2026)

---

## WEEKLY COMPETITIVE PAPER READING (Ongoing Jan-Sep 2026)

**Format:** 1 competitive paper per week, lightweight tracking

| Month   | Paper                       | Focus                    |
| ------- | --------------------------- | ------------------------ |
| Jan     | DeepRare                    | Multi-agent architecture |
| Jan-Feb | AlphaGenome                 | Regulatory variants      |
| Feb     | GenoMAS                     | Multi-agent reasoning    |
| Feb-Mar | Follow-up papers from above | Refinements, benchmarks  |
| Mar     | Exomiser recent updates     | Baseline comparison      |
| Mar-Apr | PhenoLinker / similar       | Phenotype prediction     |
| Apr-May | IMPPROVE deep dive          | Your direct comparison   |
| May-Jun | Other competitive papers    | Catch-up reading         |
| Jun-Aug | Your deep-dive topics       | Specialized knowledge    |

---

## SUCCESS CRITERIA (By Sep 2026)

**Core competencies (Must have):**
- [ ] Fluent in ACMG variant classification
- [ ] **Understand ancestry confounding and can implement mitigation strategies**
- [ ] **Genomic foundation model selected and benchmarked**
- [ ] **Understand how network topology predicts phenotypic breadth**
- [ ] **Apply regulatory motif logic to disease mechanism**
- [ ] Build end-to-end variant ranking system (phenotype-aware + network-aware + ancestry-robust)
- [ ] Explain exactly where DeepRare/AlphaGenome/GenoMAS succeed and fail
- [ ] Design and orchestrate multi-agent diagnostic system in HAO
- [ ] Have prospective study protocol draft ready
- [ ] Have 5 completed projects + GitHub portfolio (60+ hours of code)
- [ ] Have 40+ hours of reading notes from competitive papers + network medicine papers + ancestry/robustness papers

**If all true → Ready to hit ground running on Aim 1**

---

## GitHub Repository Structure

```
gfm-discovery/
├── README.md
├── requirements.txt
├── setup.sh
│
├── 01_interpretability/
│   ├── attention_visualization_esm2.ipynb
│   ├── attention_visualization_encoder_decoder.ipynb
│   ├── shap_variant_ranking.ipynb
│   ├── counterfactual_variants.ipynb
│   └── interpretability_comparison.md
│
├── 02_genomics_domain/
│   ├── acmg_classification_exercise.ipynb
│   ├── ancestry_stratification_analysis.ipynb  # NEW
│   ├── pca_ancestry_correction.ipynb  # NEW
│   ├── training_data_bias_mitigation.md  # NEW
│   ├── hpo_exploration.ipynb
│   ├── annotation_pipeline.py
│   └── gene_phenotype_baseline.ipynb
│
├── 03_genomic_models/  # NEW DIRECTORY
│   ├── model_evaluation_framework.md
│   ├── enformer_evaluation.ipynb
│   ├── nucleotide_transformer_evaluation.ipynb
│   ├── model_comparison_report.md
│   └── genomic_model_selection.md
│
├── 04_network_medicine/
│   ├── ppi_network_analysis.ipynb
│   ├── network_propagation.py
│   ├── network_vs_baseline_comparison.ipynb
│   ├── multi_layer_networks.ipynb
│   └── network_phenotype_correlation.md
│
├── 05_integrated_projects/
│   ├── project1_variant_effect_system.ipynb
│   ├── project2_phenotype_ranking_pipeline.ipynb
│   ├── project3_network_medicine_analysis.ipynb
│   ├── project4_hao_workflow/
│   │   ├── workflow_design.md
│   │   ├── orchestrator.py
│   │   └── agent_implementations.py
│   └── project5_competitive_reproduction.ipynb
│
├── 06_competitive_analysis/
│   ├── deeprare_analysis.md
│   ├── alphagenome_analysis.md
│   ├── genomas_analysis.md
│   └── comparison_matrix.xlsx
│
└── 07_clinical_context/
    ├── diagnostic_odyssey_cases.md
    ├── workflow_integration_design.md
    ├── prospective_study_protocol_draft.md
    └── study_feasibility_notes.md
```

---

## Total Time Investment

- **Track A:** 20-25 hours
- **Track B:** 105-130 hours (expanded from 100-120 to include ancestry + GLM selection)
- **Track C:** 40-50 hours
- **Concurrent projects:** 50-60 hours
- **Weekly paper reading:** 20-30 hours
- **Total:** ~235-295 hours over 40 weeks = 6-7.5 hours/week average (very doable)

**Result:** Deeply prepared, competitive foundation for PhD thesis with network medicine expertise + ancestry robustness

---

## Final Notes

1. **You're in a unique position**: You have state-of-the-art models (ESM2 + your encoder-decoder + genomic model TBD) + you're embedded at Mayo + you understand ML deeply + **you have network science fundamentals** + **you're addressing ancestry bias proactively**. Your differentiation is network-aware + ancestry-robust + prospective validation.

2. **Genomic model flexibility**: By deferring the genomic foundation model selection to Jan 2026, you can choose based on: (1) latest models released, (2) usability improvements, (3) performance on your specific use cases. This is strategic given the fast-moving field.

3. **Ancestry robustness as differentiator**: Most competitors (DeepRare, AlphaGenome) don't explicitly address ancestry confounding. By building this into your methodology from the start, you create a publication/funding angle around fairness and generalizability.

4. **Pragmatic time management**: ~6-7 hours/week is sustainable alongside Mayo work and GoodFire. Front-load reading during weeks with lighter Mayo commitments.

5. **Continuous feedback**: Share Project 1-2 outputs with Eric Klee lab early (Jan-Feb 2026) to get clinical feedback and refine direction before committing to full thesis approach.

Good luck! 🚀
