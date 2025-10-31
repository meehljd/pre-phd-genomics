# Pre-PhD Intensive Study Plan (REVISED - WITH SYSTEMS BIOLOGY)
## Nov 2025 - Sep 2026 (10 Months)
### Optimized for your advanced ML background: Master genomics + clinical context + network medicine + competitive landscape

---

## YOUR CURRENT STATE (Recalibrated)

**You already know:**
- ✅ Transformers, BERT fundamentals
- ✅ ESM papers, Enformer
- ✅ Working with ESM2 + proprietary dual-llama encoder-decoder (last token bottleneck architecture)
- ✅ Saliency maps (DL course 2 years ago)
- ✅ Network science fundamentals (Georgia Tech course, Barabasi textbook, Uri Alon regulatory motifs)

**You need to master (Priority order):**
1. ❌ **Genomics domain** - ACMG classification, HPO, phenotype prediction, rare disease workflows
2. ❌ **Network medicine & systems biology** - PPI networks, network propagation, pathway logic, genotype→phenotype via topology
3. ❌ **Clinical context** - Diagnostic odyssey, Mayo workflows, IRB/regulatory
4. ❌ **Competitive landscape** - DeepRare, AlphaGenome, GenoMAS (June-July 2025)
5. ❌ **HAO multi-agent systems** - Microsoft orchestrator platform
6. ❌ **Prospective study design** - Clinical trial methodology
7. ⚠️ **Attention maps + interpretability** - Apply to your models (can do in parallel)

**This plan structure:**
- **Track A (2-3 weeks):** Interpretability quick-wins (attention maps + SHAP on your models)
- **Track B (10-11 weeks):** Genomics domain + network medicine + clinical (heavy focus) **[EXPANDED]**
- **Track C (6-8 weeks):** Competitive analysis + HAO + study design
- **Concurrent:** Weekly competitive papers + integrated projects

---

## TRACK A: INTERPRETABILITY ON YOUR MODELS (Nov-Dec 2025 - 2-3 weeks total)

### Phase 1: Attention Maps on ESM2 & Your Encoder-Decoder (Nov 2025 - 1.5 weeks)
**Goal:** Extract + visualize attention patterns; understand what your bottleneck learns

### Phase 2: SHAP + LIME + Counterfactual (Dec 2025 - 1.5 weeks)
**Goal:** Advanced explanations on your variant ranking system


## TRACK B: GENOMICS + NETWORK MEDICINE (Dec 2025 - Apr 2026 - 10-11 weeks)

### Phase 1: ACMG + Variant Interpretation Fundamentals (Dec 2025 - 2 weeks)

**Goal:** Fluency in how clinicians classify variants; this is your target framework

### Phase 2: Phenotype Prediction & Gene-to-Phenotype Mapping via Networks (Jan 2026 - 4 weeks) **[EXPANDED WITH SYSTEMS BIOLOGY]**

**Goal:** Link variants → genes → phenotypes via HPO + biological network topology; understand how network position predicts phenotypic breadth and disease mechanism

### Phase 3: Rare Disease Diagnostic Workflows & Clinical Context (Feb 2026 - 2 weeks)

**Goal:** Deep understanding of diagnostic odyssey; how Mayo's system works; integration points

### Phase 4: Competitive Deep-Dive & Multi-Agent Design (Mar 2026 - 2 weeks)

**Goal:** Master competitive systems; understand where you differentiate; design HAO workflow

---

## TRACK C: HAO IMPLEMENTATION + STUDY DESIGN (Apr 2026 - May 2026 - 6-8 weeks)

### Phase 1: HAO Multi-Agent System Implementation (Apr 2026 - 3 weeks)

**Goal:** Build working orchestrator platform (using Microsoft HAO or custom framework)

### Phase 2: Prospective Study Design & IRB Preparation (May 2026 - 2-3 weeks)

**Goal:** Design realistic prospective validation; prepare for regulatory review

### Track C Integration: Weekly Competitive Paper Reading (Ongoing Jan-Sep 2026)

---

## INTEGRATED PROJECTS (Concurrent: Nov 2025 - Sep 2026)

### Project 1: Variant Effect Prediction System (Nov-Jan 2026)

### Project 2: Phenotype-Driven Variant Ranking (Jan-Feb 2026)

### Project 3: Patient-Scale Heterogeneous Graph Networks for Diagnosis (Jan-Apr 2026) **[MAJOR - NEW]**

### Project 4: Multi-Agent HAO Workflow (Feb-Apr 2026)

### Project 5: Competitive Paper Reproduction (Mar-Apr 2026)

---

## WEEKLY RHYTHM (Suggested Schedule)

### Monday
- Read competitive paper or theory paper (1-2 hours)
- Track B: Domain knowledge work (2-3 hours)

### Wednesday
- Hands-on: Projects or Track A/C implementation (3-4 hours)
- Code/documentation (1 hour)

### Friday
- Integration: Connect projects, reflect (2-3 hours)
- Plan next week (0.5 hours)

### Total: 12-18 hours/week (manageable)

---

## MAJOR MILESTONES

| Milestone | Target Date | What Success Looks Like |
|-----------|-------------|------------------------|
| **Track A complete** | Dec 31, 2025 | Attention + SHAP notebooks ready |
| **ACMG fluency** | Jan 31, 2026 | Can classify variants accurately |
| **Phenotype + network prediction** | Feb 28, 2026 | HPO + network propagation working on validation cases |
| **Competitive deep-dive** | Mar 31, 2026 | Understand DeepRare/AlphaGenome/GenoMAS deeply |
| **HAO workflow** | Apr 30, 2026 | Multi-agent system working on 10+ test cases |
| **Study design prep** | May 31, 2026 | Protocol draft + IRB ready |
| **Portfolio review** | Aug 2026 | 5 projects + notebooks + competitive analysis + network medicine results |
| **Ready for PhD** | Sep 1, 2026 | Can start Aim 1 immediately |

---

## SUCCESS CRITERIA (Sep 2026)

By program start, you should be able to:

- [ ] Explain what ESM2 and your encoder-decoder learn (via attention analysis)
- [ ] Generate clinical explanations for variants (SHAP + counterfactual)
- [ ] Classify variants using ACMG framework independently
- [ ] Navigate HPO ontology and link phenotypes to genes
- [ ] **Build network-based gene prioritization system (new!)**
- [ ] **Understand how network topology predicts phenotypic breadth (new!)**
- [ ] **Apply regulatory motif logic to disease mechanism (new!)**
- [ ] Build end-to-end variant ranking system (phenotype-aware + network-aware)
- [ ] Explain exactly where DeepRare/AlphaGenome/GenoMAS succeed and fail
- [ ] Design and orchestrate multi-agent diagnostic system in HAO
- [ ] Have prospective study protocol draft ready
- [ ] Have 5 completed projects + GitHub portfolio (60+ hours of code)
- [ ] Have 35+ hours of reading notes from competitive papers + network medicine papers

**If all true → Ready to hit ground running on Aim 1**

---

## GitHub Repository Structure

```
pre-phd-genomics/
├── README.md
├── requirements.txt
├── setup.sh
│
├── 01_interpretability/
│   ├── attention_visualization_ESM2.ipynb
│   ├── attention_visualization_encoder_decoder.ipynb
│   ├── shap_variant_ranking.ipynb
│   ├── counterfactual_variants.ipynb
│   └── interpretability_comparison.md
│
├── 02_genomics_domain/
│   ├── acmg_classification_exercise.ipynb
│   ├── hpo_exploration.ipynb
│   ├── annotation_pipeline.py
│   └── gene_phenotype_baseline.ipynb
│
├── 03_network_medicine/
│   ├── ppi_network_analysis.ipynb
│   ├── network_propagation.py
│   ├── network_vs_baseline_comparison.ipynb
│   ├── multi_layer_networks.ipynb
│   └── network_phenotype_correlation.md
│
├── 04_integrated_projects/
│   ├── project1_variant_effect_system.ipynb
│   ├── project2_phenotype_ranking_pipeline.ipynb
│   ├── project3_network_medicine_analysis.ipynb
│   ├── project4_hao_workflow/
│   │   ├── workflow_design.md
│   │   ├── orchestrator.py
│   │   └── agent_implementations.py
│   └── project5_competitive_reproduction.ipynb
│
├── 05_competitive_analysis/
│   ├── deeprare_analysis.md
│   ├── alphagenome_analysis.md
│   ├── genomas_analysis.md
│   └── comparison_matrix.xlsx
│
└── 06_clinical_context/
    ├── diagnostic_odyssey_cases.md
    ├── workflow_integration_design.md
    ├── prospective_study_protocol_draft.md
    └── study_feasibility_notes.md
```

---

## Total Time Investment

- **Track A:** 20-25 hours
- **Track B:** 100-120 hours (expanded from 40-50 to include network medicine)
- **Track C:** 30-40 hours
- **Concurrent projects:** 50-60 hours (added network medicine project)
- **Weekly paper reading:** 20-30 hours
- **Total:** ~220-260 hours over 40 weeks = 5-6.5 hours/week average (very doable)

**Result:** Deeply prepared, competitive foundation for PhD thesis with network medicine expertise

---
