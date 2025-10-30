# Pre-PhD Intensive Study Plan (REVISED)
## Nov 2025 - Sep 2026 (10 Months)
### Optimized for your advanced ML background: Master genomics + clinical context + competitive landscape

---

## YOUR CURRENT STATE (Recalibrated)

**You already know:**
- ✅ Transformers, BERT fundamentals
- ✅ ESM papers, Enformer
- ✅ Working with Evo2 + proprietary dual-llama encoder-decoder (last token bottleneck architecture)
- ✅ Saliency maps (DL course 2 years ago)

**You need to master (Priority order):**
1. ❌ **Genomics domain** - ACMG classification, HPO, phenotype prediction, rare disease workflows
2. ❌ **Clinical context** - Diagnostic odyssey, Mayo workflows, IRB/regulatory
3. ❌ **Competitive landscape** - DeepRare, AlphaGenome, GenoMAS (June-July 2025)
4. ❌ **HAO multi-agent systems** - Microsoft orchestrator platform
5. ❌ **Prospective study design** - Clinical trial methodology
6. ⚠️ **Attention maps + interpretability** - Apply to your models (can do in parallel)

**This plan structure:**
- **Track A (2-3 weeks):** Interpretability quick-wins (attention maps + SHAP on your models)
- **Track B (8-9 weeks):** Genomics domain + clinical (heavy focus)
- **Track C (6-8 weeks):** Competitive analysis + HAO + study design
- **Concurrent:** Weekly competitive papers + integrated projects

---

## TRACK A: INTERPRETABILITY ON YOUR MODELS (Nov-Dec 2025 - 2-3 weeks total)

### Phase 1: Attention Maps on Evo2 & Your Encoder-Decoder (Nov 2025 - 1.5 weeks)

**Goal:** Extract + visualize attention patterns; understand what your bottleneck learns

**Minimal reading (you know transformers):**
1. **Avsec et al. 2021 (Enformer) Sections 3-4 only** - 1-2 hours (refresher on bio-focused attention)
2. **Olah et al. 2017 "Feature Visualization" blog** - 1 hour (making attention interpretable to non-ML)

**Hands-On (Core work):**

```python
# Project: Attention Visualization Comparison

# 1. Evo2 attention extraction (3-4 hours)
- Load Evo2 model on HuggingFace
- For 10 pathogenic variants: 500bp genomic window
- Extract attention matrices from all layers
- Aggregate across heads: attention_scores[layer][position_i][position_j]
- Visualize: Heatmap showing which positions attend to regulatory regions?
- Validate: Compare to known TFBS sites, CpG islands

# 2. Your dual-llama encoder-decoder (4-5 hours)
- Encoder: DNA sequence attention patterns
- Decoder: Cross-attention to bottleneck embedding
- Key analysis: What does the bottleneck compress?
  - Does it capture protein folding signals?
  - Does it preserve phenotype-relevant information?
  - Can you reconstruct biological signal from bottleneck alone?

# 3. Comparative analysis (2 hours)
- Side-by-side: Evo2 vs your model attention on same variants
- Interpretation: Where do they agree/disagree?
- Insight: Why does your bottleneck work?
```

**Output:**
- Jupyter notebook: "Attention Analysis in Genomic Foundation Models"
- 10-12 publication-quality heatmaps
- Document: "Bottleneck Architecture Analysis" (500 words)

---

### Phase 2: SHAP + LIME + Counterfactual (Dec 2025 - 1.5 weeks)

**Goal:** Advanced explanations on your variant ranking system

**Reading (targeted, quick):**
1. **Lundberg & Lee 2017 SHAP** - 1.5 hours (you know gradients-based methods)
2. **Goyal et al. 2019 Counterfactual** - 1 hour (quick read on what-if scenarios)

**Hands-On (Core work):**

```python
# Project: Variant Ranking Interpretability

# 1. SHAP on pathogenicity prediction (3 hours)
- Model: Your encoder-decoder predicting pathogenic/benign
- Dataset: 100 variants (50 pathogenic ClinVar, 50 benign gnomAD)
- SHAP analysis per variant:
  - Which nucleotide positions drive pathogenicity score?
  - Do top-ranked positions align with known functional domains?
  - SHAP visualization: force plots + bar plots
- Validation: Compare to known pathogenic regions in literature

# 2. LIME local explanations (2-3 hours)
- For 10 diverse variants (5 pathogenic, 5 benign)
- Generate LIME local surrogate models
- Output: "For this specific variant, these features matter most"
- Test with clinician (ask Eric Klee lab): Is this interpretable?

# 3. Counterfactual variants (3-4 hours)
- Pathogenic example: "BRCA1 p.R1699W"
- Query: "What single amino acid subs flip this to benign?"
- Generate top 20 counterfactuals ranked by confidence
- Validate: Are these realistic? What do they teach?
  - E.g., "R→K keeps charge, more likely benign" = mechanistic insight
```

**Output:**
- Notebook: "SHAP Analysis of Variant Pathogenicity"
- Notebook: "Counterfactual Variant Generation"
- Comparison document: SHAP vs LIME vs Attention (700 words)

---

### Track A Summary (By Dec 31, 2025):
- ✅ Understand what Evo2 and your model learn
- ✅ Can generate clinical explanations for variants
- ✅ Ready to integrate into Aim 1 methodology
- **Time investment: ~20-25 hours**

---

## TRACK B: GENOMICS DOMAIN & CLINICAL INTEGRATION (Dec 2025 - Apr 2026 - 8-9 weeks)

### Phase 1: ACMG + Variant Interpretation Fundamentals (Dec 2025 - 2 weeks)

**Goal:** Fluency in how clinicians classify variants; this is your target framework

**Reading:**
1. **Richards et al. 2015** - "Standards and Guidelines for the Interpretation of Sequence Variants" (Nature Reviews Genetics)
   - Focus: ACMG classification framework (PVS1, PS1-4, PM1-6, PP1-5, BS1-4, BP1-7)
   - Time: 3 hours
   - Key: Your model must align with / improve upon this framework

2. **Rehm et al. 2021** - "ClinGen Guidelines for Reporting Sequence Variants" (Clinical Chemistry)
   - Focus: Practical variant annotation, evidence codes
   - Time: 2 hours

3. **Quick reference resources:**
   - Variant Effect Predictor (VEP) documentation
   - InterPro protein domain structure
   - Time: 2 hours exploring

**Hands-On (Core work):**

```python
# Exercise 1: Manual ACMG Classification (4 hours)
- Pick 20 pathogenic variants from ClinVar (diverse rare diseases)
- For each: Manually classify using ACMG criteria
- Document your reasoning for each evidence code
- Compare your classification to ClinVar classification
- Accuracy target: 80%+

# Exercise 2: Build annotation pipeline (3-4 hours)
- Input: VCF file with 100 variants
- Pipeline:
  * Functional impact (SIFT, PolyPhen predictions)
  * Frequency (gnomAD filtering: <0.1% rare disease variants)
  * Domain annotation (InterPro)
  * ClinVar cross-reference
  * Population frequency stratification
- Output: Annotated variants with ACMG classification
- Code: Python script, reproducible
```

**Output:**
- Spreadsheet: Manual variant classifications (20 examples with reasoning)
- Python script: Variant annotation pipeline
- Document: "ACMG Classification Cheat Sheet" (2-3 pages for reference)

---

### Phase 2: Phenotype Prediction & Gene-to-Phenotype Mapping (Jan 2026 - 2.5 weeks)

**Goal:** Understand how to link variants → genes → phenotypes; integrate HPO ontology

**Reading:**
1. **Köhler et al. 2019** - "The Human Phenotype Ontology in 2021" (NAR)
   - Focus: HPO structure, phenotype hierarchies, disease associations
   - Time: 2 hours

2. **Luo et al. 2022** - "IMPPROVE: Imputing Phenotypes for Rare Disease Diagnosis via Integration of Genotype and Deep Learning" (BMC Genomics, Jan 2025)
   - **This is your key competitive comparison paper**
   - Focus: Phenotype prediction from variants, RNA-seq integration, isoform-level effects
   - Time: 3-4 hours (deep read)

3. **Oren et al. 2020** - "Sparse Modeling of Cell-Type-Specific Gene Expression" (Nature Genetics)
   - Focus: Tissue-specific variant effects
   - Time: 1.5 hours

**Hands-On (Core work):**

```python
# Exercise 1: HPO exploration (2-3 hours)
- Download HPO database (free)
- For 3 rare diseases (e.g., BRCA1-associated cancer, Marfan, cystic fibrosis):
  * Map disease → phenotype terms
  * Build HPO hierarchy visualization
  * Identify: Are phenotypes clustered? Predictable from genotype?

# Exercise 2: Gene-phenotype linking (2-3 hours)
- Dataset: 100 rare disease patients (genotype + phenotype)
- Model baseline: Simple logistic regression
  * Input: One-hot encoded gene variants
  * Output: HPO term (binary classification)
- Evaluate: AUC, precision@5
- Compare to random baseline

# Exercise 3: Understand IMPPROVE approach (2-3 hours)
- Read their methodology for isoform-level phenotype prediction
- Replicate their simple baseline on public data
- Document: Key insights for your Aim 1
```

**Output:**
- HPO exploration notebook
- Gene-phenotype baseline model (logistic regression)
- Document: "IMPPROVE Analysis & Implications for Your Aim 1" (1000 words)

---

### Phase 3: Rare Disease Diagnostic Workflows & Clinical Context (Feb 2026 - 2 weeks)

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
     * Focus: Impact, diagnostic delay costs, patient burden
     * Time: 1 hour

**Hands-On (Core work):**

```python
# Exercise 1: Diagnostic odyssey case studies (3-4 hours)
- Find 5 published rare disease cases (literature/blogs)
- For each, construct timeline:
  * Symptom onset
  * Referrals to specialists
  * Tests ordered
  * Time to diagnosis
- Analyze: Where would AI help most?
- Document: Timeline diagrams for each case

# Exercise 2: Workflow integration design (2-3 hours)
- Interview Eric Klee lab or Mayo genomics staff (if possible):
  * Current diagnostic workflow (what they actually do)
  * Current bottlenecks
  * Where AI could fit
- Diagram: 
  * Current state (no AI)
  * Future state (with your AI system)
  * Integration points, data flows, decision handoffs

# Exercise 3: Identify prospective study barriers (2 hours)
- Brainstorm: What could go wrong in prospective trial?
  * Patient enrollment bottlenecks
  * Data quality issues
  * Clinician buy-in challenges
- Document: Risk mitigation strategies for each
```

**Output:**
- Case study document (5 diagnostic odysseys with timelines)
- Workflow diagram (before/after integration)
- Risk assessment document (1-2 pages)

---

### Phase 4: HPO + ACMG Integration & Phenotype-Driven Ranking (Feb-Mar 2026 - 2 weeks)

**Goal:** Build system combining ACMG classification + HPO phenotypes for variant ranking

**Project: Phenotype-Driven Variant Ranking System**

```python
# Integrated project (8-10 hours total)

# Input: 
#   - Patient HPO terms (symptoms, findings)
#   - Exome/WGS data (rare variants)

# Pipeline:
# 1. Gene prioritization (HPO → gene ranking)
#    - Which genes explain patient phenotypes?
#    - Use gene-disease associations in OMIM/ClinGen
#
# 2. Variant prioritization (variant → pathogenicity)
#    - ACMG classification
#    - Frequency filtering (gnomAD)
#    - Conservation scores
#
# 3. Phenotype-gene-variant integration
#    - Rank variants in phenotype-relevant genes highest
#    - Score = ACMG classification × gene_relevance × conservation
#
# 4. Output: Top 10 candidates with explanations

# Test on 5 published rare disease cases
# Metric: Did true diagnosis rank #1? Top 5?
```

**Output:**
- Notebook: Full phenotype-driven ranking pipeline
- Validation on 5 test cases
- Document: "Integration of Phenotypes in Variant Interpretation" (800 words)

---

### Track B Summary (By Mar 31, 2026):
- ✅ ACMG fluency (can classify variants independently)
- ✅ HPO ontology understanding (can navigate disease-phenotype relationships)
- ✅ Understand rare disease diagnostic process (deep)
- ✅ Know Mayo's workflow and integration points
- ✅ Built integrated phenotype-driven ranking system
- **Time investment: ~40-50 hours**

---

## TRACK C: COMPETITIVE ANALYSIS, HAO INTEGRATION, STUDY DESIGN (Jan-Aug 2026 - ongoing)

### Phase 1: Deep Competitive Analysis (Jan-Feb 2026 - concurrent with Track B)

**Goal:** Understand exactly what DeepRare, AlphaGenome, GenoMAS do; where your differentiation comes from

**Core Papers (Deep-dive reads):**

1. **DeepRare (arXiv June 25, 2025)**
   - Multi-agent system for rare disease diagnosis
   - Read: Full paper + appendix
   - Time: 4-5 hours
   - Key questions:
     * What agents do they use? How do they communicate?
     * What's their variant ranking method?
     * What clinical validation have they done?
     * What's the prospective claim?

2. **AlphaGenome (bioRxiv June 25, 2025)**
   - Foundation model for regulatory variants
   - Read: Full paper sections 1-3, skim methods
   - Time: 2-3 hours
   - Key questions:
     * How do they handle regulatory regions?
     * What's their training data?
     * Performance benchmarks vs Exomiser/SIFT?

3. **GenoMAS (arXiv July 28, 2025)**
   - Multi-agent genomic analysis system
   - Read: Full paper
   - Time: 3-4 hours
   - Key questions:
     * How is this different from DeepRare?
     * Agent architecture and reasoning?

**Hands-On (Critical):**

```python
# For EACH competitive paper:

# 1. Create summary document (1-2 hours each)
#    - What problem does it solve?
#    - Technical approach (1 paragraph)
#    - Key innovations vs prior work
#    - Benchmarks / results
#    - Prospective validation? Clinical deployment?
#    - Where does YOUR work differentiate?

# 2. Attempt to reproduce (2-4 hours each, if code available)
#    - Get paper code from GitHub (if released)
#    - Run on toy dataset
#    - Can you match their reported numbers?
#    - What works? What doesn't?

# 3. Build comparison matrix
#    - Row: Each competitive system
#    - Col: Technical features, validation type, clinical deployment, interpretability
#    - Your row: Your planned Aim 1 approach
```

**Output:**
- DeepRare summary + analysis (3-4 pages + code repo link)
- AlphaGenome summary + analysis (2-3 pages)
- GenoMAS summary + analysis (3-4 pages)
- Comparison matrix: All competitive systems vs your approach
- Document: "Where You Differentiate" (1 page for thesis framing)

---

### Phase 2: HAO Multi-Agent System Integration (Feb-Mar 2026 - 2-3 weeks)

**Goal:** Learn Microsoft HAO platform; design multi-agent workflow for your prospective trial

**Reading (Minimal—platform-focused):**
1. **Microsoft HAO documentation** (2-3 hours exploration)
2. **Example HAO workflows** (if available, 1-2 hours)

**Hands-On (Core work):**

```python
# Project: Multi-Agent Diagnostic Workflow in HAO

# Design (no code yet): 3-4 hours
# - Agent 1: Patient History Analyzer
#   * Input: Clinical notes, symptom list
#   * Output: HPO term predictions
#
# - Agent 2: Genomics Analyzer  
#   * Input: VCF file, exome data
#   * Output: Rare variants + functional predictions
#
# - Agent 3: Gene Ranker
#   * Input: HPO terms + variants
#   * Output: Likely disease genes (ranked)
#
# - Agent 4: Variant Validator
#   * Input: Top variant candidates
#   * Output: ACMG classification + confidence + explanation
#
# - Agent 5: Diagnosis Synthesizer
#   * Input: Gene ranking + variant classifications
#   * Output: Top 5 diagnostic hypotheses with confidence scores

# Implementation in HAO (4-6 hours):
# - Set up Azure AI Foundry account
# - Create agent workflow (connect your models/tools)
# - Test on 3 simulated patient cases
# - Document: How agents communicate, what works/doesn't

# Validation (2-3 hours):
# - Compare to baseline (no agents, just pipeline)
# - Does agent orchestration help vs pipeline?
# - Where does multi-step reasoning add value?
```

**Output:**
- HAO workflow design document (2-3 pages + diagrams)
- Working HAO implementation with 3 test cases
- Document: "Multi-Agent Reasoning for Rare Disease Diagnosis" (1000 words)

---

### Phase 3: Prospective Clinical Study Design (Mar-Apr 2026 - ongoing prep)

**Goal:** Prepare research proposal for prospective validation study

**Reading:**
1. **Your PhD thesis documents** (review retrospective plan)
2. **Clinical trial design primers:**
   - Consult: Mayo IRB guidelines (if available)
   - Papers on prospective genetic studies

**Hands-On (Critical for thesis):**

```python
# Document 1: Prospective Study Protocol (5-10 pages)
# - Study design (enrollment criteria, sample size justification)
# - Outcomes (diagnostic yield, time-to-diagnosis, health outcomes)
# - Timeline (enrollment phases)
# - Data collection forms
# - Statistical analysis plan
# - Regulatory/IRB considerations

# Document 2: IRB Submission Preparation (3-5 pages)
# - Human subjects research checklist
# - Risk/benefit analysis
# - Data privacy/security
# - Informed consent form outline
# - Study protocol template

# Discussion with Eric Klee (1-2 hours)
# - Realistic enrollment rates at Mayo?
# - Current patient population in Undiagnosed Diseases Program?
# - What's feasible in prospective study?
# - Partnership opportunities (multi-site)?
```

**Output:**
- Complete prospective study protocol draft (ready for IRB)
- Enrollment timeline projections
- Statistical power analysis
- Meeting notes with Eric Klee on feasibility

---

### Track C Integration: Weekly Competitive Paper Reading (Ongoing Jan-Sep 2026)

**Format:** 1 competitive paper per week, lightweight tracking

| Month | Paper | Focus |
|-------|-------|-------|
| Jan | DeepRare | Multi-agent architecture |
| Jan-Feb | AlphaGenome | Regulatory variants |
| Feb | GenoMAS | Multi-agent reasoning |
| Feb-Mar | Follow-up papers from above | Refinements, benchmarks |
| Mar | Exomiser recent updates | Baseline comparison |
| Mar-Apr | PhenoLinker / similar | Phenotype prediction |
| Apr-May | IMPPROVE deep dive | Your direct comparison |
| May-Jun | Other competitive papers | Catch-up reading |
| Jun-Aug | Your deep-dive topics | Specialized knowledge |

---

## INTEGRATED PROJECTS (Concurrent: Nov 2025 - Sep 2026)

### Project 1: Variant Effect Prediction System (Nov-Jan 2026)
- **Combines:** Your encoder-decoder + attention maps + SHAP
- **Input:** Variant (DNA sequence)
- **Output:** Pathogenicity score + attention heatmap + SHAP explanation
- **Deliverable:** Reproducible notebook, GitHub repo

---

### Project 2: Phenotype-Driven Variant Ranking (Jan-Feb 2026)
- **Combines:** HPO ontology + ACMG classification + gene prioritization
- **Input:** Patient phenotypes + genome
- **Output:** Top 10 diagnostic candidates with confidence
- **Test:** 5 published rare disease cases
- **Deliverable:** Full pipeline code + validation report

---

### Project 3: Multi-Agent HAO Workflow (Feb-Mar 2026)
- **Combines:** All agents above orchestrated in HAO
- **Input:** Patient clinical data + genomics
- **Output:** Diagnosis + confidence + reasoning chain
- **Test:** Simulated patient scenarios
- **Deliverable:** Working HAO implementation + documentation

---

### Project 4: Competitive Paper Reproduction (Mar-Apr 2026)
- **Pick:** DeepRare OR AlphaGenome (choose one that excites you)
- **Goal:** Can you reproduce their results?
- **Output:** Reproduction report + comparison to your approach

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

### Total: 12-17 hours/week (manageable)

---

## MAJOR MILESTONES

| Milestone | Target Date | What Success Looks Like |
|-----------|-------------|------------------------|
| **Track A complete** | Dec 31, 2025 | Attention + SHAP notebooks ready |
| **ACMG fluency** | Jan 31, 2026 | Can classify variants accurately |
| **Competitive deep-dive** | Feb 28, 2026 | Understand DeepRare/AlphaGenome/GenoMAS |
| **Phenotype ranking system** | Mar 31, 2026 | Working end-to-end pipeline |
| **HAO workflow** | Apr 30, 2026 | Multi-agent system prototyped |
| **Study design prep** | May 31, 2026 | Protocol draft + IRB ready |
| **Portfolio review** | Aug 2026 | 4 projects + notebooks + competitive analysis |
| **Ready for PhD** | Sep 1, 2026 | Can start Aim 1 immediately |

---

## SUCCESS CRITERIA (Sep 2026)

By program start, you should be able to:

- [ ] Explain what Evo2 and your encoder-decoder learn (via attention analysis)
- [ ] Generate clinical explanations for variants (SHAP + counterfactual)
- [ ] Classify variants using ACMG framework independently
- [ ] Navigate HPO ontology and link phenotypes to genes
- [ ] Build end-to-end variant ranking system (phenotype-aware)
- [ ] Explain exactly where DeepRare/AlphaGenome/GenoMAS succeed and fail
- [ ] Design and orchestrate multi-agent diagnostic system in HAO
- [ ] Have prospective study protocol draft ready
- [ ] Have 4 completed projects + GitHub portfolio (50+ hours of code)
- [ ] Have 30+ hours of reading notes from competitive papers

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
│   ├── attention_visualization_evo2.ipynb
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
├── 03_integrated_projects/
│   ├── project1_variant_effect_system.ipynb
│   ├── project2_phenotype_ranking_pipeline.ipynb
│   ├── project3_hao_workflow/
│   │   ├── workflow_design.md
│   │   └── implementation.py
│   └── project4_competitive_reproduction.ipynb
│
├── 04_competitive_analysis/
│   ├── deeprare_analysis.md
│   ├── alphagenome_analysis.md
│   ├── genomas_analysis.md
│   └── comparison_matrix.xlsx
│
└── 05_clinical_context/
    ├── diagnostic_odyssey_cases.md
    ├── workflow_integration_design.md
    ├── prospective_study_protocol_draft.md
    └── study_feasibility_notes.md
```

---

## Total Time Investment

- **Track A:** 20-25 hours
- **Track B:** 40-50 hours
- **Track C:** 30-40 hours
- **Concurrent projects:** 40-50 hours
- **Weekly paper reading:** 20-30 hours
- **Total:** ~170-200 hours over 40 weeks = 4-5 hours/week average (very doable)

**Result:** Deeply prepared, competitive foundation for PhD thesis

---

## Final Notes

1. **You're in a unique position**: You have state-of-the-art models (Evo2 + your encoder-decoder) + you're embedded at Mayo + you understand ML deeply. Focus on domain expertise now.

2. **Competitive window is narrow**: DeepRare/AlphaGenome/GenoMAS published June-July 2025. Understanding them deeply by March 2026 is critical. Your differentiation will come from prospective validation + clinical context + interpretability, NOT from more sophisticated models.

3. **HAO is your leverage**: Multi-agent systems for clinical reasoning are hot. By Sep 2026, you should be comfortable designing complex workflows. This will differentiate you from pure ML candidates.

4. **Study design matters**: The difference between a good PhD and great PhD is often execution of prospective validation. Spend time on protocol now so you don't waste time during PhD.

5. **Code is portfolio**: Every notebook should be publication-ready. Clean code, good documentation, reproducible. This impresses thesis committees.

6. **Teach as you learn**: Write blog posts (even for yourself). Teaching solidifies understanding.

---

**You're ready. Let's execute this.**
