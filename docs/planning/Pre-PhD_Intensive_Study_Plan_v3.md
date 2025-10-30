# Pre-PhD Intensive Study Plan (REVISED - WITH SYSTEMS BIOLOGY)
## Nov 2025 - Sep 2026 (10 Months)
### Optimized for your advanced ML background: Master genomics + clinical context + network medicine + competitive landscape

---

## YOUR CURRENT STATE (Recalibrated)

**You already know:**
- ✅ Transformers, BERT fundamentals
- ✅ ESM papers, Enformer
- ✅ Working with Evo2 + proprietary dual-llama encoder-decoder (last token bottleneck architecture)
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

## TRACK B: GENOMICS + NETWORK MEDICINE (Dec 2025 - Apr 2026 - 10-11 weeks)

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

### Phase 2: Phenotype Prediction & Gene-to-Phenotype Mapping via Networks (Jan 2026 - 4 weeks) **[EXPANDED WITH SYSTEMS BIOLOGY]**

**Goal:** Link variants → genes → phenotypes via HPO + biological network topology; understand how network position predicts phenotypic breadth and disease mechanism

**Reading (HPO & phenotype prediction - Week 1):**
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

**Reading (Network medicine & systems biology - Weeks 2-3):**
4. **Guney et al. 2016** - "Network-based in silico drug efficacy screening" (Nature Communications)
   - Focus: Network proximity kernel; shortest-path metrics for disease gene prioritization
   - Time: 2 hours

5. **Menche et al. 2015** - "Disease networks. Uncovering disease-disease relationships through the incomplete human interactome" (Nature Communications)
   - Focus: Network modularity and disease modules; how network architecture predicts phenotypic pleiotropy
   - Time: 1.5 hours

6. **Barabási et al. 2011** - "Network medicine: a network-based approach to human disease" (Nature Reviews Genetics)
   - Focus: Hub genes vs. peripheral genes; network degree predicts phenotypic breadth; scale-free topology implications
   - Time: 1.5 hours

**Hands-On (Part A: HPO + Baseline - Week 1, 6-8 hours):**

```python
# Exercise 1: HPO exploration (2-3 hours)
- Download HPO database (free)
- For 3 rare diseases (e.g., BRCA1-associated cancer, Marfan, cystic fibrosis):
  * Map disease → phenotype terms
  * Build HPO hierarchy visualization
  * Identify: Are phenotypes clustered? Predictable from genotype?

# Exercise 2: Gene-phenotype linking baseline (2-3 hours)
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

**Hands-On (Part B: Network-based phenotype prediction - Weeks 2-4, 18-22 hours):**

```python
# Exercise 4: Patient-scale heterogeneous graph construction (5-6 hours)
# ** Core for Project 3: Patient-scale GNN **
- Dataset: 100-150 rare disease patients (genotype + phenotype + network)
- For each patient:
  * Extract variant genes from VCF
  * Query STRING/Reactome/ENCODE for: PPI, TF regulation, pathways, co-expression
  * Build heterogeneous subgraph: K=50 genes (mutated + 1-2 hop neighbors)
  * Assign node features: pathogenicity (your encoder-decoder) + expression + conservation + centrality
  * Create edge tensors per type: PPI, TF, pathway, co-expr, homology
  * Verify: ~100-150 PyTorch Geometric heterogeneous graphs ready for GNN training

# Exercise 5: Network propagation baseline (4-5 hours)
# ** Baseline comparison for Project 3 **
- Implement heat diffusion kernel on PPI network (Guney et al. approach)
- Input: Variant in gene X on patient with phenotype profile P
- Algorithm: Propagate variant "signal" through network starting at X
- Output: Top 20 genes by diffusion score (most likely to contribute to P)
- Test on 5 known Mendelian disease cases: does network proximity improve gene ranking?
- Comparison: Network + ACMG vs. ACMG alone (benchmark against baseline)
- Save this as Baseline 2 for Project 3 evaluation

# Exercise 6: Simple GNN on global PPI (3-4 hours)
# ** Baseline comparison for Project 3 **
- Train single-layer GCN on shared PPI network (no edge type distinction)
- Compare performance to: ACMG alone, heat diffusion kernel, heterogeneous GNN (later)
- Save as Baseline 3 for Project 3

# Exercise 7: Multi-layer network & pathway logic (3-4 hours)
- For 5-10 genes from Exercise 4:
  * Map upstream regulators (TFs that control them)
  * Map downstream targets (genes they regulate)
  * Identify regulatory motifs (feedforward loops, feedback, etc.)
  * Hypothesis: Do genes in coherent FFLs exhibit coordinated phenotypes?
  * Example: Gene disruption in coherent FFL → oscillation loss → specific phenotype signature
- Validate: Does regulatory logic align with disease mechanism in literature?
```

**Output:**
- HPO exploration notebook
- Gene-phenotype baseline model (logistic regression)
- Document: "IMPPROVE Analysis & Implications for Your Aim 1" (1000 words)
- PPI network analysis notebook with degree vs. phenotype analysis
- Network propagation implementation (Python module) + validation on 5 cases
- Multi-layer network visualization + mechanistic interpretation document (1500 words)
- Comparison table: ACMG alone vs. ACMG + network propagation on test cases

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

# Exercise 3: Phenotype-driven triage (2-3 hours)
- Build clinical decision support mockup:
  * Input: Patient phenotypes (from EHR or clinical form)
  * Output: Top 5 genes to sequence + confidence
  * Integrate: ACMG + network propagation from Phase 2
- Test interface with Eric or clinician if possible
```

**Output:**
- Case study timeline analysis document (1000 words)
- Workflow integration diagrams (current vs. future state)
- Clinical decision support prototype (Python notebook with interactive UI concept)
- Meeting notes + actionable feedback from Eric Klee lab

---

### Phase 4: Competitive Deep-Dive & Multi-Agent Design (Mar 2026 - 2 weeks)

**Goal:** Master competitive systems; understand where you differentiate; design HAO workflow

**Reading:**
1. **DeepRare paper + supplementary** (preprint or published)
   - Focus: Multi-agent architecture, reasoning chain, validation approach
   - Time: 3 hours

2. **AlphaGenome paper + supplementary**
   - Focus: Regulatory variant handling, sequence model, clinical integration
   - Time: 3 hours

3. **GenoMAS paper + supplementary**
   - Focus: Multi-agent reasoning, task decomposition, knowledge graphs
   - Time: 3 hours

4. **Recent HAO literature:**
   - **Sap et al. 2022** or similar on multi-agent reasoning
   - Time: 1.5 hours

**Hands-On (Core work):**

```python
# Exercise 1: Competitive system reproduction (4-5 hours)
- Pick ONE: DeepRare or AlphaGenome
- Attempt to reproduce key results on 10 validation cases
- Document:
  * What works? (architecture, heuristics, integration points)
  * What doesn't? (where do they fail or lack mechanism?)
  * Where do they differ from IMPPROVE?

# Exercise 2: Comparison matrix (2-3 hours)
- Build detailed comparison: Your approach vs. DeepRare vs. AlphaGenome vs. GenoMAS
  * Input processing
  * Reasoning chain
  * Network integration (if any)
  * Clinical validation
  * Interpretability
- Identify: Your unique angles

# Exercise 3: HAO workflow design (3-4 hours)
- Design multi-agent system orchestration:
  * Agent 1: Variant pathogenicity (your encoder-decoder + attention + SHAP)
  * Agent 2: Gene prioritization (ACMG + network propagation)
  * Agent 3: Phenotype-genotype linking (HPO + network logic)
  * Agent 4: Clinical reasoning (diagnostic odyssey patterns)
  * Orchestrator: Coordinates, aggregates confidence, returns reasoning chain
- Pseudocode or flowchart
```

**Output:**
- Reproduction report (1500 words)
- Competitive comparison matrix (spreadsheet)
- HAO workflow design document (2000 words)
- Pseudo-implementation of orchestrator logic

---

## TRACK B Summary (By Apr 30, 2026):
- ✅ ACMG fluency + variant annotation pipeline
- ✅ HPO navigation + baseline phenotype prediction
- ✅ **Network medicine:** PPI networks, propagation, multi-layer logic
- ✅ Understand diagnostic workflow at Mayo
- ✅ Deep competitive understanding (DeepRare, AlphaGenome, GenoMAS)
- ✅ HAO workflow design
- **Time investment: ~100-120 hours**

---

## TRACK C: HAO IMPLEMENTATION + STUDY DESIGN (Apr 2026 - May 2026 - 6-8 weeks)

### Phase 1: HAO Multi-Agent System Implementation (Apr 2026 - 3 weeks)

**Goal:** Build working orchestrator platform (using Microsoft HAO or custom framework)

**Reading:**
1. **Microsoft HAO documentation** (if using that platform)
   - Time: 2-3 hours

2. **LLM-based reasoning papers (optional if using Claude/GPT agents):**
   - **Wei et al. 2022 Chain-of-Thought**
   - Time: 1 hour

**Hands-On (Critical for thesis):**

```python
# Project: Multi-Agent Diagnostic System

# 1. Agent implementation (10-12 hours)
- Variant pathogenicity agent:
  * Input: DNA sequence + variant location
  * Output: Pathogenicity score + attention heatmap + SHAP explanation
  
- Gene prioritization agent:
  * Input: Variant, patient phenotypes, background genetics
  * Output: Top 10 genes ranked by combined score (ACMG + network propagation)
  
- Phenotype linking agent:
  * Input: Gene list, patient phenotypes
  * Output: Confidence of gene-phenotype associations + pathway explanations
  
- Clinical reasoning agent:
  * Input: Patient history, phenotypes, genetics
  * Output: Diagnostic confidence, next-step recommendations

# 2. Orchestrator implementation (8-10 hours)
- Coordinate agents in sequence or parallel
- Aggregate confidence scores
- Generate reasoning chain (interpretable output for clinician)
- Error handling & fallback strategies

# 3. End-to-end testing (4-5 hours)
- Simulate 10 patient scenarios (diverse rare diseases)
- For each: Does system reach correct diagnosis?
- Document reasoning chain transparency
- Benchmark against baseline (ACMG alone)
```

**Output:**
- Fully working multi-agent system
- Reasoningchain documentation for 10 test cases
- System architecture document (technical + clinical motivation)

---

### Phase 2: Prospective Study Design & IRB Preparation (May 2026 - 2-3 weeks)

**Goal:** Design realistic prospective validation; prepare for regulatory review

**Reading:**
1. **FDA guidance on software as medical device (SaMD)**
   - Time: 1.5 hours

2. **Clinical trial design papers:**
   - **Friedman et al. 2015** - "Fundamentals of Clinical Trials" (excerpt on study design)
   - Time: 1 hour

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
- **Combines:** HPO ontology + ACMG classification + gene prioritization + **network propagation**
- **Input:** Patient phenotypes + genome
- **Output:** Top 10 diagnostic candidates with confidence
- **Test:** 5 published rare disease cases
- **Deliverable:** Full pipeline code + validation report

---

### Project 3: Patient-Scale Heterogeneous Graph Networks for Diagnosis (Jan-Apr 2026) **[MAJOR - NEW]**
- **Novel approach:** Treat each patient as a heterogeneous multigraph (PPI, transcriptional regulation, metabolic pathways, co-expression)
- **Core method:** Heterogeneous GNN (R-GCN or HAN) to learn patient-level embeddings
- **Task:** Graph classification for diagnostic prediction (rare disease category) + phenotype prediction
- **Comparison:** Heterogeneous GNN vs. network propagation vs. ACMG alone vs. simple GNN
- **Interpretability:** Attention extraction to identify disease-driving subgraph per patient
- **Validation:** Edge-type ablation, mechanistic case studies, phenotype-network correlation analysis
- **Deliverable:** Complete GNN pipeline (training + evaluation + interpretation), case studies, competitive positioning document
- **Publication angle:** "Network medicine insights via heterogeneous graph embeddings captures phenotypic pleiotropy"

---

### Project 4: Multi-Agent HAO Workflow (Feb-Apr 2026)
- **Combines:** All agents above orchestrated in HAO
- **Input:** Patient clinical data + genomics
- **Output:** Diagnosis + confidence + reasoning chain
- **Test:** Simulated patient scenarios (10-15 diverse cases)
- **Deliverable:** Working HAO implementation + documentation + architecture diagram

---

### Project 5: Competitive Paper Reproduction (Mar-Apr 2026)
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

- [ ] Explain what Evo2 and your encoder-decoder learn (via attention analysis)
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

## Final Notes

1. **You're in a unique position**: You have state-of-the-art models (Evo2 + your encoder-decoder) + you're embedded at Mayo + you understand ML deeply + **you have network science fundamentals**. Your differentiation is network-aware