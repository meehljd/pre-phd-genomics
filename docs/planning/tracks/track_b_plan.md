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