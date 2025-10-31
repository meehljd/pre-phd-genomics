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