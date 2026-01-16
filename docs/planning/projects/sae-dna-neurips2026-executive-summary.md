# SAE DNA NeurIPS 2026: Executive Summary

**One-Sentence Pitch:** We're applying sparse autoencoders to DNA foundation models to discover interpretable disease mechanisms for IBD, enabling faithful variant-level explanations that clinicians can trust.

---

## The Problem

DNA foundation models (like Nucleotide Transformer, Evo-2) can predict disease risk from genomic sequences, but they're black boxes. Clinicians can't use a model that says "this variant is pathogenic" without explaining *why*. The ACMG-AMP variant interpretation framework requires mechanistic evidence (PP3/BP4 criteria), not just a score.

**Current interpretability methods (attention maps, gradients) are "plausible but not faithful"** - they show biologically reasonable patterns but don't prove the model actually uses them. This creates false confidence: a feature highlighted by attention might not matter to the prediction at all.

---

## Our Solution

**Sparse autoencoders (SAEs)** decompose model representations into interpretable, monosemantic features. Each SAE feature activates on specific biological patterns (e.g., NOD2 LRR domains, HNF4A binding sites). We validate that these features are *computationally necessary* - ablating them degrades prediction performance.

**Why This Works:**
1. **Sparsity:** Each feature represents one concept (unlike neurons that mix multiple concepts)
2. **Faithfulness:** Ablation experiments prove the model relies on features
3. **Biological Validation:** Features align with known disease genes (NOD2, IL23R, ATG16L1 for IBD)

**Why DNA is Hard:**
- Protein SAEs (InterPLM, Nature Methods 2025) had 1kb context; DNA needs 6-131kb
- Proteins have rich annotations (UniProt, PDB); DNA regulatory elements are sparse (JASPAR ~700 motifs)
- Phenotype prediction requires multi-gene aggregation, not just single-protein analysis

---

## Our Approach

**4-Phase Plan (18 weeks, Jan-May 2026):**

### Phase 1 (Weeks 1-6): Train SAEs on DNA Foundation Models
- Train 32 SAEs (one per Nucleotide Transformer layer) on Helix 100k IBD cohort
- Target: 100-200 interpretable features per layer (1024-dim embeddings → 4096-dim SAE)
- Validate: Reconstruction R² ≥0.95, sparsity L0 ≤20 active features/sequence

### Phase 2 (Weeks 7-10): Biological Annotation
- Annotate features using JASPAR motifs, ENCODE regulatory elements, ClinVar variants, GWAS loci
- Target: ≥30% features match known biology (p < 0.01)
- Identify IBD-specific features: NOD2 LRR domains, IL23R protective variants, ATG16L1 autophagy

### Phase 3 (Weeks 11-14): Computational Validation
- Ablation experiments: Remove features, measure AUROC drop
- Target: ≥25 features show computational necessity (ΔAUROC ≥0.03)
- Biological validation: GWAS enrichment (OR ≥3), ClinVar pathogenicity prediction (AUROC ≥0.65)

### Phase 4 (Weeks 15-18): Paper Writing
- NeurIPS 2026 submission (May deadline)
- 6-8 publication-quality figures, 8-9 pages main text
- Code release: SAE catalog, training pipeline

---

## Impact

### Scientific Contribution
1. **First SAE application to DNA foundation models** (InterPLM did proteins)
2. **Novel findings about DNA representations:** Layer specialization (early=motifs, middle=variant effects, late=gene functions), superposition in DNA vs. proteins, phenotype-specific circuits
3. **Methodology:** Multi-scale interpretability (variant → gene → patient), phenotype-conditioned feature discovery

### Clinical Utility
1. **ACMG-AMP Evidence:** SAE features provide mechanistic PP3/BP4 evidence for variant classification
2. **Rare Disease Diagnosis:** Explain *why* a novel variant is predicted pathogenic (e.g., "disrupts NOD2 LRR domain feature 1247")
3. **Trustworthy AI:** Clinicians can validate model reasoning against biological knowledge

### Broader Impact
1. **Reproducibility:** Public SAE catalog for community use (like InterPLM for proteins)
2. **Mayo Clinic:** Differentiator for genomic medicine program (interpretable, not just accurate)
3. **PhD Thesis:** Becomes Chapter 2 (Interpretability) for Josh's PhD (starts Sep 2026)

---

## Why This Will Succeed

### Infrastructure (Already Exists)
- **gfm-platform:** Two-stage embedding pipeline (ADR-002), layer hunting (ADR-003), Nucleotide Transformer implemented
- **Data:** Helix 100k IBD cohort (2000 patients, 15 genes), Regeneron 80k (pending), 1000 Genomes (public)
- **Compute:** Mayo HPC (10 A100 GPUs, ~1000 GPU-hours available)
- **Validation:** GWAS catalog, ClinVar, JASPAR, ENCODE (all public)

### Team (Mayo Clinic + Collaborators)
- **Josh Meehl:** Sr. Data Scientist, Mayo Digital Pathology, ML expertise
- **Eric Klee:** Genomics advisor, IBD domain expertise
- **Carl Molnar:** ML engineer, compute resources
- **GoodFire AI:** Planned partnership (SAE training expertise from LLM work)

### Prior Work (De-Risked)
- **InterPLM (Nature Methods 2025):** Proved SAEs work for biological foundation models (proteins)
- **gfm-platform IBD study:** Baseline AUROC 0.72 (shows phenotype is learnable)
- **Track A interpretability work:** Attention, ISM, IG baselines established

---

## Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SAEs don't find interpretable features | Medium | High | Negative result publishable; frame as "DNA vs. protein SAE differences" |
| Features plausible but not faithful | Medium | Medium | Ablation experiments required for all claims; never claim biology without necessity |
| Timeline too aggressive | Medium | High | Go/no-go decision points (Week 2, 6, 10, 12); can pivot to ICLR 2027 (Oct deadline) |
| NeurIPS main conference rejects | High | Medium | Target workshop (50% acceptance) or ICLR 2027 as backup |

**Honest Assessment:** NeurIPS main conference (25% acceptance) is a stretch goal. Workshop acceptance (50% rate) is realistic. Publication at top venue within 12 months is likely.

---

## Success Criteria

**Minimum Success (NeurIPS Workshop):**
- 50+ interpretable features (30% JASPAR match)
- 10+ necessary features (ΔAUROC ≥0.03)
- Methods paper documenting DNA SAE challenges

**Target Success (NeurIPS Main Conference):**
- 100+ interpretable features
- 25+ necessary features
- IBD feature circuits identified (NOD2/IL23R/ATG16L1)
- Biological validation (GWAS, ClinVar)

**Aspirational Success (Nature Methods Trajectory):**
- All target criteria plus:
- Cross-model universality (NT + Evo-2)
- Experimental validation (CRISPR, reporter assays, 6-12 months)
- Clinical deployment (ACMG-AMP variant interpretation pipeline)

---

## Timeline and Budget

**Duration:** 18 weeks (Jan 15 - May 22, 2026)

**Resources:**
- Josh time: 490 hours (27 hrs/week, 50% of work time)
- GPU: 760 hours (feasible on 10 Mayo A100s)
- CPU: 705 hours (trivial on HPC)
- Storage: ~720 GB (feasible)

**Key Milestones:**
- Week 2 (Jan 21): Go/No-Go 1 - SAE architecture validated
- Week 6 (Feb 26): Milestone 1 - SAE training complete
- Week 10 (Mar 26): Milestone 2 - Features annotated
- Week 12 (Apr 9): Go/No-Go 4 - Computational necessity validated
- Week 14 (Apr 23): Milestone 3 - Biological validation complete
- Week 18 (May 22): Milestone 4 - NeurIPS submission

**Budget:** $0 direct costs (uses existing Mayo infrastructure, public data, open-source tools)

---

## Differentiation from Competitors

**vs. Attention-Based Interpretability (Most Current Work):**
- Attention maps are plausible but not faithful (saturation problem)
- SAEs provide computational necessity (ablation-validated)

**vs. Gradient-Based Methods (ISM, IG, DeepLIFT):**
- Gradients are position-level; SAEs are concept-level (features span multiple positions)
- SAEs discover recurring patterns; gradients are instance-specific

**vs. Probing Classifiers:**
- Probes ask "does representation encode X?"; SAEs ask "what does representation encode?"
- SAEs are unsupervised (no need to predefine concepts)

**vs. InterPLM (Protein SAEs):**
- DNA is longer context (6kb-131kb vs. <1kb)
- DNA features less characterized (JASPAR ~700 motifs vs. UniProt ~20k domains)
- Phenotype prediction requires multi-gene aggregation (novel)

---

## Long-Term Vision

**Phase 1 (NeurIPS 2026):** Methods foundation - prove SAEs work for DNA

**Phase 2 (Nature Methods 2027):** Biological discovery - experimental validation (CRISPR, reporter assays), cross-model universality (NT + Evo-2 + DNABERT-2), public SAE catalog

**Phase 3 (PhD Thesis 2027-2029):** Clinical deployment - ACMG-AMP integration, prospective validation, network-aware + ancestry-robust phenotype prediction with SAE-based variant attribution

**Phase 4 (Mayo Product 2029+):** Production system - interpretable genomic medicine platform, regulatory approval (SaMD), clinical decision support

---

## Call to Action

**For Reviewers:** This is the first rigorous mechanistic interpretability work for DNA foundation models, with clear path to clinical utility. Negative results (if features aren't interpretable) are also publishable as methodological insights.

**For Collaborators:** We have infrastructure (gfm-platform), data (Helix 100k), and team (Mayo + GoodFire). Looking for SAE training expertise, genomics domain knowledge, or experimental validation partners.

**For Funders:** Interpretability unlocks clinical adoption of genomic AI. This project addresses the "trust gap" between black-box predictions and clinician decision-making.

---

## Contact

**Josh Meehl**
- Email: meehl.joshua@mayo.edu
- Organization: Mayo Clinic, Digital Pathology (Digital Biology)
- GitHub: meehljd
- Location: Remote (Minneapolis area)

**Project Links:**
- Research Plan: `/root/gfm-discovery/docs/planning/projects/sae-dna-neurips2026-research-plan.md`
- gfm-platform: `/root/gfm-platform/` (infrastructure)
- gfm-book Chapter 25: Interpretability theory (plausible vs. faithful)

---

**Last Updated:** 2026-01-15
**Version:** 1.0 (Executive Summary)
