# Sparse Autoencoders for DNA Foundation Models (NeurIPS 2026)

**Project Overview:** Mechanistic interpretability research applying sparse autoencoders to DNA foundation models for inflammatory bowel disease (IBD) phenotype prediction.

**Target Venue:** NeurIPS 2026 (Main Conference or Workshop)
**Timeline:** 18 weeks (Jan 15 - May 22, 2026)
**Status:** Planning → Execution (Week 1 starting Jan 15)

---

## Quick Links

### Core Documentation
1. **[Research Plan](sae-dna-neurips2026-research-plan.md)** (40 pages)
   - Full scientific plan with 9 sections
   - Research questions, hypotheses, validation strategy
   - Phase-by-phase breakdown, resource requirements
   - Risk assessment, go/no-go decision criteria

2. **[Executive Summary](sae-dna-neurips2026-executive-summary.md)** (5 pages)
   - One-sentence pitch, problem/solution
   - Impact, differentiation, long-term vision
   - Perfect for grant applications, collaborator pitches

3. **[Timeline and Critical Path](sae-dna-neurips2026-timeline.md)** (10 pages)
   - 18-week Gantt chart
   - Go/no-go decision points
   - Resource budgets (Josh hours, GPU, CPU, storage)
   - Contingency plans by phase

4. **[Week 1 Action Items](sae-dna-neurips2026-week1-actions.md)** (7 pages)
   - Day-by-day checklist (Jan 15-22)
   - Deliverables, progress log template
   - Go/No-Go Decision 1 criteria

### Supporting Materials (To Be Created)
- [ ] Weekly progress log (sae-dna-neurips2026-progress-log.md)
- [ ] Go/no-go decision log (sae-dna-neurips2026-decisions.md)
- [ ] Literature review (InterPLM, SAE methods, DNA interpretability)
- [ ] NeurIPS submission materials (manuscript, figures, supplements)

---

## Project At-A-Glance

### The One-Paragraph Pitch
We're extending sparse autoencoder (SAE) methodology from protein language models (InterPLM, Nature Methods 2025) to DNA foundation models, targeting inflammatory bowel disease (IBD) phenotype prediction. SAEs decompose Nucleotide Transformer representations into 100+ interpretable features corresponding to regulatory elements (JASPAR motifs), disease mechanisms (NOD2 LRR domains), and variant effects (ClinVar pathogenicity). Unlike attention maps (plausible but not faithful), we validate computational necessity via ablation experiments and biological validity via GWAS/ClinVar enrichment. This enables mechanistic ACMG-AMP variant interpretation ("disrupts HNF4A binding feature 1247") rather than black-box scores, addressing the clinical trust gap in genomic AI.

### Key Innovation
**First SAE application to DNA foundation models** with unique challenges: longer context (6kb-131kb vs. <1kb proteins), weaker ground truth (JASPAR ~700 motifs vs. UniProt ~20k domains), and multi-gene aggregation for phenotype prediction.

### Success Criteria
- **Minimum (Workshop):** 50+ interpretable features, 10+ necessary features (ΔAUROC ≥0.03)
- **Target (Main Conference):** 100+ interpretable features, 25+ necessary features, biological validation
- **Aspirational (Nature Methods):** Cross-model universality, experimental validation, clinical deployment

---

## 4-Phase Timeline

| Phase | Dates | Goal | Deliverable |
|-------|-------|------|-------------|
| **Phase 1** | Jan 15 - Feb 26 (6 weeks) | SAE Training | 32 trained SAEs (R² ≥0.95, L0 ≤20) |
| **Phase 2** | Feb 27 - Mar 26 (4 weeks) | Feature Characterization | 100+ annotated features (JASPAR, ENCODE, ClinVar) |
| **Phase 3** | Mar 27 - Apr 23 (4 weeks) | Validation | 25+ necessary features (ablation ΔAUROC ≥0.03) |
| **Phase 4** | Apr 24 - May 22 (4 weeks) | Paper Writing | NeurIPS submission (8-9 pages, 6-8 figures) |

**Critical Milestones:**
- Week 2 (Jan 21): Go/No-Go 1 - SAE architecture validated
- Week 6 (Feb 26): Milestone 1 - SAE training complete
- Week 10 (Mar 26): Milestone 2 - Features annotated
- Week 12 (Apr 9): Go/No-Go 4 - Computational necessity validated
- Week 18 (May 22): NeurIPS submission

---

## Resource Summary

### People
- **Josh Meehl:** Project lead, SAE training, validation, paper writing (490 hours, 27h/week)
- **Carl Molnar:** ML engineer, compute resources, code review
- **Eric Klee:** Genomics advisor, IBD domain expertise, clinical context
- **GoodFire AI:** Partnership (SAE training expertise, optional)

### Compute
- **GPU:** 760 hours (10 A100 GPUs × 76 hours wall-clock, Mayo HPC)
- **CPU:** 705 hours (embarrassingly parallel, Mayo HPC cluster)
- **Storage:** 720 GB (embeddings, SAE models, feature catalogs)

### Data
- **Helix 100k:** IBD cohort (2000 patients, 15 genes, 30k sequences) - PRIMARY
- **Regeneron 80k:** External validation (pending DUA, Feb 2026 target)
- **Public:** 1000 Genomes (prototyping), GWAS catalog, ClinVar, JASPAR, ENCODE

### Budget
- **Direct Costs:** $0 (uses existing Mayo infrastructure, public data, open-source tools)

---

## Risk Assessment (Top 5)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Features not interpretable (<10% JASPAR match) | Medium | High | Pivot to methods paper; frame as DNA vs. protein differences |
| Features plausible but not faithful (no ablation effect) | Medium | Medium | Require necessity for all claims; compare to baselines |
| Timeline too aggressive (NeurIPS deadline missed) | Medium | High | Go/no-go at Week 2, 6, 10, 12; ICLR 2027 as backup |
| Compute resources insufficient | Low | High | Prioritize top 5 layers; request additional allocation |
| Concurrent work scoops results | Low | High | Emphasize unique aspects (IBD, clinical validation, Mayo data) |

---

## Differentiation from Prior Work

### vs. InterPLM (Protein SAEs, Nature Methods 2025)
- **Context:** DNA 6kb-131kb vs. protein <1kb
- **Ground Truth:** JASPAR ~700 motifs vs. UniProt ~20k domains
- **Task:** Phenotype prediction (multi-gene aggregation) vs. protein function (single-sequence)

### vs. Attention-Based Interpretability
- **Faithfulness:** SAEs validated by ablation vs. attention (plausible but not faithful)
- **Concepts:** SAEs discover features unsupervised vs. attention highlights positions

### vs. Gradient Methods (ISM, IG, DeepLIFT)
- **Level:** SAEs are concept-level (features span positions) vs. gradients are position-level
- **Recurring Patterns:** SAEs find shared features vs. gradients are instance-specific

---

## Next Steps (Immediate)

**Jan 15-16 (Day 1-2):**
1. Create project directory: `/root/gfm-discovery/08_sae_interpretability/`
2. Email GoodFire AI: Partnership inquiry
3. Clone InterPLM code: Review SAE training methodology
4. Verify Helix 100k IBD data access

**Jan 17-19 (Day 3-5):**
5. Set up SAE training environment (PyTorch, HuggingFace)
6. Extract IBD embeddings (use gfm-platform two-stage pipeline)
7. Run layer hunting (identify top 5 layers for IBD)
8. Baseline interpretability (ISM, IG, Attention on 100 sequences)

**Jan 20-21 (Day 6-7):**
9. Implement SAE training code
10. Small-scale test (1000 seqs, layer 15, R² ≥0.90 target)
11. Hyperparameter sweep design (for Week 3-4)
12. **Go/No-Go Decision 1** (Jan 21)

**Details:** See [Week 1 Action Items](sae-dna-neurips2026-week1-actions.md)

---

## Key References

**SAE Methodology:**
- InterPLM (Simon & Zou, Nature Methods 2025): Protein SAE reference
- Anthropic SAE work: Gated SAEs, training best practices
- GoodFire AI: LLM interpretability SAEs

**DNA Interpretability:**
- gfm-book Chapter 25: Plausible vs. faithful, validation hierarchy
- ADR-034 (ISM), ADR-035 (IG), ADR-039 (Attention): Platform baselines

**IBD Biology:**
- NOD2, IL23R, ATG16L1: Well-characterized disease genes
- IBD GWAS: International IBD Genetics Consortium (~200 loci)
- ClinVar: Pathogenic variant catalog

**Clinical Context:**
- ACMG-AMP Variant Interpretation (Richards et al. 2015)
- PP3/BP4 Computational Evidence criteria

---

## Document Index

| Document | Purpose | Pages | Last Updated |
|----------|---------|-------|--------------|
| **Research Plan** | Full scientific plan | 40 | 2026-01-15 |
| **Executive Summary** | Elevator pitch, grants | 5 | 2026-01-15 |
| **Timeline** | Gantt chart, critical path | 10 | 2026-01-15 |
| **Week 1 Actions** | Day-by-day checklist | 7 | 2026-01-15 |
| **README** (this doc) | Quick reference | 5 | 2026-01-15 |
| Progress Log | Weekly updates | TBD | TBD |
| Decision Log | Go/no-go decisions | TBD | TBD |
| Literature Review | InterPLM, SAE methods | TBD | TBD |
| Manuscript Draft | NeurIPS submission | TBD | TBD |

**Total Documentation:** 62+ pages (comprehensive planning)

---

## Repository Structure

```
gfm-discovery/
├── docs/planning/projects/
│   ├── README-SAE-DNA-NEURIPS2026.md (this file)
│   ├── sae-dna-neurips2026-research-plan.md
│   ├── sae-dna-neurips2026-executive-summary.md
│   ├── sae-dna-neurips2026-timeline.md
│   ├── sae-dna-neurips2026-week1-actions.md
│   ├── sae-dna-neurips2026-progress-log.md (to be created)
│   ├── sae-dna-neurips2026-decisions.md (to be created)
│   └── neurips-submission/ (created Week 15+)
│
├── 08_sae_interpretability/ (NEW - to be created Week 1)
│   ├── 01_sae_training/
│   │   ├── train_sae.py
│   │   ├── config/
│   │   └── models/
│   ├── 02_feature_extraction/
│   ├── 03_biological_annotation/
│   ├── 04_validation/
│   ├── 05_analysis_notebooks/
│   └── 06_paper_figures/
```

---

## Contact and Collaboration

**Project Lead:**
- Josh Meehl (meehl.joshua@mayo.edu)
- Mayo Clinic, Digital Pathology (Digital Biology)
- GitHub: meehljd

**Looking For:**
- SAE training expertise (GoodFire AI partnership)
- Genomics domain knowledge (regulatory elements, IBD biology)
- Experimental validation partners (CRISPR, reporter assays for Phase 2)

**Open to:**
- Co-authorship (GoodFire AI, other collaborators)
- Code sharing (public release planned upon NeurIPS acceptance)
- Data sharing (within PHI constraints - aggregates only, no patient-level)

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-15 | Initial comprehensive plan (62 pages total) | Josh Meehl |
| 1.1 | TBD | Week 1 updates, Go/No-Go 1 decision | TBD |

---

**Status:** Ready to start (Week 1 begins Jan 15, 2026)
**Next Review:** Jan 22, 2026 (end of Week 1)
**Next Milestone:** Feb 26, 2026 (SAE training complete)
