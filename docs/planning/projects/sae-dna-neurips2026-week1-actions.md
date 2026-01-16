# SAE DNA NeurIPS 2026: Week 1 Action Items

**Week 1 Dates:** Jan 15-22, 2026
**Goal:** Project setup, environment configuration, baseline establishment
**Status:** Planning → Execution

---

## Daily Breakdown

### Day 1-2 (Jan 15-16): Project Initialization

**Action 1.1: Repository Setup**
- [ ] Create project directory: `/root/gfm-discovery/08_sae_interpretability/`
- [ ] Copy directory structure from Appendix A of research plan
- [ ] Initialize Git tracking (separate branch: `sae-neurips2026`)
- [ ] Create GitHub project board with 4 phase milestones

**Action 1.2: GoodFire Partnership**
- [ ] Email GoodFire AI contacts (check gfm-platform/partners/goodfire/ for contact info)
- [ ] Subject: "SAE Collaboration for DNA Foundation Models - Mayo Clinic"
- [ ] Attach: Research plan executive summary
- [ ] Ask: Code access, collaboration model, co-authorship discussion
- [ ] Timeline: Need response by Jan 22 (go/no-go decision)

**Action 1.3: InterPLM Code Review**
- [ ] Clone InterPLM repo: https://github.com/...  (find from gfm-literature Zotero)
- [ ] Read SAE training code: `train_sae.py`, understand architecture
- [ ] Document hyperparameters: sparsity λ, hidden dim, activation function
- [ ] Note differences needed for DNA: context length, tokenization

**Action 1.4: Verify Data Access**
- [ ] Check gfm-platform: Helix 100k IBD cohort location
- [ ] Verify DVC pull works: `cd /root/gfm-platform && dvc pull data/helix_ibd/`
- [ ] Count samples: 2000 patients × 15 genes = 30k sequences (expected)
- [ ] Check IRB approval: Confirm interpretability research is covered use case

---

### Day 3-5 (Jan 17-19): Environment Setup and Baseline

**Action 2.1: SAE Training Environment**
- [ ] Create conda environment: `gfm-sae` with PyTorch 2.x, HuggingFace
- [ ] Install dependencies: `pip install sparse_autoencoder` (if GoodFire SDK available)
- [ ] Test GPU access: `nvidia-smi` on Mayo HPC node
- [ ] Request compute allocation: 10 A100 GPUs for 100 hours (Jan-Feb)

**Action 2.2: Extract IBD Embeddings**
- [ ] Use gfm-platform two-stage pipeline (ADR-002)
- [ ] Run: `python gfm_platform/pipelines/extract_embeddings.py --cohort ibd --model nucleotide_transformer --genes NOD2,IL23R,ATG16L1,HNF4A,...`
- [ ] Output location: `/root/gfm-platform/data/embeddings/NT/helix_ibd/`
- [ ] Verify shape: [30000 sequences, 32 layers, 1024 dim]
- [ ] Cache to gfm-discovery: Copy subset for prototyping (1000 seqs)

**Action 2.3: Layer Hunting**
- [ ] Run layer hunting on IBD: Use `gfm_eval/layer_hunting.py`
- [ ] Config: 5-fold CV, logistic regression classifier, AUROC metric
- [ ] Output: Best layer per fold, mean AUROC per layer
- [ ] Expected result: Identify top 5 layers (e.g., layers 8, 12, 15, 22, 28)
- [ ] Save results: `08_sae_interpretability/layer_hunting_results.yaml`

**Action 2.4: Baseline Interpretability**
- [ ] Sample 100 test sequences (50 IBD cases, 50 controls)
- [ ] Run ISM: `gfm_eval/interpretability/ism.py` (use ADR-034 implementation)
- [ ] Run Integrated Gradients: `gfm_eval/interpretability/integrated_gradients.py` (ADR-035)
- [ ] Run Attention: `gfm_eval/interpretability/attention.py` (ADR-039)
- [ ] Save attribution scores: Will compare to SAE features later
- [ ] Timing baseline: How long does ISM take for 100 seqs? (expect: ~2 hours)

---

### Day 6-7 (Jan 20-21): SAE Architecture Testing

**Action 3.1: Implement SAE Training**
- [ ] Code location: `08_sae_interpretability/01_sae_training/train_sae.py`
- [ ] Architecture: Standard linear encoder/decoder with ReLU (InterPLM baseline)
- [ ] Input dim: 1024 (NT embedding dim)
- [ ] Hidden dim: 4096 (4x expansion, following InterPLM)
- [ ] Sparsity penalty: L1 with λ = 0.001 (initial guess, will sweep)
- [ ] Loss: Reconstruction MSE + λ * L1(activations)

**Action 3.2: Small-Scale Test**
- [ ] Train on 1000 sequences, layer 15 (best layer from layer hunting)
- [ ] Training: 10k steps, batch size 32, Adam optimizer
- [ ] Monitor: Reconstruction R², sparsity L0 (# active features per seq)
- [ ] Target: R² ≥ 0.90, L0 ≤ 30
- [ ] Time: ~2 hours on single GPU (feasibility check)

**Action 3.3: Hyperparameter Sweep Design**
- [ ] Grid for Week 3-4:
  - Sparsity λ: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
  - Hidden dim: [2048, 4096, 8192]
  - Architecture: [standard, gated] (if time permits)
- [ ] Config files: `08_sae_interpretability/01_sae_training/config/sweep_*.yaml`
- [ ] Total runs: 6 λ × 3 hidden × 1 arch = 18 runs (5 hours each = 90 GPU-hours)

**Action 3.4: Go/No-Go Decision 1**
- [ ] Criterion: Small-scale SAE achieves R² ≥ 0.90, L0 ≤ 30
- [ ] If YES: Proceed to Week 2 (full hyperparameter sweep)
- [ ] If NO: Troubleshoot architecture (try gated SAE, adjust hidden dim, check data quality)
- [ ] Document decision: `go-no-go-decisions.md`

---

## Week 1 Deliverables Checklist

### Code
- [ ] SAE training script: `train_sae.py` (functional, tested on 1000 seqs)
- [ ] Embedding extraction: Cached 30k IBD sequences, 32 layers, 1024 dim
- [ ] Baseline interpretability: ISM, IG, Attention results for 100 seqs

### Data
- [ ] Layer hunting results: Top 5 layers for IBD (YAML)
- [ ] Small-scale SAE: Trained on layer 15, 1000 seqs (checkpoint saved)
- [ ] Baseline attribution scores: ISM/IG/Attention (HDF5)

### Documentation
- [ ] Research plan: Published (this document's parent)
- [ ] Week 1 progress log: Daily updates (see below)
- [ ] Go/No-Go Decision 1: Documented (Jan 21)

### Communications
- [ ] GoodFire email sent: Partnership inquiry (Jan 15)
- [ ] Compute allocation request: Mayo HPC (Jan 17)
- [ ] Eric Klee update: Project kickoff (Jan 16)

---

## Week 1 Progress Log Template

**Date:** Jan 15, 2026
- **Completed:**
  - Action 1.1: Repository setup ✓
  - Action 1.2: GoodFire email sent ✓
- **In Progress:**
  - Action 1.3: InterPLM code review (50% complete)
- **Blocked:**
  - None
- **Notes:**
  - GoodFire contact: [name]@goodfire.ai, expect response by Jan 18

**Date:** Jan 16, 2026
- **Completed:**
- **In Progress:**
- **Blocked:**
- **Notes:**

[Continue daily through Jan 22]

---

## Week 1 Risks and Mitigation

| Risk | Likelihood | Mitigation | Status |
|------|-----------|------------|--------|
| GoodFire no response by Jan 22 | Medium | Proceed with InterPLM code (open source) | Monitoring |
| Helix data access issues | Low | Use 1000 Genomes + synthetic for prototyping | N/A |
| SAE reconstruction R² < 0.90 | Medium | Try gated architecture, increase hidden dim | Test Day 6 |
| GPU allocation delayed | Low | Use personal workstation for small-scale tests | N/A |

---

## Week 1 Success Criteria

By end of Jan 22, we should have:
1. ✓ SAE training code functional (tested on 1000 seqs)
2. ✓ IBD embeddings cached (30k sequences, 32 layers)
3. ✓ Layer hunting complete (top 5 layers identified)
4. ✓ Baseline interpretability established (ISM/IG/Attention on 100 seqs)
5. ✓ Small-scale SAE trained (R² ≥ 0.90, L0 ≤ 30)
6. ✓ Go/No-Go Decision 1 made (proceed to Week 2 or troubleshoot)

**If all criteria met:** Proceed to Week 2 (hyperparameter sweep, Week 3-4 planning)

**If criteria not met:** Extend Week 1 by 3-5 days, address blockers before Week 2

---

## Appendix: Key File Paths

```bash
# Project root
/root/gfm-discovery/08_sae_interpretability/

# Embeddings (from gfm-platform)
/root/gfm-platform/data/embeddings/NT/helix_ibd/embeddings.h5

# Layer hunting results
/root/gfm-discovery/08_sae_interpretability/layer_hunting_results.yaml

# SAE training
/root/gfm-discovery/08_sae_interpretability/01_sae_training/train_sae.py
/root/gfm-discovery/08_sae_interpretability/01_sae_training/models/sae_layer15_test.pt

# Baseline interpretability
/root/gfm-discovery/08_sae_interpretability/baselines/ism_scores.h5
/root/gfm-discovery/08_sae_interpretability/baselines/ig_scores.h5
/root/gfm-discovery/08_sae_interpretability/baselines/attention_scores.h5

# Documentation
/root/gfm-discovery/docs/planning/projects/sae-dna-neurips2026-research-plan.md
/root/gfm-discovery/docs/planning/projects/sae-dna-neurips2026-week1-actions.md
/root/gfm-discovery/docs/planning/projects/sae-dna-neurips2026-progress-log.md
```

---

**Next Review:** Jan 22 (end of Week 1, before Week 2 kickoff)
**Owner:** Josh Meehl
**Last Updated:** 2026-01-15
