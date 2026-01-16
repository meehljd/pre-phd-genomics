# SAE DNA NeurIPS 2026: Project Timeline and Critical Path

**Project Duration:** 18 weeks (Jan 15 - May 22, 2026)
**Target Submission:** NeurIPS 2026 (Early May deadline)

---

## Timeline Overview (Gantt-Style)

```
Jan  │ Week 1-2: Setup          │ Week 3-4: SAE Training (Initial) │ Week 5-6: SAE Training (Full)
     │ ████████████████████████ │ ████████████████████████████████ │ ████████████████████████████
     │ 15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  01  02  03  04  05  06
     │                          │                                  │
     │ ← Go/No-Go 1 (Jan 21)   │                                  │ ← Milestone 1 (Feb 26)

Feb  │ Week 7: Annotation       │ Week 8: IBD Analysis   │ Week 9: Cross-Layer │ Week 10: Baselines
     │ ████████████████████████ │ ██████████████████████ │ ███████████████████ │ ██████████████████
     │ 27  28  01  02  03  04  05  06  07  08  09  10  11  12  13  14  15  16  17  18  19  20  21
                                │                        │                     │
                                │                        │                     │ ← Milestone 2 (Mar 26)

Mar  │ Week 11-12: Ablation      │ Week 13: Sufficiency │ Week 14: Bio Validation
     │ ████████████████████████████████████████████████ │ ████████████████████ │ ████████████████████
     │ 27  28  29  30  31  01  02  03  04  05  06  07  08  09  10  11  12  13  14  15  16  17  18  19  20
                                                         │                     │
                                                         │ ← Go/No-Go 4 (Apr 9)│ ← Milestone 3 (Apr 23)

Apr  │ Week 15: Figures         │ Week 16-17: Manuscript Drafting    │ Week 18: Revision & Submit
     │ ████████████████████████ │ ████████████████████████████████████ │ ████████████████████████
     │ 24  25  26  27  28  29  30  01  02  03  04  05  06  07  08  09  10  11  12  13  14  15  16  17
                                │                                        │
                                │                                        │ ← Milestone 4 (May 22)
May  │                           │                                        │
     │                           │ 18  19  20  21  22 ← NEURIPS DEADLINE │
```

---

## Critical Path Analysis

### Critical Path 1: SAE Training Quality (Weeks 1-6)
**Why Critical:** If SAEs don't achieve R² ≥0.95 and L0 ≤20, entire project fails
**Dependencies:**
- Week 1-2: Environment setup → Small-scale test
- Week 3-4: Hyperparameter sweep → Best config selection
- Week 5-6: Full-scale training (32 layers)
**Risk:** Medium (InterPLM worked for proteins, but DNA may be harder)
**Mitigation:** Early go/no-go (Week 2), contingency time built in

### Critical Path 2: Feature Interpretability (Weeks 7-10)
**Why Critical:** If <30% features match JASPAR, no biological story
**Dependencies:**
- Week 6: Feature activations extracted → Week 7: JASPAR scanning
- Week 7: Annotations complete → Week 8: IBD-specific analysis
- Week 10: Baseline comparisons → Paper Methods section
**Risk:** Medium-High (DNA features less characterized than proteins)
**Mitigation:** Go/no-go at Week 10, pivot to methods paper if needed

### Critical Path 3: Computational Necessity (Weeks 11-12)
**Why Critical:** Plausibility without faithfulness = unpublishable
**Dependencies:**
- Week 10: Feature catalog → Week 11: Select 100 features for ablation
- Week 11: Ablation experiments → Week 12: Statistical validation
- Week 12: Results → Go/no-go decision (≥25 necessary features)
**Risk:** High (Saturation, redundancy could hide necessity)
**Mitigation:** Test on subset first (Week 11 Day 1-2), adjust threshold if needed

### Critical Path 4: Paper Writing (Weeks 15-18)
**Why Critical:** NeurIPS deadline is hard (no extensions)
**Dependencies:**
- Week 14: All validation complete → Week 15: Figures
- Week 15: Figures → Week 16-17: Manuscript
- Week 17: Draft → Week 18: Revision + submission
**Risk:** Medium (Timeline tight, but standard academic pace)
**Mitigation:** Start Methods/Introduction in parallel during Phase 2-3

---

## Go/No-Go Decision Points (Critical Milestones)

### Decision 1 (Jan 21, End of Week 2): SAE Architecture Validated
**Criteria:**
- ✓ Small-scale SAE (1000 seqs, layer 15) achieves R² ≥0.90, L0 ≤30
- ✓ Training time <5 hours per layer (scalable to 32 layers)
- ✓ GoodFire partnership status resolved (yes/no)

**GO:** Proceed to Week 3-4 hyperparameter sweep
**NO-GO:** Extend Week 2 by 3-5 days, try gated SAE architecture, adjust hyperparameters

---

### Decision 2 (Feb 26, End of Week 6): SAE Training Complete
**Criteria:**
- ✓ 32 SAEs trained (all layers) with R² ≥0.95, L0 ≤20
- ✓ Feature activations extracted for 30k sequences
- ✓ Sanity check: Shuffled SAE shows random patterns (negative control)

**GO:** Proceed to Week 7-10 biological annotation
**NO-GO:** Retrain with adjusted hyperparameters, focus on top 5 layers only

---

### Decision 3 (Mar 26, End of Week 10): Features Are Interpretable
**Criteria:**
- ✓ ≥30% features match JASPAR motifs (p < 0.01)
- ✓ ≥50 features enriched in IBD cases (FDR < 0.05)
- ✓ Baseline comparisons complete (SAE vs. neurons, SAE vs. ISM/IG)

**GO:** Proceed to Week 11-14 validation (necessity/sufficiency)
**NO-GO Options:**
- Pivot A: Methods paper ("Why DNA SAEs Are Hard") → NeurIPS workshop
- Pivot B: Add multi-model analysis (train SAEs on Evo-2) to boost interpretability
- Pivot C: Target ICLR 2027 (Oct deadline) for more time

---

### Decision 4 (Apr 9, End of Week 12): Computational Necessity Validated
**Criteria:**
- ✓ ≥25 features show ablation ΔAUROC ≥0.03 (p < 0.05)
- ✓ Negative control: Random features show ΔAUROC ~0
- ✓ Biological validation in progress (GWAS, ClinVar)

**GO:** Proceed to Week 15-18 paper writing (NeurIPS main conference target)
**NO-GO Options:**
- Pivot A: Target NeurIPS workshop instead of main conference
- Pivot B: Extend validation to sufficiency + multi-model (defer to ICLR 2027)

---

## Phase Dependencies (Waterfall vs. Parallel)

### Sequential Dependencies (Cannot Parallelize)
1. **Week 1-2 → Week 3-4:** Must validate architecture before hyperparameter sweep
2. **Week 5-6 → Week 7:** Must extract feature activations before annotation
3. **Week 10 → Week 11:** Must identify IBD-enriched features before ablation
4. **Week 14 → Week 15:** Must complete validation before generating figures

### Parallel Opportunities (Can Run Concurrently)
1. **Week 7-9:** JASPAR scanning (CPU) || ENCODE overlap (CPU) || ClinVar enrichment (CPU)
2. **Week 11:** Ablation for multiple features (embarrassingly parallel, 100 jobs)
3. **Week 15:** Figure generation (multiple figures by different team members)
4. **Week 16-17:** Writing Methods (early) || Results (later) || Discussion (concurrent)

---

## Weekly Time Budgets (Josh's Hours)

Assumes ~30-35 hours/week dedicated to this project (50% of work time, rest for Mayo/GoodFire)

| Week | Phase | Josh Hours | GPU Hours | CPU Hours |
|------|-------|------------|-----------|-----------|
| 1-2  | Setup | 40h (20h/wk) | 10h (small tests) | 20h |
| 3-4  | SAE Train (Initial) | 25h (monitoring sweeps) | 90h (hyperparameter grid) | 10h |
| 5-6  | SAE Train (Full) | 30h (32 layer training) | 640h (parallelized) | 20h |
| 7    | Annotation | 35h (pipeline setup) | 0h | 500h (JASPAR, ENCODE) |
| 8    | IBD Analysis | 30h (interpretation) | 0h | 50h |
| 9    | Cross-Layer | 25h (clustering, UMAP) | 0h | 20h |
| 10   | Baselines | 30h (comparison analysis) | 5h (IG, ISM) | 40h |
| 11-12| Ablation | 40h (experiment design, analysis) | 10h (ablation runs) | 10h |
| 13   | Sufficiency | 25h (insertion experiments) | 5h | 5h |
| 14   | Bio Validation | 30h (GWAS, ClinVar analysis) | 0h | 20h |
| 15   | Figures | 35h (publication-quality viz) | 0h | 10h |
| 16-17| Manuscript | 60h (30h/wk) | 0h | 0h |
| 18   | Revision | 30h (final polish) | 0h | 0h |
| **Total** | | **490h (27h/wk avg)** | **760h** | **705h** |

**Feasibility:**
- Josh time: 27 hrs/week is tight but doable (assumes 50-60 hour work weeks, 50% on this project)
- GPU: 760 hours = 76 hours on 10 GPUs (feasible within Mayo allocation)
- CPU: 705 hours = trivial on HPC cluster (100+ nodes available)

---

## Contingency Plans by Phase

### Phase 1 Contingency (Weeks 1-6)
**Risk:** SAE training fails to achieve R² ≥0.95
**Contingency:**
- Try gated SAE architecture (Week 3 Day 1-2)
- Increase hidden dim to 8192 or 16384 (Week 3 Day 3-4)
- Focus on top 5 layers only (reduce scope)
- **Timeline Impact:** +1 week (extend Phase 1 to Week 7)

### Phase 2 Contingency (Weeks 7-10)
**Risk:** <30% features match JASPAR (low interpretability)
**Contingency:**
- Add multi-model analysis (train SAEs on Evo-2 in parallel, Week 8-9)
- Broaden validation sources (RegulomeDB, DeepSEA predictions)
- Frame as "DNA SAEs discover novel patterns beyond JASPAR"
- **Timeline Impact:** +1 week (extend Phase 2 to Week 11)

### Phase 3 Contingency (Weeks 11-14)
**Risk:** <25 features show computational necessity
**Contingency:**
- Group ablation (ablate all NOD2 features together, stronger signal)
- Lower threshold to ΔAUROC ≥0.02 (still meaningful)
- Target NeurIPS workshop instead of main conference
- **Timeline Impact:** +1 week (extend Phase 3 to Week 15)

### Phase 4 Contingency (Weeks 15-18)
**Risk:** NeurIPS deadline missed due to delays
**Contingency:**
- Target ICLR 2027 (Oct 2026 deadline, +5 months)
- Use extra time for experimental validation (CRISPR, reporter assays)
- Aim for Nature Methods instead (higher impact, longer timeline)
- **Timeline Impact:** +20 weeks (but better paper)

---

## External Dependencies and Lead Times

| Dependency | Lead Time | Status | Deadline | Risk |
|------------|-----------|--------|----------|------|
| **Helix 100k Data Access** | 0 days (already available) | ✓ Available | N/A | Low |
| **Regeneron 80k Data** | ~4 weeks (DUA signing) | Pending | Feb 15 target | Medium (not critical) |
| **Mayo HPC GPU Allocation** | ~1 week (request + approval) | Requested | Jan 22 target | Low |
| **GoodFire Partnership** | ~2 weeks (initial response) | Pending | Jan 29 target | Medium (can proceed without) |
| **NeurIPS Reviewer Responses** | ~8 weeks (Jun-Aug 2026) | N/A | Aug 2026 | N/A (post-submission) |

**Critical External Dependency:** None (all contingencies in place)

---

## Success Probability Estimate

### Conservative Estimate (50% Probability)
- SAE training succeeds: 80%
- Features are interpretable (≥30% JASPAR): 60%
- Computational necessity (≥25 features): 50%
- Paper accepted at NeurIPS main: 25% (given ~25% acceptance rate)
- **Overall Success (NeurIPS Main):** 80% × 60% × 50% × 25% = **6%**

### Realistic Estimate (Workshop as Success)
- SAE training succeeds: 80%
- Features are interpretable: 60%
- Computational necessity: 50%
- Paper accepted at NeurIPS workshop OR main: 50% (workshop ~50% acceptance)
- **Overall Success (NeurIPS Workshop):** 80% × 60% × 50% × 50% = **12%**

### Optimistic Estimate (ICLR 2027 Fallback)
- SAE training succeeds: 80%
- Features are interpretable: 60%
- Computational necessity (with extended time): 70%
- Paper accepted at ICLR 2027 OR NeurIPS workshop: 60%
- **Overall Success (Any Top Venue):** 80% × 60% × 70% × 60% = **20%**

**Interpretation:**
- NeurIPS 2026 main conference is a stretch goal (6% success)
- NeurIPS workshop is realistic (12% success)
- Publication at a top venue (NeurIPS or ICLR) within 12 months is likely (20% success)
- Methods paper at workshop level is highly likely (>50% success)

---

## Next Steps

1. **Immediate (Jan 15):** Start Week 1 Day 1-2 actions (see sae-dna-neurips2026-week1-actions.md)
2. **Week 1 End (Jan 22):** Go/No-Go Decision 1 review
3. **Week 6 End (Feb 26):** Milestone 1 review, assess Phase 2 readiness
4. **Monthly:** Risk review, timeline adjustment if needed

---

**Last Updated:** 2026-01-15
**Next Review:** 2026-01-22 (end of Week 1)
**Owner:** Josh Meehl
