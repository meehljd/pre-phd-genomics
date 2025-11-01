# Pre-PhD Research Proposal (Grounding Draft)
**Genomic AI for Rare Disease: Prospective Clinical Validation**

---

## Core Thesis Idea

Build and **prospectively validate** an interpretable AI agent system for rare disease diagnosis—measuring real-world diagnostic yield and health outcomes.

**Key differentiator:** Competitors (DeepRare, AlphaGenome, GenoMAS) have retrospective validation only. This would be first prospective clinical trial.

---

## Three Research Aims

### Aim 1: Technical Development (Months 1-18)
**Goal:** Build interpretable AI system with retrospective validation

- Gene-scale foundation models + interpretability (attention viz, counterfactuals)
- Isoform-specific phenotype prediction (ESM1b + variant effect)
- Multi-ancestry training data (address population stratification)
- Multi-omics integration (WES/WGS + RNA-seq + phenotype)

**Output:** Paper 1 - retrospective validation on Mayo cohorts  
**Target:** Nature Genetics / Genome Medicine (May 2027)

### Aim 2: Prospective Trial (Months 12-36)
**Goal:** Deploy in real-time with undiagnosed patients

- **Design:** Stepped-wedge cluster RCT by clinical team
- **Primary endpoint:** Diagnostic yield within 90 days (target: 15% absolute increase)
- **Control:** Standard care + AI silent (audit only)
- **Intervention:** Standard care + AI decision support (add-on, <48hr turnaround)
- **Sample size:** 180-200 patients, 24-month enrollment (Sep 2027 - Aug 2029)
- **Key innovation:** 25 secondary endpoints including **6 diagnostic quality metrics**:
  - Clinical actionability, phenotype resolution, diagnostic certainty, clinical utility, concordance, reclassification rate
- **Governance:** Eric Klee (PI), me (co-I/lead analyst, 1st author)

**Output:** Paper 2 - prospective clinical utility + diagnostic quality  
**Target:** NEJM / Nature Medicine (Jan 2030)

### Aim 3: Longitudinal Outcomes & Health Economics (Months 24-48)
**Goal:** Assess long-term impact and cost-effectiveness

- **Follow-up study of Aim 2 patients** (same cohort, 12-24 months post-diagnosis)
- Long-term clinical outcomes (hospitalizations, QOL, mortality)
- Diagnostic stability (variant reclassification rates)
- Cost-effectiveness analysis (ICER, cost per QALY)
- Family cascade screening impact
- Treatment initiation & adherence tracking

**Output:** Paper 3 - long-term outcomes & cost-effectiveness  
**Target:** JAMA / Health Affairs (Apr 2030)

---

## Pre-PhD Preparation Plan (Nov 2025 - Sep 2026)

**235-295 hours over 10 months** — three parallel tracks:

**Track A: Interpretability** (20-25 hrs) - Attention viz, SHAP, counterfactuals on ESM2

**Track B: Genomics Domain** (105-130 hrs) - ACMG guidelines, HPO phenotyping, ancestry/stratification (6-7 hrs), network medicine, genomic foundation model evaluation (Jan 2026 decision)

**Track C: Competitive & Clinical** (40-50 hrs) - DeepRare/AlphaGenome analysis, Microsoft HAO, prospective study design + IRB

**5 Integrated Projects:** Variant ranking system, multi-agent orchestrator, ancestry-robust pipeline, competitive benchmark, prospective protocol draft

---

## Questions for You

**Feasibility:**
- Is prospective validation feasible with Mayo infrastructure + your lab?
- 180-200 patients in 24 months realistic? What's current diagnostic yield in undiagnosed patients?
- Stepped-wedge cluster RCT feasible? Can we do 6-month silent-run pilot to measure baseline + ICC?

**Scope:**
- Is longitudinal follow-up (Aim 3) suitable expansion for PhD scope?
- Are 6 diagnostic quality metrics (actionability, phenotype resolution, etc.) clinically meaningful?
- Multi-ancestry + isoform-level prediction enough novelty for Aim 1, or add regulatory variants?

**Timeline:**
- Paper 1 by Jul 2027 aggressive? Prelim exam before or after Paper 1 submission?
- If DeepRare/AlphaGenome publish prospective validation before us, what's Plan B?

---

## Pre-PhD Deliverables (Ready Sep 2026)

By program start, I'll have:
- ✅ 5 completed projects (GitHub portfolio, 60+ hrs code)
- ✅ Genomic foundation model selected + benchmarked
- ✅ Interpretability methods tested, ancestry/bias mitigation documented
- ✅ Competitive analysis complete (40+ hrs reading notes)
- ✅ Draft prospective study protocol + HAO orchestrator prototype

---

## Funding Strategy

- **NSF GRFP** (apply Fall 2026 as 1st year) — higher priority
- **NIH F31** (backup if NSF unsuccessful)
- **ARPA-H RAPID** — explicitly funds rare disease AI clinical deployment
- **NIH U01** — multi-site prospective trials

---

## Timeline Milestones

| Event                          | Date             |
|--------------------------------|------------------|
| PhD application                | Dec 2025         |
| Program start                  | Aug 2026         |
| **Paper 1 submission**         | **Jul 2027**     |
| Prelim written exam            | Spring 2027      |
| Prelim oral exam               | June 2027        |
| Model freeze                   | Jul 2027         |
| Prospective trial launch       | Sep 2027         |
| Aim 3 data collection begins   | Jul 2028         |
| Trial enrollment complete      | Aug 2029         |
| **Paper 2 submission**         | **Jan 2030**     |
| **Paper 3 submission**         | **Apr 2030**     |
| Defense                        | Apr 2030         |

---

**Biggest ask:** Tell me if I'm off base on feasibility, timeline, or technical scope. Want to calibrate before diving into 10-month prep plan.
