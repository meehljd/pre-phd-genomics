# CLAUDE.md -- Pre-PhD Genomics Research

**Last updated:** 2026-01-09
**Project:** Pre-PhD Research Preparation
**Focus:** Genomic Foundation Models, Interpretability, Network Medicine, Clinical Applications
**Timeline:** Nov 2025 - Sep 2026 (PhD start)

> **Purpose:** Configure Claude Code for PhD research preparation.
> Supports exploration, skill-building, and publishable work across 3 tracks.

---

## 0) Quick Start

### Getting Best Results

When asking for help, include:
- **Goal + acceptance criteria** (what "done" means)
- **Track/Phase context** (e.g., "Track A Phase 1: Attention visualization")
- **Time budget** (e.g., "2-hour exploration" vs "8-hour implementation")
- **Output type** (notebook / notes / code / paper draft)

**Prompt template:**
```
Goal:
Track/Phase:
Time budget:
Relevant files:
Expected outputs:
```

---

## 1) Cross-Repository Integration

This repo is part of a **3-repo workspace**:

### genomic-foundation-platform (../genomic-foundation-platform)
Production GFM platform for phenotype prediction at Mayo Clinic.

**Use for:**
- Model implementations: `gfm_foundation/` (NT, DNABERT-2, ESM2 adapters)
- Pipeline patterns: `gfm_tasks/`, `gfm_data/`
- Evaluation approaches: `gfm_eval/`
- Architecture decisions: `docs/ADR/`

### gfm_book (../gfm_book)
Textbook on Genomic Foundation Models (7 parts, 32 chapters).

**Use for:**
- Conceptual explanations: `part_1/` through `part_7/`
- Methodology references: interpretability, fairness, clinical chapters
- Terminology: `meta/glossary/`

---

## 2) Research Context

### Study Plan Overview (v4)
- **Total time:** 235-295 hours over 10 months (~6-7 hrs/week)
- **PhD Start:** Sep 2026
- **Advisor:** Eric Klee (Mayo Clinic)
- **Full plan:** `docs/planning/pre-phd_intensive_study_plan.md`

### Track Structure

| Track | Focus | Duration | Hours |
|-------|-------|----------|-------|
| **A** | Interpretability (attention, SHAP, counterfactuals) | 2-3 weeks | 20-25 |
| **B** | Genomics + Ancestry + Network Medicine | 10-12 weeks | 105-130 |
| **C** | Competitive Analysis + HAO + Study Design | 6-8 weeks | 40-50 |

### Success Criteria (By Sep 2026)
- [ ] Fluent in ACMG variant classification
- [ ] Understand ancestry confounding + mitigation strategies
- [ ] Genomic foundation model selected and benchmarked
- [ ] Network topology → phenotype understanding
- [ ] Explain DeepRare/AlphaGenome/GenoMAS strengths and weaknesses
- [ ] Prospective study protocol draft ready
- [ ] 5 completed projects + GitHub portfolio

---

## 3) Repository Structure

```
pre-phd-genomics/
├── 01_interpretability/     # Track A: attention viz, SHAP, counterfactuals
├── 02_genomics_domain/      # Track B Phase 1: ACMG, HPO, annotations
├── 02_debiasing/            # Track B: ancestry robustness work
├── 03_network_medicine/     # Track B Phase 2: PPI, propagation
├── 04_integrated_projects/  # Cross-track projects
├── 05_clinical_context/     # Track C: diagnostic workflows
├── 06_competitive_analysis/ # Track C: DeepRare, AlphaGenome, GenoMAS
├── 07_agentic_design/       # HAO multi-agent systems
├── docs/
│   ├── planning/            # Study plans, track details, weekly plans
│   ├── reading_notes/       # Paper summaries
│   ├── papers/              # Paper management
│   ├── experiments/         # Experiment tracking
│   ├── journal/             # Learning journal entries
│   └── reports/             # Agent output reports
└── scripts/                 # Utility scripts
```

---

## 4) Key Workflows

### Literature Review
1. Find papers: `/paper search <query>` or invoke `paper-hunter`
2. Summarize: `/paper summarize <id>` or invoke `paper-summarizer`
3. Log in `docs/papers/reading-list.yaml` and reading notes

### Experiment Tracking
1. Create notebook in appropriate track directory
2. Log experiment: `/experiment new <name>`
3. Record results: `/experiment log`
4. Update weekly progress

### Learning Journal
1. Daily/weekly reflections: `/learning-log`
2. Track skill progression
3. Review progress against success criteria

### Paper Implementation
1. Read paper with `paper-summarizer`
2. Plan implementation with `implementation-helper`
3. Review code with `research-code-reviewer`

---

## 5) Agent Quick Reference

### Global Agents (available in all repos)
| Agent | Purpose | Model |
|-------|---------|-------|
| `paper-hunter` | Find relevant papers | sonnet |
| `paper-summarizer` | Structured paper summaries | sonnet |
| `research-planner` | Break down goals, weekly plans | sonnet |
| `phd-mentor` | Strategic guidance, career advice | opus |
| `cross-repo-navigator` | Find resources across repos | haiku |

### Local Agents (this repo)
| Agent | Purpose |
|-------|---------|
| `experiment-tracker` | Log experiments by track |
| `learning-journal` | Daily learning reflections |
| `implementation-helper` | Implement papers |
| `research-code-reviewer` | Review notebooks |
| `research-orchestrator` | Coordinate workflows |
| `advisory-committee` | Convene committee for comprehensive review |
| `advisor` | Primary advisor mentorship feedback |
| `oral-examiner` | Rigorous methodology critique |
| `program-chair` | Institutional/timeline perspective |

---

## 6) Slash Commands

| Command | Purpose |
|---------|---------|
| `/experiment` | Quick experiment logging |
| `/paper` | Paper management (add, summarize, search) |
| `/learning-log` | Daily learning entry |
| `/weekly-review` | Weekly progress review + planning |
| `/research-question` | Explore research questions |

---

## 7) Standard Commands

### Environment
```bash
source .venv/bin/activate
# or
conda activate pre-phd
```

### Run Notebooks
```bash
jupyter lab
```

### Quality Checks
```bash
ruff format . && ruff check .
```

---

## 8) Quick Reference

| Need | File |
|------|------|
| Study plan | `docs/planning/pre-phd_intensive_study_plan.md` |
| Track A details | `docs/planning/tracks/track_a_plan.md` |
| Track B details | `docs/planning/tracks/track_b_plan.md` |
| Track C details | `docs/planning/tracks/track_c_plan.md` |
| Reading notes | `docs/reading_notes/` |
| Experiment log | `docs/experiments/experiment-log.yaml` |
| Paper list | `docs/papers/reading-list.yaml` |

---

## 9) Safety & Compliance

- Treat all genomic data as potentially PHI
- Use synthetic/public data (1000 Genomes, gnomAD) for development
- Mayo data workflows require separate compliance review

---

## Context Snapshot (keep after /compact)

```
Repo: pre-phd-genomics (PhD prep, 3 tracks: A/B/C)
Cross-repo: GFP (platform code), gfm_book (conceptual reference)
Timeline: Nov 2025 - Sep 2026, then PhD starts
Focus: GFM interpretability, network medicine, clinical validation
Deliverables: 5 projects, 40+ hrs reading notes, portfolio
Key differentiators: Network-aware + ancestry-robust + prospective validation
```
