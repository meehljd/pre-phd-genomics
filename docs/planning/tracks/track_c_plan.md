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

| Month   | Paper                       | Focus                    |
| ------- | --------------------------- | ------------------------ |
| Jan     | DeepRare                    | Multi-agent architecture |
| Jan-Feb | AlphaGenome                 | Regulatory variants      |
| Feb     | GenoMAS                     | Multi-agent reasoning    |
| Feb-Mar | Follow-up papers from above | Refinements, benchmarks  |
| Mar     | Exomiser recent updates     | Baseline comparison      |
| Mar-Apr | PhenoLinker / similar       | Phenotype prediction     |
| Apr-May | IMPPROVE deep dive          | Your direct comparison   |
| May-Jun | Other competitive papers    | Catch-up reading         |
| Jun-Aug | Your deep-dive topics       | Specialized knowledge    |
