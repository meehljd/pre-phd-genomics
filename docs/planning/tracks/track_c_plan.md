## TRACK C: COMPETITIVE ANALYSIS + HAO + STUDY DESIGN (Apr-Jun 2026 - 6-8 weeks)

### Phase 1: Competitive Paper Deep Dive (Apr-May 2026 - 3-4 weeks)

**Goal:** Understand exactly where DeepRare, AlphaGenome, GenoMAS succeed and fail

**Reading (1 paper per week, deep analysis):**

1. **DeepRare (June 2025)** - Multi-agent rare disease system
   - Time: 6-8 hours (read + reproduce key result)
   - Focus: Multi-agent architecture, how agents coordinate, diagnostic accuracy

2. **AlphaGenome (June 2025)** - Regulatory variant foundation model
   - Time: 6-8 hours (read + test on your data)
   - Focus: Non-coding variant interpretation, how it differs from your approach

3. **GenoMAS (July 2025)** - Multi-agent genomic analysis
   - Time: 6-8 hours (read + compare to DeepRare)
   - Focus: Agent reasoning, explainability, clinical integration

4. **Your choice** - Pick most relevant competitor or related work
   - Options: Exomiser updates, PhenoLinker, MARRVEL-AI
   - Time: 6-8 hours

**Hands-On:**

```python
# Exercise 17: Competitive reproduction (8-10 hours per paper)
- Pick ONE paper (DeepRare or AlphaGenome)
- Goal: Can you reproduce their core result?
- Steps:
  * Download their code/model (if available)
  * Test on their benchmark dataset (if available)
  * Test on YOUR data (5-10 variants)
  * Document: Where does it work? Where does it fail?
  * Comparison: Your approach vs. theirs (strengths/weaknesses)

# Exercise 18: Competitive matrix (4-5 hours)
- Create comparison spreadsheet:
  * Methods: DeepRare, AlphaGenome, GenoMAS, Exomiser, Yours
  * Metrics: Diagnostic yield, interpretability, multi-ancestry, prospective validation
  * Gaps: What does no one do well?
- Identify: Your unique value proposition
```

**Output:**
- Competitive analysis documents (3-4 papers, 2000 words each)
- Reproduction notebook for 1 competitor
- Competitive comparison matrix (Excel + summary document)

---

### Phase 2: HAO Multi-Agent System Design (May 2026 - 2-3 weeks)

**Goal:** Design and implement multi-agent diagnostic system using Microsoft Healthcare Agent Orchestrator

**Reading:**
1. **Microsoft HAO documentation** (online)
   - Time: 2-3 hours

2. **Multi-agent systems papers:**
   - **Wu et al. 2023** - "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (Microsoft)
   - Time: 1.5 hours

**Hands-On:**

```python
# Exercise 19: Agent design (4-6 hours)
- Define 5 agents for rare disease diagnosis:
  1. Variant annotation agent (ACMG classification)
  2. Phenotype matching agent (HPO-based)
  3. Network analysis agent (PPI propagation)
  4. Literature search agent (PubMed API)
  5. Diagnostic reasoning agent (coordinator)
- For each agent:
  * Input/output specification
  * Tools/APIs needed
  * Confidence scoring method

# Exercise 20: HAO orchestrator implementation (8-10 hours)
- Coordinate agents in sequence or parallel
- Aggregate confidence scores
- Generate reasoning chain (interpretable output for clinician)
- Error handling & fallback strategies

# Exercise 21: End-to-end testing (4-5 hours)
- Simulate 10 patient scenarios (diverse rare diseases)
- For each: Does system reach correct diagnosis?
- Document reasoning chain transparency
- Benchmark against baseline (ACMG alone)
```

**Output:**
- Fully working multi-agent system
- Reasoning chain documentation for 10 test cases
- System architecture document (technical + clinical motivation)

---

### Phase 3: Prospective Study Design & IRB Preparation (Jun 2026 - 2-3 weeks)

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
# - Statistical analysis plan (include ancestry-stratified analysis)
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
# - Ancestry diversity in Mayo patient population?
```

**Output:**
- Complete prospective study protocol draft (ready for IRB)
- Enrollment timeline projections
- Statistical power analysis (with ancestry stratification)
- Meeting notes with Eric Klee on feasibility

---

### Track C Summary (By Jun 30, 2026):
- ✅ Deep understanding of competitive landscape
- ✅ Working multi-agent HAO system
- ✅ Prospective study protocol ready for IRB
- **Time investment: ~40-50 hours**