# Genomic Foundation Model Selection Framework
## Decision Point: Track B Phase 2 (January 2026)

---

## Context

**Why this decision point exists:**
- Original plan specified Evo2, but usability issues emerged
- Genomic foundation model landscape is rapidly evolving
- Deferring selection to Jan 2026 allows:
  1. Learning interpretability fundamentals with ESM2 (proteins) first
  2. Evaluating newest models released in late 2025
  3. Testing multiple candidates for usability and performance
  4. Making informed decision based on thesis requirements

**Timeline:**
- **Now (Oct-Dec 2025):** Track A uses ESM2 for interpretability learning
- **Jan 2026 (Track B Phase 2):** Evaluate and select genomic model
- **Feb-Apr 2026:** Implement selected model in Projects 1-2

---

## Candidate Models

### 1. Enformer (DeepMind, 2021)
**Strengths:**
- 100kb context window (captures long-range regulatory interactions)
- Strong performance on regulatory element prediction
- Well-documented attention visualization (Avsec et al. 2021)
- Published in Nature, widely cited

**Weaknesses:**
- Large model (compute intensive)
- Focused on regulatory variants (may be less useful for coding variants)
- Released 2021 (may be superseded)

**Use case fit:**
- Excellent for non-coding regulatory variants
- Good for enhancer-promoter interactions
- May need to combine with protein model (ESM2) for coding variants

---

### 2. Nucleotide Transformer (InstaDeep, 2023)
**Strengths:**
- General-purpose DNA foundation model
- Handles sequences up to 1000bp
- Pre-trained on diverse genomic regions
- More recent than Enformer

**Weaknesses:**
- Shorter context than Enformer (1kb vs 100kb)
- Less proven for rare disease variant interpretation
- Documentation quality TBD

**Use case fit:**
- Good for coding + nearby regulatory regions
- May struggle with long-range regulatory interactions
- Versatile for multiple variant types

---

### 3. Hyena-DNA (Stanford, 2023)
**Strengths:**
- 1 million token context (longest available)
- Efficient architecture (subquadratic attention)
- Fast inference
- State-of-the-art on several genomic benchmarks

**Weaknesses:**
- Novel architecture (less familiar)
- Interpretability unknown (can we extract attention?)
- Newer, less battle-tested

**Use case fit:**
- Best for very long-range interactions
- Efficiency good for large-scale deployment
- May require custom interpretability methods

---

### 4. DNABERT-2 (2023)
**Strengths:**
- BERT-style architecture (familiar)
- Efficient tokenization
- Good performance on variant effect prediction

**Weaknesses:**
- Context limited to ~512bp
- Less suitable for regulatory variants

**Use case fit:**
- Good for coding variants and splice sites
- Less suitable for enhancers and distal regulatory elements

---

### 5. Your Dual-Llama Encoder-Decoder
**If genomic (not protein):**

**Strengths:**
- Proprietary, you understand it deeply
- Bottleneck architecture (unique interpretability)
- Already integrated into your workflow

**Weaknesses:**
- May lack pretraining on genomic data
- Documentation/support limited to you
- Validation burden higher

**Use case fit:**
- Strong if pre-trained on genomic sequences
- Bottleneck interpretability unique advantage
- Requires benchmark comparison to public models

---

## Evaluation Framework (Jan 2026)

### Phase 1: Quick Screen (1-2 hours per model)

**For each candidate:**

1. **Installation test (30 min)**
   ```python
   # Can you install?
   # Dependencies manageable?
   # GPU requirements?
   # Documentation quality?
   ```

2. **Basic functionality (30-45 min)**
   ```python
   # Load model
   # Test on 3 genomic sequences (500bp, 2kb, 10kb)
   # Extract embeddings
   # Inference time and memory usage
   ```

3. **Quick decision (15 min)**
   - If installation fails or documentation poor → SKIP
   - If basic functionality broken → SKIP
   - Otherwise → proceed to Phase 2

**Expected result:** 2-3 candidates advance to Phase 2

---

### Phase 2: Deep Evaluation (3-4 hours per model)

**For advanced candidates:**

1. **Attention extraction (1-1.5 hours)**
   ```python
   # Can you extract attention matrices?
   # Layer-wise access?
   # Head-wise access?
   # Aggregation methods available?
   # Visualization pipeline (heatmaps)?
   
   # Test on 5 sequences
   # Document: Is attention biologically interpretable?
   ```

2. **Variant effect prediction (1.5-2 hours)**
   ```python
   # Test on 10 variants (5 pathogenic, 5 benign)
   # Can model discriminate pathogenic vs benign?
   # Does attention align with known functional regions?
   
   # Compute simple metrics:
   # - Embedding distance (wild-type vs mutant)
   # - Attention disruption score
   # - Classification accuracy (if model has prediction head)
   ```

3. **Integration feasibility (30-45 min)**
   ```python
   # Can you combine with ESM2?
   # Can you integrate into Project 1-2 pipelines?
   # API/interface clean?
   # Batch processing support?
   ```

**Expected result:** 1-2 finalists

---

### Phase 3: Final Decision (2-3 hours)

**For finalists:**

1. **Create comparison matrix**

| Criterion                 | Weight | Model A  | Model B  | Model C  |
| ------------------------- | ------ | -------- | -------- | -------- |
| **Usability**             | 20%    |          |          |          |
| - Installation ease       |        | ★★★★★    | ★★★☆☆    | ★★★★☆    |
| - Documentation quality   |        | ★★★★☆    | ★★★★★    | ★★★☆☆    |
| - API design              |        | ★★★★☆    | ★★★★☆    | ★★★★★    |
| **Performance**           | 30%    |          |          |          |
| - Inference speed         |        | ★★★☆☆    | ★★★★★    | ★★★★☆    |
| - Memory efficiency       |        | ★★★☆☆    | ★★★★☆    | ★★★★★    |
| - Variant discrimination  |        | ★★★★★    | ★★★★☆    | ★★★★☆    |
| **Interpretability**      | 25%    |          |          |          |
| - Attention extraction    |        | ★★★★★    | ★★★☆☆    | ★★★★★    |
| - Biological alignment    |        | ★★★★☆    | ★★★★☆    | ★★★☆☆    |
| **Context & Scope**       | 15%    |          |          |          |
| - Context length          |        | ★★★★★    | ★★★☆☆    | ★★★★☆    |
| - Variant type coverage   |        | ★★★★☆    | ★★★★★    | ★★★☆☆    |
| **Support & Maintenance** | 10%    |          |          |          |
| - Active development      |        | ★★★★☆    | ★★★★★    | ★★★☆☆    |
| - Community support       |        | ★★★★★    | ★★★☆☆    | ★★★★☆    |
| **WEIGHTED SCORE**        |        | **X.XX** | **X.XX** | **X.XX** |

2. **Write decision document**
   ```markdown
   # Genomic Foundation Model Selection: Decision Document
   
   ## Final Selection: [MODEL NAME]
   
   ## Rationale:
   - [Why this model best fits thesis requirements]
   - [Key advantages over alternatives]
   - [How it complements ESM2 for protein-level analysis]
   
   ## Integration Plan:
   - [How to use in Project 1: Variant Effect Prediction]
   - [How to use in Project 2: Phenotype-Driven Ranking]
   - [Interpretability pipeline design]
   
   ## Known Limitations:
   - [What this model doesn't do well]
   - [Mitigation strategies]
   
   ## Alternative considered:
   - [Second-choice model + why not selected]
   ```

3. **Commit decision**
   - Save to `03_genomic_models/genomic_model_selection.md`
   - Update Track B plan to reference selected model
   - Proceed with implementation in Projects 1-2

---

## Decision Criteria Weights (Rationale)

**Performance (30%):**
- Most important: Model must work well for variant interpretation
- If it doesn't discriminate pathogenic vs benign → unusable
- Speed matters for large-scale deployment (prospective trial)

**Interpretability (25%):**
- Critical for thesis: Aim 1 emphasizes interpretability
- Attention visualization required for clinical explanations
- Biological alignment validates model learns meaningful patterns

**Usability (20%):**
- Practical consideration: Can't spend weeks debugging
- Poor documentation = high time cost
- Clean API enables faster iteration

**Context & Scope (15%):**
- Context length determines variant types you can handle
- Longer context better for regulatory variants
- But coding variants only need ~1kb

**Support & Maintenance (10%):**
- Active development means bug fixes and improvements
- Community support helps with debugging
- Less critical than other factors for PhD timeline

---

## Contingency Plans

**If no model meets requirements:**
- **Option A:** Use combination approach
  - ESM2 for protein-coding variants
  - Enformer for regulatory variants
  - Integration layer combines predictions

- **Option B:** Use your encoder-decoder exclusively
  - Requires additional validation on genomic sequences
  - May need to retrain/fine-tune on genomic data
  - Bottleneck interpretability becomes key differentiator

- **Option C:** Wait for better models
  - Delay genomic work by 1-2 months
  - Continue with protein-level analysis (ESM2)
  - Re-evaluate in March 2026

**If top two models are tied:**
- Test both in Project 1 (parallel implementation)
- Compare results on same 10 variants
- Choose based on empirical performance

---

## Success Criteria

**By end of Track B Phase 2 (Feb 2026), you should have:**
- [ ] Selected genomic foundation model
- [ ] Documented selection rationale
- [ ] Attention extraction pipeline working
- [ ] Tested on 10+ variants
- [ ] Integrated into Project 1 codebase
- [ ] Comparison to ESM2 documented

**If all true → Ready to proceed with genomic variant interpretation work**

---

## Notes

1. **Don't rush this decision:** Taking 1 week (6-8 hours) to evaluate thoroughly is worth it. Wrong model choice costs much more time later.

2. **Document everything:** Even rejected models should have notes on why they failed. Helps with paper Methods section.

3. **Get feedback:** Share evaluation matrix with Eric Klee lab or GoodFire colleagues before final decision.

4. **Stay flexible:** If a new model releases in late 2025 that looks promising, evaluate it. Field moves fast.

5. **Remember the goal:** Model is a tool for thesis, not the thesis itself. Choose based on what enables best prospective validation story.

---

**This decision point is strategic. By Jan 2026, you'll have ESM2 experience, deeper genomics knowledge, and can make an informed choice for your thesis success.**
