# Week 1 Study Plan: Oct 27 - Nov 2, 2025
## Track A Phase 1 Kickoff: Attention Visualization on Your Models

---

## OVERVIEW FOR WEEK 1

**Goal:** Set up infrastructure + read foundational papers + begin ESM2 attention extraction

**Time budget:**
- Weekday mornings (Mon-Fri): 1.5 hrs each = 7.5 hrs total
- Saturday: 2 × 1.5 hr blocks = 3 hrs
- Sunday: 2-2.5 hr block = 2-2.5 hrs
- **Total: ~12.5-13 hrs**

**Deliverable by Sunday:** 
- GitHub repo initialized with clean structure
- ESM2 model loaded and tested
- Attention extraction pipeline started (not complete, but framework in place)
- Reading notes on Enformer attention + Olah feature viz

---

## WEEKDAY MORNINGS (Mon Oct 27 - Fri Oct 31) - 1.5 hrs each

### Monday Oct 27 (1.5 hours) - Setup & Planning

**What to do:**
1. **Create GitHub repo structure (20 min)**
   ```bash
   git init pre-phd-genomics
   # Create directories:
   # 01_interpretability/
   # 02_genomics_domain/
   # 03_integrated_projects/
   # 04_competitive_analysis/
   # 05_clinical_context/
   # docs/
   # README.md (overview of study plan)
   ```

2. **Setup Python environment (30 min)**
   ```bash
   # Create virtual environment
   python -m venv phd_study
   source phd_study/bin/activate
   
   # Install core packages
   pip install torch transformers numpy pandas jupyter matplotlib seaborn
   pip install scikit-learn shap lime
   
   # Create requirements.txt
   pip freeze > requirements.txt
   ```

3. **ESM2 model check (40 min)**
   - Verify you can load ESM2 from HuggingFace
   - Test on toy DNA sequence (100bp)
   - Confirm GPU availability
   - Document setup in `docs/environment_setup.md`

**Output:** GitHub repo live, environment ready, ESM2 loads

---

### Tuesday Oct 28 (1.5 hours) - Reading + Prep

**What to do:**
1. **Read: Avsec et al. 2021 (Enformer) Sections 3-4 (1 hour)**
   - Focus: How they visualize attention for biology
   - How to interpret heatmaps for biological signal
   - Take notes in `docs/reading_notes/enformer_attention.md`
   - Key questions to answer in notes:
     - How do they visualize attention across layers?
     - What biological patterns do they find?
     - Why is attention useful for rare disease variants?

2. **Create analysis plan doc (30 min)**
   - Open `docs/week1_analysis_plan.md`
   - Document: For ESM2, what biological signals should we expect?
   - Example: Attention to TFBS sites? CpG islands? Splice sites?
   - For encoder-decoder bottleneck: What should it compress?

**Output:** Reading notes + analysis plan document

---

### Wednesday Oct 29 (1.5 hours) - Reading + Notebook Setup

**What to do:**
1. **Read: Olah et al. 2017 "Feature Visualization" (30 min)**
   - Blog post: https://distill.pub/2017/feature-visualization/
   - Focus: Making interpretability accessible to non-ML
   - Take notes: How do we explain attention heatmaps to clinicians?
   - Save to `docs/reading_notes/feature_visualization.md`

2. **Create first Jupyter notebook (1 hour)**
   - File: `01_interpretability/00_ESM2_attention_setup.ipynb`
   - Cells:
     * Load ESM2 model
     * Test on 3 toy DNA sequences (100bp each)
     * Extract attention from last layer, last head
     * Visualize as simple heatmap (position × position attention)
     * Don't worry about biological interpretation yet—just get it working
   - Save to repo

**Output:** Setup notebook with basic attention extraction on ESM2

---

### Thursday Oct 30 (1.5 hours) - Continuing Setup

**What to do:**
1. **Debug + fix ESM2 attention extraction (45 min)**
   - Run notebook from Wed, fix any issues
   - Can you extract attention matrices from multiple layers?
   - Can you aggregate across heads?
   - Add utility functions to `01_interpretability/utils.py`:
     ```python
     def extract_attention(model, sequence, layer_idx, head_idx=None):
         # Extract from specific layer/head
         pass
     
     def aggregate_heads(attention_tensor):
         # Average across heads for visualization
         pass
     ```

2. **Test on real variant data (45 min)**
   - Pick 1 known pathogenic variant (e.g., BRCA1 p.R1699W)
   - Get genomic sequence context (500bp window around variant)
   - Run through ESM2, extract attention
   - Save output to `01_interpretability/data/variant1_attention.pkl`
   - Document: Did extraction work? What does attention look like?

**Output:** Working attention extraction on real variants

---

### Friday Oct 31 (1.5 hours) - Documentation + Reflection

**What to do:**
1. **Documentation pass (45 min)**
   - Add docstrings to all functions in utils.py
   - Add markdown cells in notebook explaining each step
   - Create `01_interpretability/README.md` describing what's done so far

2. **Weekly reflection (45 min)**
   - Document in `docs/week1_reflection.md`:
     * What worked well this week?
     * What was harder than expected?
     * ESM2 attention extraction: Did it reveal anything interesting?
     * Plans for weekend deep dive
   - Commit all code to GitHub

**Output:** Clean, documented code ready for weekend work

---

## WEEKEND

### Saturday Nov 1 - Morning Block 1 (1.5 hours) - Deep ESM2 Analysis

**What to do:**
1. **Systematic attention analysis on ESM2 (1.5 hours)**
   - Create: `01_interpretability/01_ESM2_attention_analysis.ipynb`
   - Load 10 variants: 5 pathogenic, 5 benign
   - Extract attention from each layer (4-6 layers typically)
   - Questions to answer:
     * Do pathogenic variants have different attention patterns than benign?
     * Which positions get highest attention? Do they align with known functional domains?
     * Does attention become more "focused" in deeper layers?
   - Visualize: Heatmaps + summary statistics
   - Save attention data and plots to outputs directory

**Output:** ESM2 attention analysis on 10 variants

---

### Saturday Nov 1 - Morning Block 2 (1.5 hours) - Encoder-Decoder Exploration

**What to do:**
1. **Load your proprietary encoder-decoder (30 min)**
   - Can you load model weights?
   - Test on toy input
   - Confirm model architecture (encoder → bottleneck → decoder)
   - Document in notebook: What is input/output format?

2. **Plan encoder-decoder attention extraction (1 hour)**
   - Create: `01_interpretability/02_encoder_decoder_exploration.ipynb`
   - Cells:
     * Load model
     * For 3 test inputs, extract:
       - Encoder attention (to genomic sequence)
       - Bottleneck representation (what dimensions? How to visualize?)
       - Decoder attention
     * Key question: **What does bottleneck learn?** 
       - Can you reconstruct biological signal from bottleneck alone?
       - Document first hypothesis
   - Don't worry about final analysis—just exploration

**Output:** Encoder-decoder loaded + bottleneck examined preliminarily

---

### Sunday Nov 2 - Extended Block (2-2.5 hours) - Integration + Week 2 Prep

**What to do:**

**Hour 1: Comparison analysis (1 hour)**
- Create: `01_interpretability/03_ESM2_vs_encoder_decoder_comparison.ipynb`
- Load outputs from Saturday work
- For 5 same variants, visualize:
  * ESM2 attention (last layer)
  * Encoder-decoder encoder attention
  * Side-by-side comparison
- Initial observations: Where do they agree/disagree?

**30 min: Documentation + cleanup**
- Write: `01_interpretability/WEEK1_SUMMARY.md`
  * What we learned about ESM2
  * What we learned about encoder-decoder
  * Open questions for Week 2
- Clean up notebook code (remove debug cells, add markdown)
- Commit everything to GitHub

**30 min: Plan Week 2**
- Sketch: `docs/week2_plan.md`
  * Continue encoder-decoder analysis (deeper)
  * Begin SHAP analysis (Track A Phase 2)
  * Start ACMG reading (Track B Phase 1)
  * First competitive paper read (optional if time)

**Output:** Comparison analysis + clean repo + Week 2 plan

---

## OPTIONAL EVENING READING (Flexible)

If you have evening time, pick one:

**Option A (Technical):**
- Start skimming DeepRare paper (arXiv June 25, 2025)
- Goal: Understand their multi-agent architecture
- Take notes in `docs/competitive_papers/deeprare_notes.md`
- Not required, but getting ahead is nice

**Option B (Domain):**
- Start reading Richards et al. 2015 (ACMG classification)
- Focus: First 5 pages + ACMG classification table
- Understand the 7 categories (PVS, PS, PM, PP, BS, BP)
- Take notes in `docs/reading_notes/acmg_intro.md`

**Option C (Relaxation):**
- Skip it, enjoy your weekend
- You've got 12.5 hrs in the week, that's solid

---

## GITHUB COMMIT CHECKLIST (By Sunday Evening)

- [ ] Repo structure created (all directories)
- [ ] `README.md` with study plan overview
- [ ] `requirements.txt` with dependencies
- [ ] `docs/environment_setup.md` (environment instructions)
- [ ] `docs/reading_notes/enformer_attention.md` (Enformer notes)
- [ ] `docs/reading_notes/feature_visualization.md` (Olah notes)
- [ ] `docs/week1_analysis_plan.md` (analysis plan)
- [ ] `docs/week1_reflection.md` (reflections)
- [ ] `01_interpretability/utils.py` (utility functions)
- [ ] `01_interpretability/00_ESM2_attention_setup.ipynb` (setup)
- [ ] `01_interpretability/01_ESM2_attention_analysis.ipynb` (ESM2 analysis on 10 variants)
- [ ] `01_interpretability/02_encoder_decoder_exploration.ipynb` (encoder-decoder exploration)
- [ ] `01_interpretability/03_ESM2_vs_encoder_decoder_comparison.ipynb` (comparison)
- [ ] `01_interpretability/WEEK1_SUMMARY.md` (week summary)
- [ ] `docs/week2_plan.md` (next week plan)

---

## SUCCESS CRITERIA FOR WEEK 1

**Hard criteria (must have):**
- [ ] ESM2 attention extraction working (can extract from any sequence)
- [ ] Encoder-decoder loaded and tested
- [ ] 10 variants analyzed with ESM2
- [ ] GitHub repo clean and organized

**Soft criteria (nice to have):**
- [ ] Initial hypothesis about bottleneck function
- [ ] Comparison between ESM2 and encoder-decoder visualized
- [ ] Started evening reading on DeepRare or ACMG

**If all hard criteria true → Week 1 success ✅**

---

## Notes for Execution

1. **Strict on timing**: Use a timer. When 1.5 hrs is up, stop (unless you're in middle of fixing critical bug). This trains efficiency.

2. **Git discipline**: Commit at end of each day (or morning), even if incomplete. Messages like "WIP: ESM2 attention setup", "Fix: Attention extraction shape bug", "Doc: ACMG notes".

3. **Notebook hygiene**: As you build, add markdown cells explaining logic. By end of week, each notebook should be readable by someone else (or you in 3 months).

4. **Ask for help early**: If ESM2 attention extraction doesn't work by Wednesday afternoon, ask (PyTorch forum, GitHub issues, etc.). Don't waste 3 days debugging.

5. **Backup**: Push to GitHub at end of each day. Cloud backup is your friend.

---

## If You Get Ahead / Behind

**Ahead (all tasks done by Friday):**
- Jump to SHAP analysis (early start on Phase 2)
- Or read competitive paper

**Behind (stuck on encoder-decoder Wed):**
- Skip optional encoder-decoder work Friday, focus on ESM2 completion
- Pick up encoder-decoder Sunday with fresh eyes
- Plan to finish Week 1 objectives early Week 2

---

**Week 1 is foundation-laying. You're setting up repo structure, getting comfortable with your models, and reading just enough theory. By Sunday, you'll have clean infrastructure + first real analysis outputs. That's a win.**

Ready to launch Monday morning?
