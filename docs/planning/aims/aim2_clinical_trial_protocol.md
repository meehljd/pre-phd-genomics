# Aim 2: Prospective Diagnostic-Impact RCT
## AI-Assisted Diagnosis for Rare Genetic Disease

**Timeline:** Months 12-36 (Aug 2027 - Aug 2029)  
**Phase:** Clinical validation  
**Output:** Paper 2 - Prospective clinical utility (target: *NEJM* / *Nature Medicine*, 2029-2030)

---

## I. OBJECTIVE

**Test whether AI-assisted genomic analysis increases diagnostic yield and diagnostic quality for undiagnosed rare disease patients compared with standard care in a prospective randomized controlled trial.**

---

## II. HYPOTHESIS

### Primary Hypothesis (H1a)
AI-assisted genomic analysis increases **diagnostic yield within 90 days by ≥15% (absolute difference)** compared with standard care.

### Secondary Hypotheses
- **H2:** AI-assisted diagnosis reduces median **time-to-diagnosis** by ≥30 days
- **H3:** AI-assisted diagnosis increases **proportion of clinically actionable diagnoses** by ≥10% (absolute)
- **H4:** AI-assisted diagnosis achieves **higher phenotype resolution** (% of HPO terms explained)
- **H5:** AI-assisted diagnosis has **equivalent or better diagnostic certainty** (% Pathogenic vs Likely Pathogenic)

---

## III. TRIAL DESIGN

### Study Type
**Stepped-Wedge Cluster Randomized Controlled Trial**

**Cluster definition:** Clinical care team (attending geneticist + genetic counselor + coordinator)
- Number of clusters: 6-8 teams at Mayo Rochester
- Cluster size: ~20-25 patients per team over 24 months

**Rationale for stepped-wedge:**
1. Operationally smooth (all teams eventually get AI access)
2. Reduces contamination (clinicians learn gradually, not immediately)
3. Ethical appeal (no permanent control group)
4. Allows temporal trend adjustment

### Randomization Schedule

**Timeline:** 24 months total enrollment

| Period | Months | Teams in Control | Teams in Intervention |
|--------|--------|------------------|----------------------|
| 1      | 1-3    | 8                | 0                    |
| 2      | 4-6    | 6                | 2                    |
| 3      | 7-9    | 4                | 4                    |
| 4      | 10-12  | 2                | 6                    |
| 5      | 13+    | 0                | 8                    |

**Crossover:** Every 3 months, 2 randomly selected teams transition from control to intervention

---

## IV. STUDY ARMS

### Control Arm
**Standard diagnostic workflow + AI silent background**

**Process:**
1. Patient undergoes WES/WGS as part of standard care
2. Genetic counselor + geneticist review variants using standard tools (VarSeq, GenomOncology, etc.)
3. AI system runs silently in background (predictions logged but NOT shown to clinicians)
4. Standard interpretation turnaround: 4-6 weeks
5. Clinical report issued based on standard analysis only

**Purpose of silent AI:**
- Collect ground-truth AI predictions on control patients
- Enable direct comparison of AI vs standard care on identical population
- No ethical concerns about withholding care (AI unproven at trial start)

### Intervention Arm
**Standard diagnostic workflow + AI decision support (add-on mode)**

**Process:**
1. Patient undergoes WES/WGS as part of standard care
2. **AI system analyzes data within 48 hours** of sequence upload
3. AI output presented to clinical team via secure portal:
   - **Risk scores** (0-100) for top 5-10 candidate genes
   - **Phenotype match scores** (how well gene explains patient's HPO terms)
   - **Variant-level details** (ACMG classification, population frequency, in silico predictions)
   - **Interpretability features:**
     - Attention heatmaps (which genomic regions drove prediction)
     - Counterfactual explanations (what would change prediction)
     - Evidence summary (key papers, functional studies)
4. Clinicians review AI output **alongside** standard analysis
5. **Clinicians retain final decision authority** (can accept, modify, or reject AI recommendation)
6. Clinical report issued with clinician's final interpretation

**UI Design:**
- Succinct summary view (1 page)
- Expandable details (full report with evidence)
- Clear labeling: "AI-assisted analysis. Clinician judgment is final."
- Override logging: System records when clinician follows vs ignores AI

---

## V. POPULATION

### Inclusion Criteria
1. Age ≥18 years
2. Enrolled in Mayo Clinic Undiagnosed Diseases Program
3. Clinical suspicion of rare genetic disease (Mendelian inheritance pattern)
4. Undergoing whole exome sequencing (WES) or whole genome sequencing (WGS)
5. No prior confirmed molecular diagnosis for presenting condition
6. Willing to provide informed consent

### Exclusion Criteria
1. Prior confirmed genetic diagnosis for presenting condition
2. Participation in this trial within past 12 months
3. Opted out of research data use
4. Conditions where AI is known to be unreliable (pre-specified):
   - Somatic mosaicism (cancer variants)
   - Complex structural variants (>100kb)
   - Mitochondrial genome disorders
   - Repeat expansion disorders (e.g., Huntington's, fragile X)
5. Prisoners or other vulnerable populations per Mayo IRB

### Sample Size Calculation

**Preliminary power calculation:**

**Assumptions:**
- Baseline diagnostic yield (control): 30% (to be confirmed with Mayo data)
- Target diagnostic yield (intervention): 45% (Δ = 15% absolute increase)
- Intracluster correlation coefficient (ICC): 0.05 (estimated from silent-run pilot)
- Cluster size: 20 patients per team
- Design effect (DE): 1 + (m-1) × ICC = 1 + (20-1) × 0.05 = 1.95
- Alpha: 0.05 (two-sided)
- Power: 80%

**Sample size (patient-level parallel RCT):** ~240 patients total (120 per arm)

**Adjusted for stepped-wedge design:** 
- Stepped-wedge efficiency factor: ~0.6-0.7 (depends on exact schedule)
- Adjusted N: 240 / 0.65 ≈ 370 patients
- With 8 clusters × 24 months: ~185 patients expected
- **Conservative target: 180-200 patients** (allows 10% loss to follow-up)

**Note:** Final sample size requires:
1. Mayo baseline yield data from retrospective cohorts
2. ICC estimate from 3-6 month silent-run pilot (Aug 2027)
3. Verification of enrollment rate (7-8 patients/month feasible?)

### Enrollment
**Target:** 180 patients over 24 months (Sep 2027 - Aug 2029)
**Rate:** 7.5 patients/month across all teams (~1 patient per team per month)
**Feasibility check:** Requires validation with Eric Klee's current enrollment data

---

## VI. ENDPOINTS

### Primary Endpoint

**Diagnostic yield at 90 days** (binary outcome)

**Definition:** Proportion of patients with reference-standard confirmed diagnosis within 90 days of enrollment

**Reference standard criteria (hierarchy):**
1. **Genetic confirmation (preferred):**
   - Pathogenic (P) or Likely Pathogenic (LP) variant per ACMG/AMP 2015 guidelines
   - Variant confirmed by orthogonal method (Sanger sequencing)
   - Segregation analysis in family (if available)
   - Functional validation (if applicable)

2. **Expert consensus (for ambiguous cases):**
   - Panel of ≥3 board-certified clinical geneticists (blinded to arm assignment)
   - Review clinical phenotype + all genomic findings
   - Majority vote required
   - Graded certainty: Definite / Probable / Possible (only Definite/Probable count as diagnosis)

**Assessment timing:** 90 days post-enrollment for each patient

---

### Secondary Endpoints: Effectiveness

**1. Time-to-diagnosis** (time-to-event)
- Days from enrollment to reference-standard confirmed diagnosis
- Censored at 90 days (primary) or 180 days (extended follow-up)
- Compare median time: intervention vs control

**2. Time-to-actionable step** (time-to-event)
- Days from enrollment to first actionable clinical decision:
  - Ordering confirmatory genetic test
  - Initiating disease-specific treatment
  - Referring to specialist
  - Changing medication regimen
- Chart review to identify date of action

**3. Diagnostic yield at 180 days** (binary)
- Extended follow-up to capture delayed diagnoses
- Some diagnoses require additional testing, functional studies, or literature review

---

### Secondary Endpoints: Accuracy Metrics

**4. Sensitivity and specificity** (at locked AI threshold)
- True positive rate and true negative rate vs reference standard
- Calculated at pre-specified AI risk score threshold (e.g., ≥75)

**5. Positive predictive value (PPV) and negative predictive value (NPV)**
- Clinical relevance: If AI flags a gene, what's probability it's correct?

**6. Area under ROC curve (AUC)**
- Overall discriminative ability of AI system
- Compare intervention vs control accuracy

---

### Secondary Endpoints: Diagnostic Quality (NEW)

**7. Clinical actionability rate** ⭐ KEY QUALITY METRIC
- Proportion of confirmed diagnoses classified as "clinically actionable"
- **Actionability tiers (ACMG framework):**
  - **Tier 1 (High):** Treatment available that alters outcomes
    - Example: PKU → dietary intervention prevents disability
    - Example: Cardiomyopathy gene → ICD prevents sudden death
  - **Tier 2 (Moderate):** Management changes but no cure
    - Example: Mitochondrial disorder → medication avoidance
    - Example: Connective tissue disorder → activity restrictions
  - **Tier 3 (Low):** Reproductive/family planning value only
    - Example: Carrier status → prenatal testing options

**Measurement:**
- Blinded adjudication panel assigns actionability tier for each diagnosis
- Primary comparison: % Tier 1-2 (high/moderate) in intervention vs control
- **Hypothesis:** AI finds more actionable diagnoses (not just any diagnosis)

**8. Phenotype resolution score** ⭐ KEY QUALITY METRIC
- **Definition:** Percentage of patient's HPO terms explained by identified causal variant(s)
- **Calculation:** (# HPO terms explained by variant) / (total # HPO terms for patient) × 100%
- **Example:**
  - Patient has 12 HPO terms (neurological + cardiac + skeletal)
  - Identified gene explains 10/12 terms → 83% resolution
- **Quality thresholds:**
  - High quality: ≥90% resolution
  - Medium quality: 70-89% resolution
  - Low quality: <70% resolution (suggests wrong diagnosis or oligogenic disease)
- **Hypothesis:** AI achieves higher mean phenotype resolution (finds genes that better explain full phenotype)

**9. Diagnostic certainty distribution** ⭐ KEY QUALITY METRIC
- **Definition:** Distribution of ACMG classifications among confirmed diagnoses
- **Categories:**
  - Pathogenic (P) - highest certainty
  - Likely Pathogenic (LP) - strong evidence but not definitive
  - (VUS do not count as diagnosis)
- **Metric:** % of diagnoses that are P (not just LP)
- **Hypothesis:** AI has equivalent or higher % P diagnoses (more certain, less ambiguous)

**10. Clinical utility score** ⭐ KEY QUALITY METRIC
- **Definition:** Clinician-rated usefulness of diagnosis
- **Measurement tool:** 5-item survey administered to treating clinician at 90 days post-diagnosis
  - "How helpful was this diagnosis for clinical management?" (1-5 Likert scale)
  - "Did this diagnosis change your treatment plan?" (yes/no)
  - "Did this diagnosis provide value to your patient?" (1-5 scale)
  - "Would you have made this diagnosis without AI?" (yes/no/uncertain) [intervention arm only]
  - "How confident are you in this diagnosis?" (1-5 scale)
- **Composite score:** Mean of Likert items (1-5 scale)
- **Hypothesis:** AI-assisted diagnoses have equal or higher utility scores

**11. Diagnostic concordance** ⭐ KEY QUALITY METRIC
- **Inter-rater reliability:** Agreement among adjudication panel members
  - Fleiss' kappa for multi-rater agreement
  - Higher kappa = more consensus/less ambiguous diagnoses
- **AI-expert concordance:** Does AI top-ranked gene match expert final diagnosis?
  - Exact match (gene level)
  - Top-5 match
  - Complete mismatch
- **Hypothesis:** AI has high concordance with expert consensus

**12. Follow-up reclassification rate** (1-year timepoint)
- **Definition:** % of diagnoses that are reclassified at 12 months post-diagnosis
- **Reclassification types:**
  - P → LP or VUS (downgrade)
  - LP → P (upgrade) or VUS (downgrade)
  - New evidence in ClinVar or literature
- **Data source:** Quarterly ClinVar checks, literature review
- **Hypothesis:** AI diagnoses have similar or lower reclassification rate (more stable/robust)

---

### Secondary Endpoints: Clinical Utilization

**13. Number of downstream tests ordered**
- Count of additional genetic tests ordered post-diagnosis attempt
- Types: Single-gene sequencing, gene panels, RNA-seq, functional assays, biopsies
- Comparison: Mean tests per patient (intervention vs control)

**14. Cost of downstream testing**
- Total cost of downstream diagnostic tests per patient
- Data source: Mayo billing data
- Hypothesis: AI reduces unnecessary testing (fewer tests needed to reach diagnosis)

**15. Time to treatment initiation** (for actionable diagnoses)
- Days from diagnosis to initiation of disease-specific treatment
- Subset analysis: Only patients with actionable diagnoses
- Hypothesis: AI enables faster treatment initiation

---

### Secondary Endpoints: Safety

**16. False-positive triggered testing cascades**
- **Definition:** Patients who underwent unnecessary procedures due to false-positive AI prediction
- **Measurement:** Chart review for:
  - Invasive tests ordered based on AI flag
  - No confirmed diagnosis in that gene
  - Patient experienced harm (adverse event, anxiety, cost)
- **Threshold for concern:** >10% of intervention arm patients experience cascade

**17. False-negative harms**
- **Definition:** Missed diagnoses with clinical consequences
- **Measurement:**
  - Patients with no diagnosis at 90 days but later confirmed diagnosis at 180+ days
  - Delayed treatment initiation
  - Clinical deterioration during delay
- **Assessment:** Blinded review by safety committee

**18. Adverse events attributed to AI**
- Any serious adverse event potentially related to AI recommendation
- Examples: Wrong treatment initiated, delayed correct treatment, psychological harm
- Reviewed by Data & Safety Monitoring Board (DSMB)

---

### Secondary Endpoints: Equity & Fairness

**19. Diagnostic yield by ancestry**
- Stratified analysis: Yield in intervention vs control for each ancestry group
- **Ancestry groups:**
  - Self-reported race/ethnicity (standard NIH categories)
  - Genetic ancestry (PCA-based clustering from WES/WGS data)
- **Hypothesis:** AI maintains or improves equity (no worse performance in underrepresented groups)

**20. Accuracy metrics by ancestry**
- Sensitivity, specificity, PPV, NPV stratified by ancestry
- Identify disparities in AI performance

**21. Time-to-diagnosis by ancestry**
- Median time stratified by ancestry group
- Detect delays in specific populations

**22. Diagnostic yield by sex and age**
- Stratified analysis by sex (male/female) and age group (<40, 40-60, >60)

---

### Secondary Endpoints: Process Measures

**23. Clinician adherence rate**
- **Definition:** % of time clinicians follow AI recommendation
- **Measurement:**
  - Override logging: Did clinician include AI top-ranked gene in final report?
  - Adherence types:
    - Full acceptance (AI gene = final diagnosis)
    - Partial acceptance (AI gene included in differential)
    - Rejection (AI gene not mentioned in report)
- **Note:** Not an efficacy endpoint (clinician judgment is gold standard)
- **Purpose:** Understand AI uptake and trust

**24. Time spent reviewing AI output**
- Self-reported by clinicians (survey at end of trial)
- Mean minutes per case
- Assess workflow burden

**25. Clinician satisfaction**
- Post-trial survey (5-point Likert scale):
  - "AI output was easy to interpret"
  - "AI output was helpful for decision-making"
  - "I would use this AI system in routine practice"
  - "AI output was trustworthy"
  - "AI improved my diagnostic confidence"

---

## VII. STATISTICAL ANALYSIS PLAN

### Estimand
**Primary estimand:** Effect of AI visibility on diagnostic yield at 90 days under intention-to-treat (ITT) principle

### Analysis Populations
1. **ITT population** (primary): All enrolled patients analyzed in period they enrolled
2. **Per-protocol population** (secondary): Exclude major protocol deviations:
   - AI system unavailable >48 hours (intervention arm)
   - Clinician inadvertently saw AI output (control arm)
3. **Safety population:** All enrolled patients

### Primary Analysis

**Model:** Generalized linear mixed model (GLMM) for binary outcome

```
logit(P(Diagnosis_i = 1)) = β₀ + β₁·Intervention_i + β₂·Time_Period_i + u_j

Where:
- i = patient index
- j = team (cluster) index
- Intervention_i = 1 if AI visible, 0 if control
- Time_Period_i = calendar time (to adjust for temporal trends)
- u_j ~ N(0, σ²) = random intercept for team
```

**Software:** R (lme4::glmer) or SAS (PROC GLIMMIX)

**Output:**
- Adjusted odds ratio (OR) with 95% CI
- Convert to absolute risk difference for interpretation
- P-value (two-sided, α=0.05)

**Power:** 80% to detect 15% absolute difference (30% → 45% yield)

### Secondary Analyses

**Time-to-event outcomes (time-to-diagnosis, time-to-actionable step):**
- **Model:** Cox proportional hazards with frailty term for cluster
```
h(t) = h₀(t) · exp(β₁·Intervention + β₂·Time_Period + u_j)
```
- **Censoring:** At 90 days (primary) or 180 days (extended)
- **Output:** Hazard ratio, 95% CI, median time difference (days)

**Continuous outcomes (phenotype resolution, utility score, # tests, cost):**
- **Model:** Linear mixed model (LMM)
```
Y_i = β₀ + β₁·Intervention_i + β₂·Time_Period_i + u_j + ε_i
```
- **Output:** Mean difference, 95% CI, p-value

**Binary secondary endpoints (actionability, diagnostic certainty):**
- **Model:** Same GLMM as primary analysis
- **Output:** Odds ratio or risk difference

**Accuracy metrics (sensitivity, specificity, PPV, NPV, AUC):**
- Calculated separately for intervention arm (AI predictions) vs control arm (standard care)
- 95% CIs using Wilson score method or DeLong method (AUC)
- No mixed model needed (cluster effect minimal for accuracy metrics)

**Fairness analyses (stratified by ancestry):**
- **Interaction model:**
```
logit(P(Diagnosis = 1)) = β₀ + β₁·Intervention + β₂·Ancestry + β₃·(Intervention × Ancestry) + β₄·Time_Period + u_j
```
- Test β₃ interaction term (does AI effect differ by ancestry?)
- Report stratified results even if interaction not significant

### Multiplicity Control

**Gatekeeping hierarchy** to preserve family-wise error rate:

**Step 1:** Test primary endpoint (diagnostic yield at 90 days) at α=0.05
- If p < 0.05 → proceed to Step 2
- If p ≥ 0.05 → stop, all other tests are exploratory

**Step 2:** Test key secondary endpoints (pre-specified):
1. Time-to-diagnosis (Cox model)
2. Clinical actionability rate
3. Phenotype resolution score

Use **Holm-Bonferroni** adjustment:
- Order p-values: p₁ ≤ p₂ ≤ p₃
- Test p₁ at α/3 = 0.0167
- If significant, test p₂ at α/2 = 0.025
- If significant, test p₃ at α = 0.05

**Step 3:** If key secondaries significant, test remaining secondaries with Benjamini-Hochberg FDR control at q=0.10

### Subgroup Analyses (Pre-Specified)

**No data dredging:** Only pre-specified subgroups analyzed

1. **Ancestry:** Diagnostic yield by ancestry group
2. **Age:** <40 vs 40-60 vs >60 years
3. **Disease category:** Neurological vs metabolic vs immunological vs other
4. **Phenotypic complexity:** Simple (≤5 HPO terms) vs complex (>5 HPO terms)

**Interaction test:** Report interaction p-value but do not require significance for reporting stratified results

### Missing Data

**Primary endpoint (diagnostic confirmation at 90 days):**
- **Assumption:** Missing = no diagnosis (conservative)
- **Sensitivity analysis #1:** Multiple imputation under MAR assumption
- **Sensitivity analysis #2:** Tipping point analysis (how many missing would need to be diagnosed to change conclusion?)

**Secondary endpoints:**
- Complete-case analysis (primary)
- Multiple imputation (sensitivity)

**Loss to follow-up:**
- Expected: <5% (Mayo patients typically remain in system)
- If >10% loss, conduct sensitivity analysis

### Interim Analysis

**Timing:** One interim analysis at 50% enrollment (~90 patients, Month 12)

**Purpose:**
- Safety monitoring (excess harms in intervention arm?)
- Futility check (conditional power <20% to detect effect?)

**Statistical boundary:**
- O'Brien-Fleming spending function (preserves overall α=0.05)
- Stopping boundary for efficacy: p < 0.001 (unlikely given sample size)
- Stopping boundary for futility: Conditional power < 0.20

**DSMB:** Reviews results, trial team remains blinded to outcome data

---

## VIII. REFERENCE STANDARD & ADJUDICATION

### Adjudication Committee

**Composition:**
- **Chair:** Senior Mayo clinical geneticist (not involved in trial enrollment)
- **Members:** 3-4 external clinical geneticists from:
  - Broad Institute
  - Baylor Genetics
  - Stanford / UCSF
  - GeneDx or Invitae (industry perspective)
- **Statistician (non-voting):** Tracks adjudication process, no access to arm assignment

**Blinding:** Committee members are blinded to:
- Arm assignment (control vs intervention)
- AI predictions and risk scores
- Clinician's final interpretation

**Training:**
- Standardized adjudication manual with decision rules
- 10 practice cases with discussion
- Inter-rater reliability check on practice set (target κ > 0.7)

### Adjudication Process

**Timeline:**
- **90 days post-enrollment:** Primary endpoint adjudication
- **180 days post-enrollment:** Extended follow-up adjudication

**Materials provided to panel (de-identified):**
- Patient phenotype (HPO terms)
- Genomic findings (VCF file, filtered variant list)
- Segregation data (if available)
- Functional studies (if performed)
- All clinical notes related to diagnostic workup
- **Excluded:** Arm assignment, AI output, clinician's interpretation

**Adjudication questions:**

1. **Is there a confirmed diagnosis?**
   - Yes (proceed to Q2-6)
   - No
   - Indeterminate (insufficient data)

2. **What is the causal gene?**
   - Gene name
   - Variant(s) identified
   - ACMG classification (P, LP, VUS)

3. **Certainty grade:**
   - Definite (genetic confirmation + functional validation)
   - Probable (genetic confirmation per ACMG P/LP)
   - Possible (expert consensus without genetic confirmation)

4. **Actionability tier:**
   - Tier 1: High (treatment-altering)
   - Tier 2: Moderate (management-altering)
   - Tier 3: Low (reproductive/family planning only)

5. **Phenotype resolution:**
   - List HPO terms explained by identified variant
   - Calculate % resolution

6. **Confidence rating:**
   - How confident are you in this diagnosis? (1-5 scale)

**Disagreement resolution:**
- Majority vote for binary decisions (diagnosis yes/no)
- Mean score for continuous ratings (phenotype resolution, confidence)
- If 2-2 tie (4 panel members), Chair casts deciding vote
- Document all disagreements for analysis

### Indeterminate Cases

**Handling:**
- Primary analysis: Treat as "no diagnosis" (conservative)
- Sensitivity analysis: Exclude indeterminate cases
- Report: Document number and reasons for indeterminate classification

---

## IX. SAFETY, ETHICS, & GOVERNANCE

### IRB & Regulatory

**IRB determination:** Interventional study (AI decision support may change diagnostic workup)
- **Risk level:** Minimal to moderate risk
- **Device status:** To be determined with FDA
  - Likely Non-Significant Risk (NSR) device
  - Or 510(k) exempt (clinical decision support)

**IRB submission:** October 2026
**Expected approval:** January 2027

### Data & Safety Monitoring Board (DSMB)

**Composition:**
- 1 biostatistician (external)
- 1 clinical geneticist (external, not Mayo)
- 1 ethicist or patient advocate

**Charter:**
- Review safety signals quarterly (Year 1), biannually (Year 2)
- Access to unblinded data
- Authority to recommend trial modification or termination

**Stopping rules:**
1. **Harm:** Excess false-negative or false-positive harms (p<0.01 at interim)
2. **Futility:** Conditional power <20% to detect meaningful effect
3. **Overwhelming benefit:** p<0.001 at interim (unlikely given sample size)

### Informed Consent

**Key elements:**
- Disclosure of AI use in intervention arm
- Right to refuse AI-assisted analysis (opt-out)
- No penalty for refusal (standard care continues)
- Data will be used for research
- Results may be published (de-identified)

**Consent process:**
- Presented by trained research coordinator (not treating clinician)
- Opportunity for questions
- Signed consent before enrollment

### Data Governance

**HIPAA compliance:**
- All data pulls follow Mayo HIPAA protocols
- De-identification for analysis datasets
- Re-identification key stored securely (access limited to PI/co-I)

**Role-based access:**
| Role | Access Level |
|------|--------------|
| PI (Eric Klee) | Full access (identified data) |
| Co-I (you) | Full access (identified data) |
| Statistician | Coded data only |
| Adjudication panel | De-identified clinical records |
| DSMB | Aggregate data (unblinded) |
| Clinical teams | Only their own patients |

**Audit trail:**
- All AI predictions logged with timestamp
- Clinician override decisions logged
- Data access logged (who, what, when)

**Data retention:**
- Clinical trial data: 7 years post-publication (Mayo policy)
- AI model artifacts: Permanent archive (reproducibility)

---

## X. PRE-TRIAL PREPARATIONS

### Silent-Run Pilot (Feb 2027 - Aug 2027)

**Purpose:**
1. Measure baseline diagnostic yield (control arm estimate)
2. Estimate intracluster correlation coefficient (ICC)
3. Verify AI system stability and performance
4. Detect contamination (are clinicians inadvertently using AI output?)

**Design:**
- 6 months, all patients (both treatment arms will exist once trial starts)
- AI runs silently on all patients
- Clinicians blinded to AI output
- Collect standard-of-care diagnostic outcomes

**Outcomes:**
- Baseline yield: % patients diagnosed within 90 days
- Time-to-diagnosis distribution
- ICC estimate (variance between teams)
- AI performance metrics (sensitivity, specificity vs retrospective cohorts)

**Decision:** Finalize sample size based on pilot data

### Model Freeze (July 2027)

**Critical:** AI model, thresholds, and calibration MUST be frozen before first patient enrolled

**Freeze checklist:**
- [ ] Model weights locked (version control commit hash documented)
- [ ] Decision thresholds set (e.g., risk score ≥75 for "high priority" flag)
- [ ] Calibration finalized (on Mayo retrospective cohorts 2020-2026)
- [ ] Software version pinned (no updates during trial)
- [ ] Validation tests passed (accuracy on held-out test set)
- [ ] Infrastructure deployed (Mayo secure servers, API integration)

**No changes allowed during trial period** (Sep 2027 - Aug 2029)

### Clinician Training (Aug 2027)

**Before trial launch, all teams complete:**

1. **Online module (30 min):**
   - AI system overview
   - How to interpret risk scores
   - Limitations and failure modes
   - When to trust vs override AI

2. **In-person workshop (30 min):**
   - Practice cases (5 examples with AI output)
   - Q&A session
   - Workflow integration

3. **Job aid:**
   - 1-page laminated quick reference
   - Risk score interpretation guide
   - Override decision tree

4. **Competency assessment:**
   - 3 test cases, must correctly interpret AI output
   - Pass criterion: 2/3 correct

**Ongoing support:**
- Weekly office hours (first month after team crossover)
- Help desk (email/phone)
- Monthly feedback sessions

---

## XI. IMPLEMENTATION LOGISTICS

### Timeline

| Milestone | Date | Responsible |
|-----------|------|-------------|
| Protocol finalized | Jun 2027 | Co-I (you) + PI |
| IRB approval | Jan 2027 | PI |
| Silent-run pilot starts | Feb 2027 | Co-I |
| Silent-run complete | Aug 2027 | Co-I |
| Model freeze | Jul 2027 | Co-I |
| DSMB charter signed | Aug 2027 | PI |
| Clinician training | Aug 2027 | Co-I |
| **Trial enrollment starts** | **Sep 2027** | Clinical teams |
| 50% enrollment (interim) | Mar 2028 | Clinical teams |
| 100% enrollment | Aug 2029 | Clinical teams |
| Last patient 90-day follow-up | Nov 2029 | Adjudication panel |
| Database lock | Dec 2029 | Co-I + Statistician |
| DSMB final review | Dec 2029 | DSMB |
| **Paper 2 submission** | **Jan 2030** | **Co-I (1st author)** |

### Enrollment Tracking

**Monthly metrics:**
- Number of patients screened
- Number enrolled (by team)
- Number in control vs intervention
- Crossover compliance (teams switching on schedule?)
- Protocol deviations

**Red flags:**
- Enrollment rate <5 patients/month → risk of underpowering
- Crossover delays → consult DSMB
- High loss to follow-up (>10%) → enhance retention efforts

---

## XII. BUDGET ESTIMATE

**Personnel (24 months):**
- PhD candidate (you): 50% FTE → $50K/year × 2 = $100K
- Research coordinator: 25% FTE → $30K/year × 2 = $60K
- Biostatistician: 15% FTE → $40K/year × 2 = $80K
- **Subtotal: $240K**

**Infrastructure:**
- Compute resources (Mayo servers): $20K/year × 2 = $40K
- AI model maintenance: $10K/year × 2 = $20K
- **Subtotal: $60K**

**DSMB & Adjudication:**
- DSMB fees: $5K per meeting × 8 meetings = $40K
- Adjudication panel: $10K (honoraria for external members)
- **Subtotal: $50K**

**Other:**
- Clinician training materials: $5K
- ClinicalTrials.gov registration: $0 (free)
- Publication costs (open access): $5K
- **Subtotal: $10K**

**Total estimated budget: $360K over 2 years**

**Funding sources:**
- Mayo institutional support (PI startup funds)
- NSF GRFP (if awarded, covers PhD stipend)
- NIH F31 (backup)

---

## XIII. DELIVERABLES

### Paper 2: Prospective Clinical Utility (Primary Publication)

**Target journals:**
1. *New England Journal of Medicine* (NEJM)
2. *Nature Medicine*
3. *NEJM Evidence* (backup)
4. *Lancet Digital Health* (backup)

**Authorship:**
- **First author:** [Your Name] (PhD candidate, led analysis)
- **Senior/corresponding:** Eric Klee (PI)
- **Co-authors:** Clinical team members, statistician, DSMB members (if substantial contribution)

**Timeline:** Submit Jan 2030 (includes 1 month for manuscript writing after database lock)

**Manuscript structure:**
- Abstract: Structured (Background, Methods, Results, Conclusions)
- Introduction: 3 paragraphs (problem, gap, objective)
- Methods: Trial design, population, intervention, endpoints, statistical analysis (detailed)
- Results: Primary endpoint first, then secondaries (with quality metrics), tables/figures
- Discussion: Key findings, interpretation, limitations, clinical implications, future directions

**Key figures:**
1. CONSORT flow diagram (enrollment, allocation, follow-up, analysis)
2. Primary endpoint: Diagnostic yield (intervention vs control), forest plot
3. Time-to-diagnosis: Kaplan-Meier curves
4. Diagnostic quality metrics: Multi-panel figure (actionability, phenotype resolution, certainty)
5. Subgroup analyses: Diagnostic yield by ancestry

**Supplementary materials:**
- Full statistical analysis plan (SAP)
- CONSORT-AI / DECIDE-AI checklist
- Detailed methods (AI system architecture)
- All secondary endpoint results
- Subgroup analyses

### Conference Presentations

**Abstract submissions:**
- **ASHG 2028** (American Society of Human Genetics) - submit preliminary results at 50% enrollment
- **ACMG 2029** (American College of Medical Genetics) - submit full results before paper submission

**Format:** Oral presentation (15 min) or poster

---

## XIV. QUESTIONS FOR ADVISOR (ERIC KLEE)

**Before finalizing Aim 2 protocol:**

1. **Baseline data for power calculation:**
   - What's current diagnostic yield in Mayo's undiagnosed rare disease program? (need exact %)
   - Median time-to-diagnosis? (distribution?)
   - How many patients enrolled per month currently?

2. **Feasibility:**
   - Are 6-8 clinical teams available for cluster randomization?
   - Is stepped-wedge design operationally acceptable?
   - Can we do 6-month silent-run pilot (Feb-Aug 2027)?
   - Is 7-8 patients/month realistic enrollment rate for 24 months?

3. **IRB & governance:**
   - Does Mayo require DSMB for this risk level?
   - Device risk determination process?
   - Any Mayo-specific policies for AI clinical trials?

4. **Clinical workflow:**
   - How long does standard WES/WGS interpretation take currently? (need baseline for time-to-diagnosis)
   - Will clinicians have time to review AI output? (workflow burden assessment)
   - Preferred format for AI report delivery? (portal, email, integrated into EHR?)

5. **Diagnostic quality:**
   - Does Mayo track actionability of current diagnoses? (need baseline data)
   - Feasibility of clinician utility survey? (extra burden?)
   - Who would chair the adjudication committee? (need senior Mayo geneticist)

6. **Authorship & roles:**
   - Confirm: You as PI, me as co-I / first author on Paper 2?
   - Other key co-authors to include early?
   - Who owns the AI model IP? (Mayo, or joint?)

---

**Status:** Draft for discussion  
**Next step:** Schedule meeting with Eric Klee to review this Aim 2 protocol and get feedback on feasibility

**Document Control:**
- Version: 1.0 Draft
- Date: October 31, 2025
- Author: [Your Name]
- Reviewer: Eric Klee (pending)
