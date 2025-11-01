# Aim 3: Longitudinal Outcomes & Health Economics
## Follow-Up Study of AI-Assisted Rare Disease Diagnosis

**Timeline:** Months 24-48 (Jul 2028 - Apr 2030)  
**Phase:** Long-term impact assessment  
**Output:** Paper 3 - Clinical outcomes & cost-effectiveness (target: *JAMA* / *Health Affairs*, 2030)

---

## I. OBJECTIVE

**Assess the long-term clinical impact, diagnostic stability, and cost-effectiveness of AI-assisted genomic diagnosis in patients from the Aim 2 prospective trial.**

---

## II. RATIONALE

### Why Long-Term Follow-Up Matters

**Gap in AI diagnostics literature:**
- Most AI studies report diagnostic accuracy or short-term yield
- Few measure whether diagnoses improve patient outcomes
- No data on cost-effectiveness or real-world implementation

**Clinical questions we can answer:**
1. **Do AI-assisted diagnoses lead to better health outcomes?** (reduced hospitalizations, improved QOL)
2. **Are AI diagnoses stable over time?** (reclassification rates)
3. **Is AI cost-effective?** (cost per diagnosis, cost per QALY gained)
4. **Do families benefit from cascade screening?** (pre-symptomatic detection in relatives)
5. **Are actionable diagnoses actually acted upon?** (treatment initiation, adherence)

### Competitive Advantage
- DeepRare, AlphaGenome, GenoMAS: Retrospective accuracy only
- Your thesis: Prospective trial (Aim 2) **+ long-term outcomes (Aim 3)**
- This depth of validation is extremely rare in AI diagnostics

---

## III. STUDY DESIGN

### Design Type
**Observational cohort study** (no new intervention)

**Population:** All patients from Aim 2 trial who received confirmed diagnosis (both control and intervention arms)

**Follow-up period:** 12-24 months post-diagnosis

**Data sources:**
1. Mayo electronic health records (EHR) - automated pulls
2. Patient surveys (mailed/online)
3. Clinician surveys (for treatment assessment)
4. Family pedigree updates
5. ClinVar / literature review (for variant reclassification)
6. Mayo billing data (for cost analysis)

**No additional clinical visits required** - passive follow-up via chart review + surveys

---

## IV. STUDY POPULATION

### Inclusion Criteria
1. **Enrolled in Aim 2 trial** (Sep 2027 - Aug 2029)
2. **Received confirmed diagnosis** within 90 or 180 days (per Aim 2 reference standard)
3. Willing to participate in follow-up (consent at Aim 2 enrollment includes Aim 3 follow-up)

### Exclusion Criteria
1. No confirmed diagnosis in Aim 2 (undiagnosed patients excluded from primary analysis)
   - **Note:** May include in secondary analysis to compare diagnosed vs undiagnosed outcomes
2. Withdrew consent for follow-up
3. Lost to follow-up (moved out of Mayo system, deceased before 12-month mark)

### Expected Sample Size
- **Aim 2 enrollment:** 180 patients
- **Expected diagnosed:** ~70 patients (assuming 40% yield with AI, 30% in control)
  - Control arm: ~30 patients diagnosed
  - Intervention arm: ~40 patients diagnosed
- **Target for Aim 3:** 65-70 diagnosed patients (allowing 5-10% loss to follow-up)

**Note:** Smaller sample than Aim 2, but adequate for:
- Clinical outcome comparisons (powered for medium effect sizes)
- Diagnostic stability assessment (descriptive)
- Health economics modeling

---

## V. PRIMARY RESEARCH QUESTIONS

### Question 1: Long-Term Clinical Outcomes
**Do AI-assisted diagnoses lead to better patient health outcomes?**

**Comparison groups:**
- **Primary:** Diagnosed patients in intervention arm vs diagnosed patients in control arm
  - Hypothesis: Earlier diagnosis (intervention) → better outcomes
- **Secondary:** Diagnosed vs undiagnosed patients (pooled across arms)
  - Hypothesis: Having a diagnosis improves outcomes regardless of how it was obtained

### Question 2: Diagnostic Stability
**Are AI-assisted diagnoses stable over time, or do they get reclassified?**

**Outcome:** Reclassification rate at 12 and 24 months post-diagnosis

### Question 3: Cost-Effectiveness
**Is AI-assisted diagnosis cost-effective from healthcare system perspective?**

**Metric:** Incremental cost-effectiveness ratio (ICER) - cost per quality-adjusted life year (QALY) gained

### Question 4: Family Impact
**Do probands' diagnoses lead to cascade screening and pre-symptomatic detection in relatives?**

**Outcome:** Number of at-risk relatives tested, number with pathogenic variants detected

### Question 5: Real-World Actionability
**For diagnoses classified as "actionable" in Aim 2, was action actually taken?**

**Outcome:** Treatment initiation rate, adherence at 12 months

---

## VI. ENDPOINTS & MEASUREMENTS

### Primary Endpoints

**1. Composite clinical outcome** (at 12 months post-diagnosis)

**Definition:** Binary outcome - did patient experience any of the following adverse events?
- All-cause hospitalization
- Emergency department visit
- Disease progression (pre-defined criteria per diagnosis type)
- Death

**Hypothesis:** Intervention arm (earlier diagnosis) has lower composite event rate

**Measurement:**
- Automated EHR pull (ICD codes for hospitalizations, ED visits)
- Chart review to confirm events related to diagnosed condition
- Blinded outcome assessment (reviewer does not know arm assignment)

**2. Health-related quality of life** (HRQoL)

**Instrument:** SF-36 (Short Form 36) - validated, widely used
- Physical component summary (PCS)
- Mental component summary (MCS)
- 8 domains: physical functioning, role-physical, bodily pain, general health, vitality, social functioning, role-emotional, mental health

**Timing:**
- Baseline (at diagnosis, retrospectively assessed from Aim 2 enrollment data)
- 12 months post-diagnosis
- 24 months post-diagnosis (if feasible)

**Hypothesis:** Intervention arm has greater improvement in HRQoL (Δ PCS, Δ MCS)

**Measurement:** Mailed survey or online REDCap form

---

### Secondary Endpoints: Clinical Outcomes

**3. All-cause hospitalizations**
- Number of hospitalizations per patient (count)
- Days hospitalized (total)
- Hospitalization rate (events per patient-year)

**4. Disease-specific clinical events**
- Pre-defined per diagnosis category:
  - **Neurological disorders:** Seizures, neurological crises, functional decline
  - **Metabolic disorders:** Metabolic decompensation, emergency treatment
  - **Cardiac disorders:** Arrhythmias, heart failure exacerbation, sudden cardiac events
  - **Immunological disorders:** Infections, autoimmune flares

**5. Functional status**
- Activities of Daily Living (ADL) score - basic self-care
- Instrumental Activities of Daily Living (IADL) score - complex tasks
- Employment status (employed, disabled, retired)

**6. Patient-reported outcomes**
- Diagnostic odyssey impact: "How has having a diagnosis affected your life?" (qualitative)
- Diagnostic satisfaction: "How satisfied are you with the diagnostic process?" (1-5 scale)

---

### Secondary Endpoints: Diagnostic Stability

**7. Reclassification rate** (at 12 and 24 months)

**Definition:** Change in ACMG classification of causal variant

**Categories:**
- **Upgrade:** LP → P (more evidence accumulated)
- **Downgrade:** P → LP or VUS, or LP → VUS (new evidence contradicts pathogenicity)
- **Stable:** No change in classification

**Data sources:**
- ClinVar (monthly checks for updates)
- PubMed alerts (new papers on specific genes)
- Functional studies (if performed post-diagnosis)

**Hypothesis:** AI diagnoses have similar or lower downgrade rate (more robust initial calls)

**8. New evidence publication rate**
- Number of new papers published on identified genes (proxy for emerging gene-disease associations)
- Track via PubMed alerts

**9. Functional validation completion**
- For diagnoses without initial functional validation, was validation performed in follow-up period?
- Did validation confirm or refute diagnosis?

---

### Secondary Endpoints: Health Economics

**10. Total healthcare costs** (at 12 months post-diagnosis)

**Components:**
- Diagnostic testing costs (from Aim 2 period)
- Downstream testing costs (confirmatory, cascade screening)
- Treatment costs (medications, procedures, therapies)
- Hospitalization costs
- Outpatient visit costs
- Emergency care costs

**Data source:** Mayo billing data (comprehensive cost capture)

**Comparison:**
- Mean total cost: Intervention vs control arm
- Cost per diagnosed patient
- Cost per undiagnosed patient (for sensitivity analysis)

**11. Quality-Adjusted Life Years (QALYs)**

**Calculation:**
- Convert SF-36 scores to utility weights (using validated mapping algorithm)
- Calculate QALYs accumulated over 12-24 month follow-up
- Formula: QALY = Utility × Time (years)

**Example:**
- Patient with SF-36 → Utility = 0.75 for 1 year → 0.75 QALYs
- Healthy baseline utility = 1.0

**12. Incremental Cost-Effectiveness Ratio (ICER)**

**Definition:** Cost per QALY gained by AI-assisted diagnosis

**Formula:**
```
ICER = (Cost_Intervention - Cost_Control) / (QALY_Intervention - QALY_Control)
```

**Interpretation:**
- ICER < $50,000/QALY → Highly cost-effective (US threshold)
- ICER $50,000-$150,000/QALY → Cost-effective
- ICER > $150,000/QALY → Not cost-effective

**Sensitivity analysis:**
- Vary time horizon (1 year vs 2 years vs lifetime projection)
- Vary discount rate (0%, 3%, 5%)
- Vary utility weights (confidence intervals)

**13. Diagnostic odyssey costs avoided**

**Concept:** Years of pre-diagnosis testing and medical costs saved by reaching diagnosis sooner

**Measurement:**
- For each patient, estimate counterfactual: "What would costs have been if diagnosis delayed by X years?"
- Use historical data from Mayo undiagnosed patients (time to diagnosis distribution)
- Calculate avoided costs = (years saved) × (annual testing cost for undiagnosed patient)

---

### Secondary Endpoints: Family Cascade Screening

**14. Family screening uptake rate**

**Definition:** Proportion of at-risk first-degree relatives (FDRs) who underwent genetic testing

**Measurement:**
- Chart review: Did proband's family members get tested?
- Genetic counseling notes
- Phone survey: "Did you discuss testing with your family? Did anyone get tested?"

**Calculation:**
```
Screening rate = (# FDRs tested) / (# at-risk FDRs)
```

**At-risk FDRs:**
- Autosomal dominant: Parents, siblings, children (all at risk)
- Autosomal recessive: Siblings only (parents are carriers)
- X-linked: Sex-dependent risk pattern

**15. Pre-symptomatic diagnoses in relatives**

**Definition:** Number of at-risk relatives found to carry pathogenic variant before symptom onset

**Clinical impact:** Enables preventive interventions (surveillance, prophylactic treatment)

**Example:**
- Proband diagnosed with Lynch syndrome (cancer predisposition)
- 3 siblings tested → 1 carries variant
- Sibling undergoes colonoscopy → early cancer detected and removed
- **Lives saved** = direct clinical impact

**16. Reproductive impact**

**Measurement:**
- Did diagnosis inform reproductive decisions? (prenatal testing, preimplantation genetic diagnosis)
- Number of pregnancies where testing was performed
- Outcomes (affected vs unaffected offspring)

---

### Secondary Endpoints: Real-World Actionability

**17. Treatment initiation rate** (for actionable diagnoses)

**Population:** Patients with diagnoses classified as "actionable" (ACMG Tier 1-2) in Aim 2

**Measurement:**
- Chart review: Was disease-specific treatment initiated within 6 months of diagnosis?
- Treatment types:
  - Medications (enzyme replacement, gene therapy, metabolic formula)
  - Procedures (ICD implantation, prophylactic surgery)
  - Lifestyle modifications (dietary changes, activity restrictions)
  - Surveillance protocols (regular imaging, lab monitoring)

**Hypothesis:** High actionability → high treatment initiation rate (>70%)

**Barriers to treatment initiation:**
- Insurance denial
- Treatment unavailable (no FDA-approved therapy)
- Patient preference (declined treatment)
- Physician uncertainty (despite diagnosis, treatment not recommended)

**18. Treatment adherence** (at 12 months)

**Population:** Patients who initiated treatment

**Measurement:**
- Prescription fill rates (pharmacy data)
- Clinician assessment: "Is patient adherent to treatment plan?" (yes/no/partial)
- Patient self-report: "Are you following the recommended treatment?" (yes/no)

**Target:** >80% adherence rate

**19. Barrier analysis** (for non-actionable diagnoses)

**Population:** Patients with diagnoses classified as "non-actionable" (ACMG Tier 3) or actionable but treatment not initiated

**Measurement:** Qualitative interviews or survey
- "Why was no treatment started?" (open-ended)
- Themes: No available treatment, cost, patient preference, physician recommendation, other

**Purpose:** Identify gaps between "actionable" classification and real-world action

---

## VII. DATA COLLECTION METHODS

### Automated EHR Pulls

**Data elements:**
- Demographics (age, sex, self-reported race/ethnicity)
- Diagnoses (ICD-10 codes)
- Procedures (CPT codes)
- Medications (prescription records)
- Hospitalizations (admission/discharge dates, primary diagnosis)
- ED visits (date, chief complaint, discharge diagnosis)
- Outpatient visits (date, specialty, billing codes)
- Lab results (disease-specific monitoring labs)
- Imaging reports (disease-specific surveillance imaging)
- Vital status (death registry linkage)

**Frequency:** Monthly automated pulls (or real-time EHR queries)

**Advantage:** Minimal patient burden, comprehensive capture

---

### Patient Surveys

**Survey 1: Baseline (administered at Aim 2 diagnosis)**
- SF-36 (HRQoL)
- Employment status
- ADL/IADL (functional status)
- Contact information for follow-up

**Survey 2: 12-month follow-up**
- SF-36 (HRQoL)
- Employment status
- ADL/IADL
- Treatment adherence (self-report)
- Diagnostic satisfaction
- Family screening questions (did relatives get tested?)
- Qualitative: "How has diagnosis impacted your life?"

**Survey 3: 24-month follow-up (optional, if timeline allows)**
- Repeat measures from Survey 2

**Delivery method:**
- Mailed paper survey with pre-paid return envelope
- Online REDCap link (option for tech-savvy patients)
- Phone interview (for patients who don't respond to mail/online)

**Response rate target:** >70% (incentivize with $25 gift card)

---

### Clinician Surveys

**Survey: Treatment assessment (for actionable diagnoses)**

**Administered to:** Treating clinician (geneticist, specialist) at 6 and 12 months post-diagnosis

**Questions:**
1. Was disease-specific treatment initiated? (yes/no)
2. If yes, what treatment(s)? (medication, procedure, surveillance, lifestyle)
3. Is patient adherent to treatment? (yes/no/partial)
4. If no treatment, why not? (no available treatment, cost, patient declined, not indicated, other)
5. Has diagnosis changed clinical management? (yes/no - describe)

**Format:** Brief 5-question REDCap form (5 minutes to complete)

---

### Chart Review

**Trained research coordinator reviews medical records for:**

**Clinical events:**
- Hospitalizations: Confirm ICD codes, determine if related to diagnosed condition
- ED visits: Confirm relation to diagnosed condition
- Disease progression: Use pre-defined criteria per diagnosis type
- Treatment initiation: Date, type, indication
- Family testing: Genetic counseling notes, pedigree updates

**Blinded review:** Coordinator does not know arm assignment (intervention vs control)

**Inter-rater reliability:** 10% of charts reviewed by second coordinator (κ > 0.8 target)

---

### Variant Reclassification Tracking

**Process:**
1. **ClinVar monitoring:** Monthly automated checks for each causal variant identified in Aim 2
   - API query: Has variant been reclassified?
   - Download new submissions and evidence
2. **Literature monitoring:** PubMed alerts for each causal gene
   - New publications on gene-disease association
   - Functional studies, case reports, population data
3. **Expert review:** Quarterly review by geneticist
   - Assess new evidence
   - Determine if reclassification warranted
   - Update ACMG classification if needed

**Documentation:** Spreadsheet with timeline of variant evidence

---

### Cost Data Collection

**Data source:** Mayo billing database

**Cost categories:**
- Diagnostic testing (WES/WGS, confirmatory tests, RNA-seq, etc.)
- Downstream testing (functional assays, cascade screening)
- Medications (prescription costs)
- Procedures (CPT codes with associated costs)
- Hospitalizations (DRG-based costs)
- Outpatient visits (E&M codes)
- Emergency care

**Perspective:** Healthcare system (payer perspective)
- Use Mayo's internal cost accounting
- Convert to standardized costs (Medicare reimbursement rates) for generalizability

**Time horizon:** 12 months post-diagnosis (primary), 24 months (secondary)

---

## VIII. STATISTICAL ANALYSIS PLAN

### Primary Analyses

**1. Composite clinical outcome (binary)**

**Comparison:** Diagnosed patients in intervention vs control arm

**Model:** Logistic regression
```
logit(P(Event = 1)) = β₀ + β₁·Intervention + β₂·Age + β₃·Sex + β₄·Disease_Category + β₅·Time_to_Diagnosis
```

**Covariates:**
- Intervention (AI-assisted vs standard)
- Age at diagnosis
- Sex
- Disease category (neurological, metabolic, cardiac, etc.)
- Time to diagnosis (days from enrollment to diagnosis)

**Output:** Adjusted odds ratio, 95% CI, p-value

**Hypothesis:** OR < 1 (intervention arm has lower event rate)

**2. Quality of life (continuous)**

**Comparison:** Change in SF-36 scores (12-month - baseline)

**Model:** Linear regression
```
ΔQoL = β₀ + β₁·Intervention + β₂·Baseline_QoL + β₃·Age + β₄·Sex + β₅·Disease_Category
```

**Output:** Mean difference in ΔQoL, 95% CI, p-value

**Hypothesis:** Intervention arm has greater improvement (Δ PCS, Δ MCS)

---

### Secondary Analyses

**3. Diagnostic stability (reclassification)**

**Analysis:** Descriptive statistics
- Reclassification rate: % of variants reclassified at 12 and 24 months
- Breakdown: Upgrade vs downgrade vs stable
- Comparison: Intervention vs control (χ² test)

**4. Cost-effectiveness analysis**

**Model:** Decision-analytic model (Markov or decision tree)

**States:**
- Diagnosed (with treatment)
- Diagnosed (no treatment available)
- Undiagnosed
- Death

**Inputs:**
- Transition probabilities (from Aim 2 + Aim 3 data)
- Costs (from Mayo billing data)
- Utilities (from SF-36 → utility weights)

**Output:**
- Mean cost per patient (intervention vs control)
- Mean QALYs per patient (intervention vs control)
- ICER = ΔCost / ΔQALY

**Sensitivity analyses:**
- One-way: Vary each parameter individually
- Probabilistic: Monte Carlo simulation (1000 iterations)
- Scenario: Best case, worst case, base case

**Time horizon:**
- Primary: 1 year (observed data)
- Secondary: Lifetime projection (using Markov model with literature-based transition probabilities)

**5. Family cascade screening**

**Analysis:** Descriptive
- Screening uptake rate: % of at-risk FDRs tested
- Pre-symptomatic diagnoses: Count
- Number needed to screen (NNS): # FDRs tested per pre-symptomatic diagnosis
- Cost per pre-symptomatic diagnosis

**6. Treatment initiation and adherence**

**Analysis:**
- Treatment initiation rate: % of actionable diagnoses with treatment started
- Adherence rate: % of patients on treatment who are adherent at 12 months
- Barriers to treatment: Thematic analysis of qualitative data

---

### Sample Size Considerations

**Aim 3 is exploratory** (not powered for hypothesis testing like Aim 2)

**Rationale:**
- Sample size determined by Aim 2 enrollment, not Aim 3-specific power calculation
- Focus on effect size estimation, not p-values
- Descriptive and hypothesis-generating

**Precision estimates:**
- With n=70 diagnosed patients (35 per arm), 80% power to detect:
  - Composite outcome: OR = 0.35 (large effect) at α=0.05
  - QoL change: Cohen's d = 0.68 (medium-large effect)
- Precision for cost-effectiveness: 95% CI for ICER will be wide but informative

**Interpretation:** Focus on clinical significance and effect sizes, not just statistical significance

---

## IX. TIMELINE & MILESTONES

### Data Collection Timeline

**Rolling enrollment into follow-up:**
- **Jul 2028:** First Aim 2 patients reach 12-month post-diagnosis
  - Start collecting long-term outcome data
- **Dec 2028:** ~30 patients with 12-month data
  - Preliminary analysis feasible (progress report for committee)
- **Dec 2029:** ~70 patients with 12-month data
  - Full analysis for Paper 3
- **Apr 2030:** All patients with 12+ month follow-up
  - Final analysis, manuscript writing

### Milestones

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| Aim 3 protocol finalized | Jun 2028 | IRB amendment for follow-up |
| First patient 12-month follow-up | Jul 2028 | Data collection starts |
| 50% patients with 12-month data | Dec 2028 | Preliminary analysis |
| Survey response rate check | Mar 2029 | Assess need for reminder calls |
| All patients with 12-month data | Dec 2029 | Full dataset complete |
| Cost-effectiveness model complete | Jan 2030 | Health economics results |
| Database lock | Feb 2030 | Analysis finalized |
| **Paper 3 draft complete** | **Mar 2030** | Submit to co-authors |
| **Paper 3 submission** | **Apr 2030** | JAMA / Health Affairs |
| PhD defense | Apr 2030 | Thesis complete |

---

## X. DELIVERABLES

### Paper 3: Long-Term Outcomes & Health Economics

**Target journals:**
1. *JAMA* (Journal of the American Medical Association)
2. *Health Affairs* (health economics focus)
3. *JAMA Network Open* (backup, open access)
4. *Medical Care* (health services research)

**Authorship:**
- **First author:** [Your Name] (PhD candidate)
- **Senior author:** Eric Klee (PI)
- **Co-authors:** Biostatistician (cost-effectiveness modeling), genetic counselors (family screening), clinicians (outcome assessment)

**Manuscript structure:**

**Abstract** (structured):
- Context: Why long-term follow-up matters
- Objective: Assess clinical outcomes and cost-effectiveness of AI-assisted diagnosis
- Design: Observational cohort follow-up of RCT (Aim 2)
- Setting: Mayo Clinic rare disease program
- Participants: 70 diagnosed patients from Aim 2
- Exposures: AI-assisted diagnosis (intervention) vs standard care (control)
- Main outcomes: Composite clinical outcome, QALYs, ICER
- Results: Key findings (numbers)
- Conclusions: Interpretation and implications

**Key figures:**
1. **Flow diagram:** Aim 2 enrollment → diagnosis → Aim 3 follow-up (attrition)
2. **Clinical outcomes:** Event-free survival curves (intervention vs control)
3. **Quality of life:** Change in SF-36 scores over time (box plots, spaghetti plots)
4. **Cost-effectiveness plane:** Scatterplot of ΔCost vs ΔQALY (with willingness-to-pay threshold line)
5. **Family cascade screening:** Flowchart showing proband → relatives tested → pre-symptomatic diagnoses

**Tables:**
1. Baseline characteristics (diagnosed patients)
2. Clinical outcomes at 12 months (hospitalization rate, composite events, QoL scores)
3. Diagnostic stability (reclassification rates)
4. Cost breakdown (diagnostic costs, treatment costs, total costs)
5. Incremental cost-effectiveness (base case + sensitivity analyses)
6. Treatment initiation and adherence (for actionable diagnoses)

**Supplementary materials:**
- Full cost-effectiveness model parameters
- Sensitivity analyses (one-way, probabilistic)
- Family pedigrees (de-identified examples)
- Patient survey instruments
- Qualitative themes (barriers to treatment)

---

### Conference Presentations

**ASHG 2029** (American Society of Human Genetics):
- Abstract: "Long-term clinical outcomes of AI-assisted rare disease diagnosis"
- Format: Oral presentation

**AcademyHealth Annual Research Meeting 2030** (health services research):
- Abstract: "Cost-effectiveness of AI in rare disease diagnosis"
- Format: Oral or poster

---

## XI. ETHICAL CONSIDERATIONS

### Informed Consent

**Aim 3 follow-up consent included in Aim 2 consent:**
- "If you receive a diagnosis during this study, we may contact you in 1-2 years to ask about your health and treatment"
- Separate checkbox: Opt-in to long-term follow-up (optional)
- Can withdraw from follow-up at any time without penalty

**Separate consent for qualitative interviews** (if conducted):
- Audio recording
- Quotations in publications (de-identified)

### Privacy & Confidentiality

**Survey data:**
- Stored in secure REDCap database
- Only research team has access
- De-identified for analysis

**Family pedigree data:**
- No identifiable information in publications
- Family members not contacted directly (only through proband)
- HIPAA compliance

### Minimal Risk

**Aim 3 is observational** → minimal additional risk beyond standard care
- No interventions
- No additional clinic visits
- Surveys are voluntary
- Chart review uses existing medical records

**IRB determination:** Amendment to Aim 2 protocol (not a separate IRB submission)

---

## XII. LIMITATIONS & MITIGATION

### Expected Limitations

**1. Small sample size**
- Only ~70 diagnosed patients (limited by Aim 2 enrollment)
- Wide confidence intervals for some estimates
- **Mitigation:** Focus on effect sizes and clinical significance, not just p-values

**2. Loss to follow-up**
- Patients move, change healthcare systems, or withdraw
- **Mitigation:** Maintain contact info, send reminders, offer gift card incentives
- **Target:** <10% loss to follow-up

**3. Survey non-response bias**
- Patients who respond to surveys may differ from non-responders
- **Mitigation:** Compare responders vs non-responders on EHR data (demographics, outcomes)
- **Report:** Describe potential bias in Discussion

**4. Short follow-up (12-24 months)**
- Some long-term outcomes (mortality, disease progression) take years to manifest
- **Mitigation:** Model lifetime costs/QALYs using literature-based projections
- **Acknowledge:** Uncertainty in long-term extrapolations

**5. Single-center study**
- Mayo Clinic population may not generalize to other settings
- **Mitigation:** Compare Mayo demographics to national rare disease registries
- **Acknowledge:** Generalizability limitation in Discussion

**6. Contamination from Aim 2**
- Clinicians in control arm may have learned from AI over time (stepped-wedge design)
- Could dilute effect estimates
- **Mitigation:** Sensitivity analysis: Compare early vs late control patients

---

## XIII. BUDGET ESTIMATE

**Personnel (24 months):**
- PhD candidate (you): 50% FTE → $50K/year × 2 = $100K
- Research coordinator (chart review, surveys): 25% FTE → $30K/year × 2 = $60K
- Biostatistician (cost-effectiveness modeling): 10% FTE → $40K/year × 2 = $80K
- **Subtotal: $240K**

**Data collection:**
- EHR data pulls: $10K (IT support)
- Survey printing/mailing: $5K
- Gift cards (incentives): $70 patients × $25 = $1,750
- Phone interviews (for non-responders): $5K
- **Subtotal: $22K**

**Analysis software:**
- TreeAge Pro (cost-effectiveness modeling): $2K
- REDCap (survey hosting): $0 (Mayo license)
- **Subtotal: $2K**

**Other:**
- ClinVar/PubMed monitoring (automated): $0
- Publication costs (open access): $5K
- **Subtotal: $5K**

**Total estimated budget: $269K over 2 years**

**Funding sources:**
- Mayo institutional support (PI continuation funds)
- NSF GRFP (if awarded, covers PhD stipend)

---

## XIV. QUESTIONS FOR ADVISOR (ERIC KLEE)

**Before finalizing Aim 3 protocol:**

1. **Feasibility:**
   - Can we access Mayo EHR for automated outcome pulls? (IT infrastructure)
   - Is 12-24 month follow-up sufficient, or do you recommend longer?
   - Will patients consent to long-term follow-up in Aim 2 consent?

2. **Data access:**
   - Can we get Mayo billing data for cost analysis? (permissions, compliance)
   - How do we track patients who move to external healthcare systems?
   - Death registry linkage available?

3. **Survey burden:**
   - Is 70% survey response rate realistic?
   - Should we offer phone interviews for all patients or only non-responders?
   - Any concerns about survey burden on patients?

4. **Family screening:**
   - How do we track cascade screening? (genetic counseling notes, EHR)
   - Can we contact family members directly, or only through proband?
   - Privacy considerations?

5. **Cost-effectiveness modeling:**
   - Do you have Mayo colleagues with health economics expertise? (potential co-author)
   - Preferred cost perspective (healthcare system, societal, payer)?
   - Should we model lifetime costs or stick to observed 1-2 year data?

6. **Timeline:**
   - Does Aim 3 timeline align with PhD defense (Apr 2030)?
   - Can we submit Paper 3 before defense, or should it be a defense goal only?

---

## XV. INTEGRATION WITH AIM 2

### Linked Narrative for Thesis

**Aim 2 (Paper 2):** "AI increases diagnostic yield and shortens time-to-diagnosis"
- Establishes efficacy (does AI work?)

**Aim 3 (Paper 3):** "AI-assisted diagnoses improve long-term outcomes and are cost-effective"
- Establishes effectiveness (does AI improve patient lives and is it worth the cost?)

**Thesis story arc:**
1. **Aim 1:** Build interpretable AI system (technical development)
2. **Aim 2:** Validate prospectively in clinical trial (efficacy)
3. **Aim 3:** Demonstrate real-world impact (effectiveness + economics)

**Competitive advantage:**
- Depth > breadth
- Few AI diagnostic studies have this level of longitudinal validation
- Provides complete picture: accuracy → clinical utility → health outcomes → cost-effectiveness

---

## XVI. ALTERNATIVE APPROACHES

### If Aim 3 Not Feasible

**Plan B: Simplified version**
- Focus only on diagnostic stability (reclassification) + family screening
- No patient surveys (use only EHR data)
- Simpler health economics (cost comparison, not full ICER)
- Still publishable in *Genetics in Medicine* or *JMIR*

**Plan C: Integrate into Aim 2 paper**
- 6-month follow-up only (shorten timeline)
- Include as secondary endpoints in Aim 2 paper (extended results section)
- Sacrifice separate publication, but reduce timeline risk

**Recommendation:** Pursue full Aim 3 unless feasibility concerns arise

---

**Status:** Draft for discussion  
**Next step:** Discuss Aim 3 feasibility with Eric Klee alongside Aim 2 protocol

**Document Control:**
- Version: 1.0 Draft
- Date: October 31, 2025
- Author: [Your Name]
- Reviewer: Eric Klee (pending)
