# PhD Research Plan & Timeline
## Genomic AI in Rare Disease Diagnosis: Prospective Clinical Validation

**Version:** 2.0 - Updated with Detailed Aim Protocols  
**Date:** October 31, 2025  
**Status:** Ready for advisor review

---

## I. RESEARCH PLAN

### A. Strategic Focus: Prospective Clinical Validation

The centerpiece of this thesis is establishing the **first prospectively validated AI agent system for rare disease diagnosis** with measured health outcomes. Current competitive landscape relies exclusively on retrospective validation. This prospective approach creates a significant differentiator from DeepRare, AlphaGenome, AI-MARRVEL, and other methods validated on historical datasets.

**Key Advantages:**
- Retrospective validation overestimates real-world performance (test data from same distribution as training)
- Prospective trials demonstrate clinical utility and health outcomes measurement
- Mayo's clinical infrastructure (Eric Klee's lab, RADIaNT program, 750K+ genomes) uniquely enables this
- Strategic partnerships with Microsoft (Healthcare Agent Orchestrator), NVIDIA (BioNeMo/AI Digital Cell), and Illumina (sequencing integration) provide infrastructure competitors lack

**Thesis Philosophy:** **Depth over breadth** - three linked aims using the same patient cohort for comprehensive validation from technical development through long-term clinical outcomes.

### B. Thesis Structure: Three Integrated Research Aims

#### Aim 1: Technical Development & Retrospective Validation (Months 1-18)

**Full protocol:** See `aim1_technical_development.md`

**Objective:** Build interpretable AI agent system for rare disease variant interpretation with prospective-ready architecture.

**System Architecture - Six Integrated Components:**

1. **Genomic Foundation Model** (Decision: Jan 2026)
   - Candidates: Enformer, Nucleotide Transformer, Hyena-DNA, DNABERT-2
   - Evaluation framework with weighted criteria (Performance 30%, Interpretability 25%, Usability 20%)
   - Selection based on variant pathogenicity prediction + attention extraction capability
   - Integration: Freeze weights, fine-tune adapter layers on Mayo data

2. **Protein Foundation Model** (Fixed: ESM2)
   - 650M parameter model, proven variant effect prediction
   - Isoform-specific analysis weighted by tissue expression (GTEx)
   - Attention visualization for residue-level interpretation

3. **Gene-Scale Interpretable Network** (GenNet Architecture)
   - Visible neural network: Input → Gene Layer (20K nodes) → Pathway Layer (1K nodes) → Disease Category → Output
   - Biological hierarchy enforced through network structure
   - Gene-level attention scores for interpretability

4. **Isoform-Specific Phenotype Prediction**
   - ESM1b + IMPPROVE for variant effect at isoform resolution
   - HPO term → isoform mapping with tissue-specific expression weighting
   - Phenotype resolution score: % of patient HPO terms explained by variant

5. **Network Medicine Integration**
   - Protein-protein interaction networks (STRING, BioGRID)
   - Disease module identification (DIAMOnD algorithm)
   - Network topology features predict phenotype breadth

6. **Multi-Agent Orchestration** (Microsoft HAO)
   - Literature search agent (PubMed API)
   - ACMG classification agent (automated guideline application)
   - Evidence aggregation agent (weighted confidence scoring)
   - Report generation agent (clinical report + interpretability visualizations)

**Multi-Ancestry Training Strategy:**
- Target: ≥20% non-European ancestry in training data
- Population stratification correction (ancestry PCs as covariates)
- Ancestry-specific calibration
- Fairness metrics: Sensitivity, specificity, PPV, calibration by ancestry group

**Interpretability Methods (5 approaches):**
1. Attention visualization (genomic + protein models)
2. SHAP values (feature importance quantification)
3. Counterfactual explanations (minimal changes to flip prediction)
4. Gene-level attribution (GenNet network paths)
5. Saliency maps (critical nucleotides for sequence models)

**Validation Strategy:**
- Mayo retrospective cohorts: 5,000+ cases (solved + unsolved)
- External validation: Baylor Genetics, Broad Institute
- Ablation studies: Quantify contribution of each component
- Comparison to baselines: Standard clinical workflow, Exomiser, LIRICAL, DeepRare (if reproducible)
- Target performance: ≥40% gene in top-5 yield (vs ~30% standard), AUC-ROC ≥0.85

**Timeline:** Sep 2026 - Jul 2027 (11 months development + 1 month paper writing)

**Output:** 
- **Paper 1:** "Interpretable Multi-Omics AI for Rare Disease Diagnosis: A Retrospective Validation Study"
- **Venue:** Nature Genetics (primary), Genome Medicine (backup)
- **Submission:** Jul 2027
- **Software:** Open-source GitHub repository with pre-trained models

---

#### Aim 2: Prospective Diagnostic-Impact RCT (Months 12-36)

**Full protocol:** See `aim2_clinical_trial_protocol.md`

**Objective:** Test whether AI-assisted genomic analysis increases diagnostic yield and diagnostic quality for undiagnosed rare disease patients in a prospective randomized controlled trial.

**Trial Design:** Stepped-Wedge Cluster Randomized Controlled Trial
- **Cluster:** Clinical care team (attending + genetic counselor + coordinator)
- **Number of clusters:** 6-8 teams at Mayo Rochester
- **Cluster size:** ~20-25 patients per team over 24 months
- **Rationale:** Operationally smooth, reduces contamination, all teams eventually receive AI

**Randomization Schedule:**
- All teams start in control (Months 1-3): Standard care, AI silent
- Every 3 months, 2 teams cross over to intervention
- Final state (Months 13+): All teams in intervention

**Arms:**
- **Control:** Standard workflow + AI runs silently in background (audit only)
- **Intervention:** Standard workflow + AI decision support (add-on mode, <48 hour turnaround)
- **UI:** Risk scores, top candidate genes, attention heatmaps, evidence summary
- **Clinician:** Retains final decision authority, override logging

**Population:**
- **Inclusion:** Age ≥18, Mayo Undiagnosed Diseases Program, WES/WGS ordered, no prior diagnosis
- **Exclusion:** Prior diagnosis, opted out, conditions where AI unreliable (mosaicism, structural variants, mitochondrial)
- **Sample size:** 180-200 patients over 24 months (~7-8 patients/month)
- **Power:** 80% to detect 15% absolute increase in diagnostic yield (30% → 45%)

**Primary Hypothesis (H1a):**
AI-assisted genomic analysis increases **diagnostic yield within 90 days by ≥15% (absolute difference)** compared with standard care.

**Primary Endpoint:**
- Diagnostic yield at 90 days (binary: confirmed diagnosis yes/no)
- Reference standard: Genetic confirmation (ACMG P/LP) OR expert consensus panel (blinded)

**Secondary Endpoints - 25 Total** (organized by category):

**Effectiveness (3):**
1. Time-to-diagnosis (days, censored at 90d)
2. Time-to-actionable step (ordering confirmatory test, initiating treatment)
3. Diagnostic yield at 180 days (extended follow-up)

**Accuracy (3):**
4. Sensitivity and specificity at locked AI threshold
5. Positive/negative predictive value
6. AUC-ROC

**Diagnostic Quality (6) - KEY DIFFERENTIATORS:**
7. **Clinical actionability rate** - % of diagnoses that are ACMG Tier 1-2 (treatment/management altering)
8. **Phenotype resolution score** - Mean % of HPO terms explained by identified variant(s)
9. **Diagnostic certainty distribution** - % Pathogenic (not just Likely Pathogenic)
10. **Clinical utility score** - Clinician-rated usefulness (1-5 Likert scale)
11. **Diagnostic concordance** - Inter-rater agreement (Fleiss' kappa), AI-expert concordance
12. **Follow-up reclassification rate** - % of diagnoses reclassified at 12 months

**Clinical Utilization (3):**
13. Number of downstream tests ordered
14. Cost of downstream testing per patient
15. Time to treatment initiation (for actionable diagnoses)

**Safety (3):**
16. False-positive triggered testing cascades
17. False-negative harms (missed diagnoses with consequences)
18. Adverse events attributed to AI

**Equity & Fairness (4):**
19. Diagnostic yield by ancestry group
20. Accuracy metrics by ancestry
21. Time-to-diagnosis by ancestry
22. Diagnostic yield by sex and age

**Process Measures (3):**
23. Clinician adherence rate (% following AI recommendation)
24. Time spent reviewing AI output
25. Clinician satisfaction (post-trial survey)

**Statistical Analysis:**
- Primary: Generalized linear mixed model (GLMM) with random effect for team
- Adjusts for temporal trends, accounts for clustering
- Gatekeeping hierarchy for multiplicity control
- Ancestry-stratified subgroup analyses (pre-specified)

**Governance:**
- **PI:** Eric Klee (Mayo) | **Co-I / Lead Analyst:** [Your Name] (1st author)
- IRB approval: Jan 2027 (submitted Oct 2026)
- DSMB: 3 members (external), quarterly meetings
- Model freeze: Jul 2027 (before trial launch)
- Silent-run pilot: Feb-Aug 2027 (measure baseline yield, estimate ICC)

**Timeline:** Sep 2027 - Nov 2029 (24 months enrollment + 3 months final follow-up)

**Output:**
- **Paper 2:** "AI-Assisted Diagnosis for Rare Genetic Disease: A Prospective Diagnostic-Impact RCT"
- **Venue:** NEJM (primary), Nature Medicine, NEJM Evidence (backups)
- **Submission:** Jan 2030
- **Key message:** AI increases diagnostic yield AND improves diagnostic quality

---

#### Aim 3: Longitudinal Outcomes & Health Economics (Months 24-48)

**Full protocol:** See `aim3_longitudinal_followup_study.md`

**Objective:** Assess long-term clinical impact, diagnostic stability, and cost-effectiveness of AI-assisted genomic diagnosis in patients from the Aim 2 prospective trial.

**Study Design:** Observational cohort follow-up study (no new intervention)
- **Population:** All patients from Aim 2 who received confirmed diagnosis (both arms)
- **Expected N:** 65-70 diagnosed patients (assuming 40% intervention yield, 30% control yield)
- **Follow-up:** 12-24 months post-diagnosis
- **Data sources:** Mayo EHR (automated), patient surveys, clinician surveys, ClinVar monitoring

**Five Research Questions:**

**1. Long-Term Clinical Outcomes**
- Primary comparison: Diagnosed patients (intervention arm) vs diagnosed patients (control arm)
- Hypothesis: Earlier diagnosis → better health outcomes
- Secondary: Diagnosed vs undiagnosed patients (pooled)

**2. Diagnostic Stability**
- Are AI diagnoses stable over time or reclassified?
- Track ClinVar updates, new literature monthly
- Report reclassification rate at 12 and 24 months

**3. Cost-Effectiveness**
- Incremental cost-effectiveness ratio (ICER): Cost per QALY gained
- Healthcare system perspective using Mayo billing data
- Decision-analytic model (Markov states: diagnosed with/without treatment, undiagnosed, death)

**4. Family Impact**
- Cascade screening uptake rate (% of at-risk relatives tested)
- Pre-symptomatic diagnoses in relatives
- Lives potentially saved through early detection

**5. Real-World Actionability**
- For "actionable" diagnoses, was treatment actually initiated?
- Adherence at 12 months
- Barriers to treatment (insurance, availability, patient preference)

**Primary Endpoints (2):**

1. **Composite clinical outcome at 12 months** (binary)
   - Any of: Hospitalization, ED visit, disease progression, death
   - Hypothesis: Intervention arm has lower event rate (earlier diagnosis → better outcomes)
   - Model: Logistic regression adjusting for age, sex, disease category, time-to-diagnosis

2. **Health-related quality of life** (continuous)
   - SF-36 instrument: Physical component summary (PCS), Mental component summary (MCS)
   - Measured at baseline (diagnosis), 12 months, 24 months
   - Hypothesis: Intervention arm has greater improvement in HRQoL

**Secondary Endpoints - 19 Total:**

**Clinical Outcomes (4):**
3. All-cause hospitalizations (count, days, rate per patient-year)
4. Disease-specific clinical events (pre-defined per diagnosis category)
5. Functional status (ADL/IADL scores, employment status)
6. Patient-reported outcomes (diagnostic odyssey impact, satisfaction)

**Diagnostic Stability (3):**
7. Reclassification rate at 12 and 24 months (upgrade vs downgrade vs stable)
8. New evidence publication rate (papers on identified genes)
9. Functional validation completion (was validation performed post-diagnosis?)

**Health Economics (4):**
10. Total healthcare costs at 12 months (diagnostic, treatment, hospitalization, outpatient, ED)
11. Quality-Adjusted Life Years (QALYs) - calculated from SF-36 utility weights
12. Incremental cost-effectiveness ratio (ICER) - cost per QALY gained
13. Diagnostic odyssey costs avoided (years of testing saved × annual cost)

**Family Cascade Screening (3):**
14. Family screening uptake rate (% of at-risk first-degree relatives tested)
15. Pre-symptomatic diagnoses in relatives (count, clinical impact examples)
16. Reproductive impact (prenatal testing, PGD decisions)

**Real-World Actionability (3):**
17. Treatment initiation rate for actionable diagnoses (% starting treatment within 6 months)
18. Treatment adherence at 12 months (prescription fill rates, clinician assessment)
19. Barrier analysis for non-actionable diagnoses (qualitative themes)

**Data Collection Methods:**
- **Automated EHR pulls:** Demographics, diagnoses, procedures, medications, hospitalizations, labs, imaging, vital status (monthly)
- **Patient surveys:** SF-36, ADL/IADL, treatment adherence, family screening, qualitative impact (mailed/online at 12 and 24 months)
- **Clinician surveys:** Treatment assessment, management changes (6 and 12 months post-diagnosis)
- **Chart review:** Blinded coordinator confirms events, treatment dates, family testing notes
- **Variant tracking:** Monthly ClinVar checks, PubMed alerts, quarterly geneticist review

**Cost-Effectiveness Analysis:**
- Decision-analytic model (Markov or decision tree)
- Time horizon: 1 year (observed data), lifetime projection (sensitivity analysis)
- Sensitivity analyses: One-way, probabilistic Monte Carlo, scenario (best/worst/base case)
- Target ICER: <$50K/QALY (highly cost-effective), <$150K/QALY (cost-effective)

**Timeline:** Jul 2028 - Apr 2030
- **Jul 2028:** First Aim 2 patients reach 12-month post-diagnosis (rolling enrollment into follow-up)
- **Dec 2028:** ~30 patients with 12-month data (preliminary analysis for committee)
- **Dec 2029:** ~70 patients with 12-month data (full analysis)
- **Feb 2030:** Database lock, final analysis
- **Apr 2030:** Paper 3 submission (just before defense)

**Output:**
- **Paper 3:** "Long-Term Clinical Outcomes and Cost-Effectiveness of AI-Assisted Rare Disease Diagnosis"
- **Venue:** JAMA (primary), Health Affairs, JAMA Network Open (backups)
- **Submission:** Apr 2030
- **Key message:** AI not only increases yield but improves patient outcomes and is cost-effective

**Why This Approach (vs Multi-Site Expansion):**
- ✅ More feasible (no new sites, recruitment, IRBs)
- ✅ Better timeline fit (overlaps with Aim 2, finishes before defense)
- ✅ Deeper story (efficacy → effectiveness → economics)
- ✅ Data competitors don't have (long-term outcomes, ICER)
- ✅ Addresses "so what?" question (does AI actually help patients and is it worth the cost?)

---

### C. Competitive Differentiation

| Component               | Current Competitors                                          | Your Advantage                                                               |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| **Validation Type**     | Retrospective benchmarks (DeepRare, AlphaGenome, AI-MARRVEL) | Prospective RCT with long-term follow-up                                     |
| **Clinical Deployment** | Web demos, limited clinical integration                      | Integrated with Mayo clinical workflow                                       |
| **Health Outcomes**     | Not measured or published                                    | Measured clinical outcomes, QALYs, cost-effectiveness                        |
| **Diagnostic Quality**  | Yield only (yes/no diagnosis)                                | Quality metrics: actionability, phenotype resolution, certainty, utility     |
| **Publication Venue**   | Methods journals, preprints                                  | Top-tier clinical journals (NEJM, JAMA, Nature Medicine)                    |
| **Infrastructure**      | Academic teams building from scratch                         | Microsoft + NVIDIA + Illumina partnerships                                   |
| **Dataset Access**      | Limited cohorts                                              | Mayo 750K+ genomes + rare disease focus                                      |
| **Depth**               | Retrospective accuracy only                                  | Technical → Clinical → Economic validation (3 linked papers, same cohort)    |
| **Interpretability**    | Black boxes                                                  | 5 interpretability methods (attention, SHAP, counterfactuals, attribution)   |
| **Equity**              | Not addressed                                                | Multi-ancestry training, fairness metrics, ancestry-stratified analyses      |

---

## II. TIMELINE

### Phase 1: Pre-PhD Application (NOW - December 2025)

**Nov 2025 (NOW)**
- Identify and contact potential PhD advisors (Eric Klee priority, alternative faculty)
- Prepare PhD application materials
- Request recommendation letters
- Define preliminary research interests based on prospective clinical validation strategy

**Dec 2025**
- Submit BICB PhD applications with focus on clinical validation differentiator
- Confirm PhD admission offers
- **Accepted outcome expected:** April 2026

---

### Phase 2: PhD Program Start (August/September 2026)

**Aug/Sep 2026**
- Begin BICB program
- Establish advisor relationship (Eric Klee or identified faculty)
- Initiate committee formation (minimum 3 faculty + advisor)
- First lab meeting and project planning
- Complete preliminary course requirements

**Fall 2026 Semester (Aug-Dec)**
- Complete BICB 8510 (Computation and Biology) - begin
- Take BICB 8930 (Journal Club) - 1 credit
- Take BICB 8920 (Colloquium) - 1 credit
- **Begin Aim 1 retrospective validation work**
- Literature review and methodology development
- Establish Mayo IRB relationships for prospective study design

---

### Phase 3: Year 1 - Foundation & Retrospective Development (Sep 2026 - Aug 2027)

**Sep-Oct 2026 (Months 1-2): Setup & Data Preparation**
- Set up Mayo computing accounts, GPU access
- IRB approval for retrospective data use (submit Sep, approve Oct)
- Pull Mayo retrospective cohorts (solved + unsolved cases, n=5,000+)
- Annotate VCFs, extract HPO terms
- Create train/val/test splits
- Ancestry PCA on training data

**Nov-Dec 2026 (Months 3-4): Foundation Model Selection**
- Implement genomic model evaluation framework
- Test 4 candidate models (Enformer, NT, Hyena-DNA, DNABERT-2)
- **Decision: Select genomic foundation model (Jan 2026)**
- Set up inference pipelines (ESM2 + genomic model)

**Jan-Feb 2027 (Months 5-6): Core Model Development**
- Implement GenNet architecture
- Train GenNet on ClinVar + Mayo training data
- Implement isoform-specific phenotype prediction
- Integrate network medicine features
- Hyperparameter tuning

**Mar 2027 (Month 7): Multi-Agent Orchestration**
- Set up Microsoft Healthcare Agent Orchestrator (HAO)
- Implement 7 agents (literature, ACMG, frequency, segregation, functional, aggregation, report)
- Test on 10 example patients

**Apr 2027 (Month 8): Interpretability Implementation**
- Implement attention visualization
- Implement SHAP value calculation
- Implement counterfactual generation
- Implement gene-level attribution (GenNet)
- Create visualization scripts

**May 2027 (Month 9): Validation & Analysis**
- Evaluate on held-out test set (Mayo)
- Evaluate on external datasets (Baylor, Broad)
- Ancestry-stratified performance analysis
- Ablation studies
- Failure analysis
- Statistical significance tests vs baselines

**Spring 2027**
- **Course Work:** Complete BICB 8401 (Ethics), BICB 8970 (Entrepreneurship)
- **Manuscript 1 Preparation:** Draft methodology paper
- **IRB Approval:** Submit Aim 2 prospective study protocol to Mayo IRB (April 2027)

**June 2027 (Month 10): Paper Writing**
- Draft Paper 1 (Introduction, Methods, Results, Discussion)
- Create 6 main figures
- Create supplementary materials
- Internal review by Eric Klee and lab

**June 2027: Preliminary Exams**
- **PRELIMINARY WRITTEN EXAM (Spring 2027):**
  - Submit ~12 page research proposal formatted as NIH proposal
  - Include: literature review, Aim 1-3 hypothesis-driven plan, methodology, preliminary retrospective data, significance
  - Iterate with advisor (required step)
  - Share with committee for informal feedback (~10 days)
  - Address comments
  - Formal submission to bicb@umn.edu and chadm@umn.edu
  - Anonymous review by 2 committee + 1 BICB faculty member (3 weeks)
  - Expected result: PASS → proceed to oral preliminary exam

- **PRELIMINARY ORAL EXAM:**
  - Scheduled after written exam pass
  - Pre-thesis seminar presentation (~30-40 min)
  - Examination by 3 graduate faculty (at most 2 from prelim committee + 1 DGS-selected)
  - Grade: Pass, Conditional Pass, or Fail
  - Expected: PASS → candidate status achieved

**July 2027 (Month 11): Paper Submission & Model Freeze**
- Address co-author feedback
- Polish figures and tables
- Finalize supplementary materials
- Prepare code repository (GitHub)
- **PAPER 1 SUBMISSION:** "Interpretable Multi-Omics AI for Rare Disease Diagnosis" (Jul 2027)
  - Venue: Nature Genetics (primary), Genome Medicine (backup)

**August 2027 (Month 12): Model Freeze for Aim 2**
- **Freeze model weights** (version control commit)
- Set decision thresholds (risk score cutoffs)
- Deploy model on Mayo servers for trial
- **Silent-run testing:** 1 month pilot (measure baseline yield, estimate ICC)
- **IRB Approval:** Aim 2 prospective trial protocol approved (Aug 2027)
- **DSMB:** Recruit and charter Data & Safety Monitoring Board

---

### Phase 4: Year 2 - Prospective Trial Initiation (Sep 2027 - Aug 2028)

**Sep 2027 (Month 1 of Trial): Launch**
- **Prospective Study Launch:** Begin patient enrollment
- **Aim 2 Work:** First 2 teams (of 8) enter control period, AI silent
- All teams in control period for first 3 months
- Clinician training completed for all teams
- Monthly enrollment tracking begins

**Oct-Nov 2027 (Months 2-3): Early Enrollment**
- Continue enrollment in control period (all 8 teams)
- Paper 1 manuscript under review (Nature Genetics or Genome Medicine)
- Committee meetings and annual progress review

**Dec 2027 (Month 4): First Crossover**
- **First 2 teams cross over to intervention** (AI visible)
- 6 teams remain in control
- Clinician training for crossover teams
- First patients in control period approaching 90-day endpoint

**Jan-Mar 2028 (Months 5-7): Continued Enrollment**
- Enrollment ongoing (rolling patient entry)
- Adjudication panel begins reviewing 90-day endpoints
- Monthly data quality checks
- Committee evaluation and annual review

**Mar 2028 (Month 7): Second Crossover**
- **2 more teams cross over to intervention** (now 4 intervention, 4 control)
- 50% of teams in intervention mode

**May 2028**
- **ANNUAL COMMITTEE MEETING:** Review Aim 1 completion, Aim 2 interim data, Aim 3 plan
- Consider NSF GRFP / NIH F31 fellowship applications if eligible
- Target: ~40 patients enrolled by this point

---

### Phase 5: Year 3 - Prospective Trial Completion & Aim 3 Initiation (Sep 2028 - Aug 2029)

**Jun 2028 (Month 10): Third Crossover**
- **2 more teams cross over to intervention** (now 6 intervention, 2 control)

**Jul 2028: Aim 3 Begins**
- **First Aim 2 patients reach 12-month post-diagnosis**
- Begin Aim 3 data collection (rolling enrollment into follow-up study)
- Automated EHR pulls for clinical outcomes
- Mail 12-month patient surveys (SF-36, treatment adherence, family screening)

**Sep 2028 (Month 13): Final Crossover**
- **Final 2 teams cross over to intervention** (all 8 teams now in intervention)
- All subsequent patients enrolled in intervention arm
- Continue enrollment until target reached (~180-200 patients)

**Oct-Dec 2028: Continued Enrollment**
- Enrollment ongoing toward 180-200 patient target
- Aim 3 data collection continues (now ~10-15 patients with 12-month follow-up)
- DSMB interim analysis at 50% enrollment (~90 patients, Dec 2028)

**Dec 2028**
- **Preliminary Aim 3 analysis** (~30 patients with 12-month data)
- Committee progress report on Aim 3 initial findings

**Jan-Aug 2029: Final Enrollment Phase**
- Continue enrollment to reach 180-200 patient target
- Patients from early control period completing 90-day and 180-day follow-ups
- Adjudication panel working through accumulated cases
- Aim 3 continues (more patients reaching 12-month mark)

**Aug 2029: Enrollment Complete**
- **Last patient enrolled** (Month 24 of trial)
- Target: 180-200 patients total enrolled
- 90-day follow-up continues for last patients through Nov 2029

**May 2029**
- **ANNUAL COMMITTEE MEETING:** Review prospective trial near-completion, Aim 3 progress

---

### Phase 6: Year 4 - Data Analysis, Thesis Writing & Defense (Sep 2029 - May 2030)

**Sep-Nov 2029: Final Follow-Up & Analysis**
- **Last patient completes 90-day follow-up** (Nov 2029)
- Adjudication panel completes all case reviews
- No new enrollment
- Aim 3 continues (now ~50-60 patients with 12-month data)

**Dec 2029: Aim 2 Database Lock**
- **Database lock:** All Aim 2 patients followed, all endpoints assessed
- Begin final statistical analysis (primary + 25 secondary endpoints)
- DSMB final review
- **Manuscript 2 Preparation:** Draft prospective trial results

**Dec 2029: Aim 3 Full Dataset**
- **~70 patients with 12-month follow-up data** (full Aim 3 dataset)
- Cost-effectiveness analysis complete
- Health economics modeling finalized

**Jan 2030: Paper Submissions**
- **PAPER 2 SUBMISSION:** "AI-Assisted Diagnosis for Rare Genetic Disease: A Prospective RCT"
  - Venue: NEJM (primary), Nature Medicine, NEJM Evidence (backups)
  - Primary result: Diagnostic yield increase + diagnostic quality improvements
  - Timeline: Submit Jan 2030

**Feb 2030: Aim 3 Analysis Complete**
- Aim 3 database lock
- Final statistical analysis (2 primary + 19 secondary endpoints)
- Cost-effectiveness model sensitivity analyses complete

**Mar 2030: Thesis Manuscript Preparation**
- **Manuscript 3 Preparation:** Draft Aim 3 longitudinal outcomes paper
- Integrate all three papers into dissertation document
- Committee engagement for feedback

**April 2030: Final Submission & Defense**
- **PAPER 3 SUBMISSION:** "Long-Term Clinical Outcomes and Cost-Effectiveness of AI-Assisted Rare Disease Diagnosis"
  - Venue: JAMA (primary), Health Affairs, JAMA Network Open (backups)
  - Submission: Apr 2030 (just before defense)

- **FINAL ORAL EXAMINATION:** Schedule final defense
  - Committee: 4+ members (at least 3 BICB faculty from ≥2 budgetary units, advisor as member but not chair, 1 minor if applicable)
  - Presentation: Research summary and defense of dissertation
  - Expected: PASS

**May 2030**
- Complete degree clearance steps with Graduate School
- **PhD COMPLETION:** Graduate with PhD in Bioinformatics and Computational Biology

---

## III. MILESTONE SUMMARY

| Milestone                                 | Target Date      | Status             | Aim    |
| ----------------------------------------- | ---------------- | ------------------ | ------ |
| PhD Application Submission                | Dec 2025         | Pending            | -      |
| PhD Admission                             | Apr 2026         | Pending            | -      |
| **Program Start**                         | **Aug/Sep 2026** | **~9 months away** | -      |
| Genomic Foundation Model Selected         | Jan 2027         | Year 1             | Aim 1  |
| Retrospective Validation Complete         | May 2027         | Year 1             | Aim 1  |
| **Paper 1 Submission**                    | **Jul 2027**     | **Year 1**         | Aim 1  |
| Prelim Written Exam                       | Spring 2027      | Year 1             | -      |
| Prelim Oral Exam                          | Jun 2027         | Year 1             | -      |
| Model Freeze                              | Jul 2027         | Year 1             | Aim 1  |
| IRB Approval (Prospective Study)          | Aug 2027         | Year 1             | Aim 2  |
| Silent-Run Pilot Complete                 | Aug 2027         | Year 1             | Aim 2  |
| **Prospective Enrollment Launch**         | **Sep 2027**     | **Year 2**         | Aim 2  |
| 50% Enrollment + Interim Analysis         | Dec 2028         | Year 3             | Aim 2  |
| **Aim 3 Data Collection Begins**          | **Jul 2028**     | **Year 3**         | Aim 3  |
| Full Enrollment Complete                  | Aug 2029         | Year 3             | Aim 2  |
| Last Patient 90-Day Follow-Up             | Nov 2029         | Year 4             | Aim 2  |
| Aim 2 Database Lock                       | Dec 2029         | Year 4             | Aim 2  |
| Aim 3 Full Dataset Complete               | Dec 2029         | Year 4             | Aim 3  |
| **Paper 2 Submission**                    | **Jan 2030**     | **Year 4**         | Aim 2  |
| Aim 3 Database Lock                       | Feb 2030         | Year 4             | Aim 3  |
| **Paper 3 Submission**                    | **Apr 2030**     | **Year 4**         | Aim 3  |
| Final Defense                             | Apr 2030         | Year 4             | -      |
| **PhD Graduation**                        | **May 2030**     | **Year 4**         | -      |

---

## IV. FUNDING STRATEGY

**Recommended Funding Applications:**

1. **NSF GRFP** (Apply Fall 2027 as 1st year student)
   - Higher success rate with preliminary data
   - $37K/year for 3 years
   - Application deadline: October 2027
   - **Narrative:** Prospective AI validation + health equity (multi-ancestry training)

2. **NIH F31** (Backup if NSF unsuccessful, apply Spring 2028)
   - Requires mentor/lab sponsorship
   - $25K/year + tuition coverage
   - **Narrative:** AI in rare disease clinical trial + long-term outcomes

3. **ARPA-H RAPID Program** (When eligible, Year 2-3)
   - Explicitly funds "Rare Disease AI/ML for Precision Integrated Diagnostics"
   - Focus on clinical deployment (aligns with prospective trial)
   - Announced December 2024
   - **Narrative:** Prospective RCT + cost-effectiveness = deployment readiness

4. **NIH U01** (If expanding to multi-site, Year 3-4)
   - "Demonstrating Clinical Utility of Genomic Diagnostic Technologies"
   - Multi-site collaboration potential
   - Supports prospective clinical trials
   - **Note:** Not primary plan (depth over breadth), but option for future work

5. **Microsoft/NVIDIA Research Partnerships**
   - Compute resource access (reduce costs)
   - Joint publication opportunities
   - In-kind support for HAO and BioNeMo infrastructure

**Budget Estimates:**
- **Aim 1:** ~$100K (compute, personnel 50% FTE PhD, biostatistician 15% FTE)
- **Aim 2:** ~$360K (personnel, DSMB, adjudication, infrastructure)
- **Aim 3:** ~$270K (personnel, surveys, data analysis, cost-effectiveness modeling)
- **Total:** ~$730K over 4 years

**Funding mix:**
- Mayo institutional support (PI startup funds): ~$300K
- NSF GRFP (if awarded): ~$111K
- Partnership in-kind support (Microsoft, NVIDIA): ~$100K equivalent
- Remaining gap: NIH F31 or Mayo supplement

---

## V. PUBLICATION STRATEGY & VENUES

| Paper       | Aim                         | Timeline         | Venue                                | Impact                                                    |
| ----------- | --------------------------- | ---------------- | ------------------------------------ | --------------------------------------------------------- |
| **Paper 1** | Retrospective Validation    | Year 1 (Jul '27) | Nature Genetics, Genome Medicine     | Establishes technical methodology + interpretability      |
| **Paper 2** | Prospective Clinical Trial  | Year 4 (Jan '30) | NEJM, Nature Medicine, NEJM Evidence | **Primary differentiator** - clinical utility + quality   |
| **Paper 3** | Long-Term Outcomes & Econ   | Year 4 (Apr '30) | JAMA, Health Affairs                 | Cost-effectiveness + real-world impact                    |
| Optional    | Methods/Architecture        | Year 2-3         | NeurIPS, ICML, RECOMB                | ML community impact (if time permits)                     |
| Optional    | Protocol Paper              | Year 2 (2028)    | Trials, BMJ Open                     | Transparency (publish Aim 2 protocol before results)      |

**Conference Presentations:**
- **ASHG 2027:** Aim 1 retrospective results (abstract)
- **ASHG 2028:** Aim 2 interim results (abstract at 50% enrollment)
- **ASHG 2029:** Aim 2 full results + Aim 3 preliminary (oral presentation)
- **AcademyHealth 2030:** Aim 3 health economics (health services research audience)

**Thesis Document Structure:**
- Chapter 1: Introduction + Literature Review
- Chapter 2: Aim 1 (Paper 1 content)
- Chapter 3: Aim 2 (Paper 2 content)
- Chapter 4: Aim 3 (Paper 3 content)
- Chapter 5: Discussion + Future Directions

---

## VI. COMPETITIVE URGENCY

**Field Saturation Timeline:**
- DeepRare (Jun 2025) - multi-agent rare disease system published
- AlphaGenome (Jun 2025) - regulatory variant foundation model
- GenoMAS (Jul 2025) - multi-agent genomic analysis
- Google DeepMind, GeneDx, Broad Institute all advancing rapidly

**Competitive Window:** 12-18 months before retrospective validation field becomes saturated

**Your Strategic Advantage:**
- **No competitor has prospective validation** (all retrospective as of Oct 2025)
- **No competitor has long-term outcomes** (all report accuracy/yield only)
- **No competitor has cost-effectiveness data** (all academic, not deployment-focused)
- **No competitor has diagnostic quality metrics** (all focus on yield, not quality)

**Strategy:** 
- Target Paper 1 submission Jul 2027 (establishes technical foundation)
- Target Paper 2 submission Jan 2030 (prospective RCT - major differentiator)
- If competitors publish prospective validation before 2030:
  - Emphasize unique aspects: diagnostic quality metrics, long-term outcomes (Aim 3), cost-effectiveness
  - Multi-ancestry focus and health equity
  - Depth of validation (3 linked papers on same cohort)

---

## VII. RISK MITIGATION & CONTINGENCY

**Technical Risks (Aim 1):**
- **Risk:** Genomic foundation model underperforms
  - **Mitigation:** Evaluation framework (Jan 2027) catches early; use ensemble or protein-only fallback
- **Risk:** Multi-ancestry data insufficient (<10% non-European)
  - **Mitigation:** Flag early (Month 1), request data augmentation, partner with other sites
- **Risk:** Model doesn't outperform baselines
  - **Mitigation:** Target ≥5% improvement (conservative); emphasize interpretability as differentiator

**Trial Risks (Aim 2):**
- **Risk:** Prospective trial enrollment delays
  - **Mitigation:** Have retrospective cohort analysis ready; interim results publication strategy (50-75 patients)
- **Risk:** IRB delays data access
  - **Mitigation:** Submit IRB protocol early (Oct 2026), 6-month buffer before trial launch
- **Risk:** Model freeze delayed
  - **Mitigation:** Strict deadline Aug 2027; freeze based on validation set if test set incomplete
- **Risk:** Loss to follow-up >10%
  - **Mitigation:** Maintain contact info, gift card incentives, Mayo EHR recapture

**Follow-Up Risks (Aim 3):**
- **Risk:** Survey non-response bias
  - **Mitigation:** Compare responders vs non-responders on EHR data; phone interviews for non-responders
- **Risk:** Small sample size (only ~70 diagnosed patients)
  - **Mitigation:** Focus on effect sizes, not p-values; acknowledge exploratory nature

**Publication Risks:**
- **Risk:** Paper 1 rejected from Nature Genetics
  - **Mitigation:** Have Genome Medicine backup ready; revise quickly based on reviews; 1-month buffer before Aim 2 trial
- **Risk:** Paper 2 rejected from NEJM
  - **Mitigation:** Nature Medicine, NEJM Evidence backups; still publishable in high-impact journal regardless
- **Risk:** Competitors publish prospective validation first
  - **Mitigation:** Emphasize unique aspects (quality metrics, outcomes, economics); depth over novelty

**Timeline Risks:**
- **Risk:** Delay in any aim cascades to defense
  - **Mitigation:** Built-in buffers at each phase; Aim 3 can start earlier if Aim 2 enrollment faster
- **Risk:** Paper acceptance delays defense
  - **Mitigation:** Can defend with "submitted" papers (not required to have acceptances)

---

## VIII. COMMITTEE & MENTORSHIP

**Recommended Committee:**
- **Advisor:** Eric Klee (Mayo) - rare disease diagnosis expertise, clinical infrastructure
- **Committee Member 2:** BICB faculty - genomics/genetics background (for Aim 1 technical aspects)
- **Committee Member 3:** BICB faculty - computational/ML methods expert (for interpretability, model validation)
- **Committee Member 4:** BICB faculty - biostatistics/clinical trials expertise (for Aim 2 trial design, Aim 3 analysis)
- **External Member (optional):** Clinical genetics collaborator (Broad, Baylor) for prospective validation perspective

**Committee Meeting Schedule:**
- **Formation:** Sep-Oct 2026 (first month of program)
- **Annual meetings:** Spring each year (after prelim exams)
- **Ad hoc:** As needed for major decisions (model selection, trial design)

**Mentorship Plan:**
- Weekly meetings with Eric Klee (advisor)
- Monthly lab meetings (Eric Klee's group)
- Quarterly check-ins with committee members
- Annual formal committee review

---

## IX. INTEGRATION & THESIS NARRATIVE

**Thesis Story Arc:**
```
Aim 1: BUILD IT
↓ (Technical foundation)
Can we build an interpretable AI system that improves rare disease diagnosis on retrospective data?
→ Paper 1: Yes, with 40%+ gene in top-5 yield vs 30% standard

Aim 2: TEST IT
↓ (Clinical validation)
Does it work in real-world prospective clinical use?
Does it improve diagnostic QUALITY, not just quantity?
→ Paper 2: Yes, increases yield by 15% absolute AND improves actionability, phenotype resolution, certainty

Aim 3: PROVE IT MATTERS
↓ (Real-world impact)
Do AI diagnoses improve patient health outcomes?
Is it cost-effective for healthcare systems?
→ Paper 3: Yes, better clinical outcomes + cost-effective (<$50K/QALY)
```

**Unified Cohort Approach:**
- Same patient population flows through all three aims
- Aim 1: Retrospective development on historical Mayo cases
- Aim 2: Prospective enrollment of new patients (Sep 2027 - Aug 2029)
- Aim 3: Follow-up of Aim 2 diagnosed patients (Jul 2028 - Apr 2030)
- **This creates a coherent narrative:** Development → Validation → Impact

**Competitive Positioning:**
- **Competitors:** Retrospective accuracy only
- **You:** Retrospective → Prospective → Long-term outcomes
- **Depth > Breadth:** Three linked papers on same system, same cohort
- **Translation focus:** Not just "does AI work?" but "does AI help patients?" and "is it worth it?"

---

## X. EXPECTED OUTCOMES & IMPACT

**Academic Impact:**
- 3 high-impact publications (Nature Genetics, NEJM, JAMA)
- Open-source software (GitHub with 100+ stars)
- Conference presentations (ASHG, AcademyHealth)
- Establish prospective validation as new standard for AI diagnostics

**Clinical Impact:**
- First prospectively validated AI for rare disease diagnosis
- Demonstrated diagnostic quality improvements (not just yield)
- Cost-effectiveness data for payers (reimbursement case)
- Long-term outcome data (patient benefit evidence)

**Translational Impact:**
- Mayo Clinic deployment (continued use post-trial)
- Potential FDA clearance pathway (prospective trial data)
- Licensing/commercialization opportunities (Mayo IP office)
- Potential expansion to other sites post-PhD (as postdoc or faculty)

**Personal Impact:**
- PhD completion with strong publication record
- Expertise in AI, genomics, clinical trials, health economics
- Network: Mayo, Microsoft, NVIDIA, Broad, Baylor collaborations
- Career positioning: Academic faculty, industry (biotech/pharma/health tech), or government (FDA, NIH, ARPA-H)

---

**Prepared:** October 31, 2025  
**Next Review:** With advisor (Eric Klee) upon PhD admission (Spring 2026)  
**Version Control:** This is v2.0 with detailed aim protocols integrated

**Supporting Documents:**
- `aim1_technical_development.md` - Full Aim 1 protocol (~50 pages)
- `aim2_clinical_trial_protocol.md` - Full Aim 2 protocol (~40 pages)
- `aim3_longitudinal_followup_study.md` - Full Aim 3 protocol (~35 pages)
- `research_proposal_grounding.md` - 2-page summary for advisor discussion
