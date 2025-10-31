# PhD Research Plan & Timeline
## Genomic AI in Rare Disease Diagnosis: Prospective Clinical Validation

---

## I. RESEARCH PLAN

### A. Strategic Focus: Prospective Clinical Validation

The centerpiece of this thesis is establishing the **first prospectively validated AI agent system for rare disease diagnosis** with measured health outcomes. Current competitive landscape relies exclusively on retrospective validation. this prospective approach creates a significant differentiator from DeepRare, AlphaGenome, AI-MARRVEL, and other methods validated on historical datasets.

**Key Advantages:**
- Retrospective validation overestimates real-world performance (test data from same distribution as training)
- Prospective trials demonstrate clinical utility and health outcomes measurement
- Mayo's clinical infrastructure (Eric Klee's lab, RADIaNT program, 750K+ genomes) uniquely enables this
- Strategic partnerships with Microsoft (Healthcare Agent Orchestrator), NVIDIA (BioNeMo/AI Digital Cell), and Illumina (sequencing integration) provide infrastructure competitors lack

### B. Thesis Structure: Three Integrated Research Aims

#### Aim 1: Technical Development & Retrospective Validation (Months 1-18)

**Objective:** Build interpretable AI agent system for rare disease variant interpretation with prospective-ready architecture.

**Technical Approach:**
- Combine gene-scale foundation models with built-in interpretability (GenNet architecture + gene-level transformers)
- Integrate isoform-specific phenotype prediction (ESM1b + IMPPROVE at isoform resolution)
- Use Microsoft Healthcare Agent Orchestrator for clinical workflow integration
- Deploy NVIDIA BioNeMo for model training/optimization on Mayo retrospective cohorts

**Specific Components:**
1. Gene-scale foundation models with visible neural networks for interpretable predictions
2. Counterfactual explanations (MrVI approach) for mechanistic understanding
3. Multi-ancestry training data focus (address existing training data gaps)
4. Integration of multi-omics: genomics (WES/WGS), transcriptomics (RNA-seq), phenotypic data

**Validation Dataset:**
- Mayo retrospective rare disease cohorts (solved cases from Eric Klee's Undiagnosed Diseases Program)
- Held-out test sets to demonstrate technical performance
- External validation on Baylor/Broad clinical lab cohorts

**Output:** Paper 1 - "Retrospective Validation of Gene-Scale Interpretable Models for Mendelian Disease Variant Interpretation"
- Venue: Nature Genetics, Genome Medicine
- Timeline: Submit by Month 18 (May 2027)

---

#### Aim 2: Prospective Clinical Trial Design & Enrollment (Months 12-36)

**Objective:** Deploy validated system in real-time with undiagnosed rare disease patients; measure diagnostic yield, time-to-diagnosis, and health outcomes.

**Study Design:**
- **Enrollment:** Consecutive patients presenting with suspected rare disease (undiagnosed)
- **Duration:** 2-year prospective enrollment (Months 18-36 of program)
- **Sample Size:** 100-200 patients (target diagnostic yield improvement measurement)
- **Intervention:** AI system applied in real-time during diagnostic workup alongside standard clinical care
- **Control Comparison:** Standard of care (clinician interpretation without AI assistance)

**Measured Outcomes:**
- Diagnostic yield (% of cases where diagnosis achieved)
- Time-to-diagnosis (days from enrollment to diagnosis)
- Diagnostic accuracy (confirmed by functional studies, clinical course, genetic counseling)
- Clinical management changes (documented treatment/monitoring changes due to diagnosis)
- Health outcomes (hospitalization reduction, medication changes, symptom improvement)
- Cost-effectiveness (cost per diagnosis, healthcare utilization)

**Clinical Infrastructure:**
- IRB protocol through Mayo's established clinical research pathways
- Integration with Eric Klee lab's diagnostic pipeline
- RADIaNT program for multi-omics phenotype data collection
- Real-time deployment in Mayo clinical genomics lab

**Output:** Paper 2 - "Prospective Clinical Trial of AI-Assisted Rare Disease Diagnosis: Impact on Diagnostic Yield and Time-to-Diagnosis"
- Venue: NEJM, Nature Medicine, NEJM Evidence, Lancet Digital Health
- Timeline: Submit by Month 36 (November 2027)
- **This is your major differentiator publication**

---

#### Aim 3: Health Economics & Multi-Site Scaling (Months 24-48)

**Objective:** Demonstrate reimbursement case and scalability for clinical adoption.

**Components:**
1. **Health Economics Analysis:**
   - Cost-effectiveness analysis (cost per diagnosis, ROI)
   - Comparison to current diagnostic pathways
   - Budget impact modeling for healthcare systems

2. **Multi-Site Expansion:**
   - Leverage Microsoft Healthcare Agent Orchestrator partnership (Mayo, Stanford, Johns Hopkins, Providence, MGH)
   - Expand deployment to additional hospital systems
   - Demonstrate system portability and generalization

3. **Regulatory Pathway:**
   - FDA regulatory strategy for AI diagnostic tool
   - Clinical validation requirements documentation

**Output:** Paper 3 - "Health Economics Analysis: Cost-Effectiveness and Scalability of AI-Guided Rare Disease Diagnosis"
- Venue: JAMA, Health Affairs, Value in Health
- Timeline: Submit by Month 48 (May 2028)

---

### C. Competitive Differentiation

| Component               | Current Competitors                                          | Your Advantage                                     |
| ----------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| **Validation Type**     | Retrospective benchmarks (DeepRare, AlphaGenome, AI-MARRVEL) | Prospective RCT with clinical sites                |
| **Clinical Deployment** | Web demos, limited clinical integration                      | Integrated with Mayo clinical workflow             |
| **Health Outcomes**     | Not measured or published                                    | Measured and published outcomes                    |
| **Publication Venue**   | Methods journals, preprints                                  | Top-tier clinical journals (NEJM, Nature Medicine) |
| **Infrastructure**      | Academic teams building from scratch                         | Microsoft + NVIDIA + Illumina partnerships         |
| **Dataset Access**      | Limited cohorts                                              | Mayo 750K+ genomes + rare disease focus            |
| **Regulatory Path**     | Unclear/early stage                                          | FDA prospective validation framework               |

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
- Begin Aim 1 retrospective validation work
- Literature review and methodology development
- Establish Mayo IRB relationships for prospective study design

---

### Phase 3: Year 1 - Foundation & Retrospective Development (Sep 2026 - Aug 2027)

**Fall 2026 - Spring 2027**
- **Aim 1 Work:** Build gene-scale foundation models, phenotype prediction integration
- **Course Work:** Complete core requirements (biochemistry/genetics, biostatistics, computer science/informatics)
- **Take:** BICB 8401 (Ethics)
- **Take:** BICB 8970 (Entrepreneurship and Leadership)
- Continue colloquium and journal club participation
- Begin preliminary data generation

**Spring 2027**
- **Aim 1 Progress:** Complete retrospective validation on Mayo cohorts
- **Manuscript 1 Preparation:** Draft methodology paper with preliminary retrospective results
- **Take:** BICB 8932 (Proposal Writing Seminar) - required for prelim preparation
- **IRB Approval:** Submit prospective study protocol to Mayo IRB

**May 2027**
- **PAPER 1 SUBMISSION:** "Retrospective Validation of Gene-Scale Interpretable Models..."
- **PRELIMINARY WRITTEN EXAM (Spring 2027):**
  - Submit ~12 page research proposal formatted as NIH proposal
  - Include: literature review, Aim 1-3 hypothesis-driven plan, methodology, preliminary retrospective data, significance
  - Iterate with advisor (required step)
  - Share with committee for informal feedback (~10 days)
  - Address comments
  - Formal submission to bicb@umn.edu and chadm@umn.edu
  - Anonymous review by 2 committee + 1 BICB faculty member (3 weeks)
  - Expected result: PASS â†’ proceed to oral preliminary exam

**June 2027**
- **PRELIMINARY ORAL EXAM:**
  - Scheduled after written exam pass
  - Pre-thesis seminar presentation (~30-40 min)
  - Examination by 3 graduate faculty (at most 2 from prelim committee + 1 DGS-selected)
  - Grade: Pass, Conditional Pass, or Fail
  - Expected: PASS â†’ candidate status achieved

---

### Phase 4: Year 2 - Prospective Trial Initiation (Sep 2027 - Aug 2028)

**Sep 2027 - Nov 2027**
- **Prospective Study Launch:** Begin patient enrollment (Month 18 of PhD program)
- **Aim 2 Work:** Enroll first patients, deploy AI system in clinical workflow
- **System Deployment:** Real-time integration in Mayo clinical genomics lab
- **Data Collection:** Diagnostic yield, time-to-diagnosis, outcomes tracking
- Manuscript 1 publication (Nature Genetics or Genome Medicine)
- Committee meetings and annual progress review

**Dec 2027 - May 2028**
- **Prospective Enrollment Ongoing:** Target 40-50 enrolled by May 2028
- **Aim 3 Initiation:** Begin health economics analysis framework
- **Manuscript 2 Preparation:** Draft prospective trial interim results
- Second year committee evaluation and annual review

**May 2028**
- **ANNUAL COMMITTEE MEETING:** Review Aim 1 completion, Aim 2 interim data, Aim 3 plan
- Continue course requirements (if any remaining)
- Consider NSF GRFP / F31 fellowship applications if eligible

**November 2027 / May 2028**
- **PAPER 2 SUBMISSION (Interim):** Prospective trial preliminary results (if sufficient enrollment)
  - Alternative: Submit Year 3 with full 2-year cohort

---

### Phase 5: Year 3 - Prospective Trial Completion & Scaling (Sep 2028 - Aug 2029)

**Sep 2028 - Dec 2028**
- **Prospective Enrollment Completion:** Target 100-200 patients enrolled
- **Aim 2 Analysis:** Complete outcome measurement, statistical analysis, clinical management impact assessment
- **Aim 3 Expansion:** Expand deployment to second/third Mayo site or partner institution
- **Multi-Site Trial Design:** Plan scaling to Broad/Stanford/Johns Hopkins if feasible

**Dec 2028 - May 2029**
- **PAPER 2 PUBLICATION:** "Prospective Clinical Trial of AI-Assisted Rare Disease Diagnosis..."
  - Expected submission: December 2028 or January 2029
  - Venue: NEJM, Nature Medicine, NEJM Evidence
  - **Timeline to publication:** 3-6 months post-submission
- **Aim 3 Work:** Health economics manuscript drafted
- Continue committee meetings and annual review

**May 2029**
- **ANNUAL COMMITTEE MEETING:** Review prospective trial completion, Aim 3 progress
- **Manuscript 3 Preparation:** Draft health economics and multi-site scaling results

---

### Phase 6: Year 4 - Completion & Thesis Defense (Sep 2029 - May 2030)

**Sep 2029 - Dec 2029**
- **Aim 3 Completion:** Finalize health economics analysis and multi-site deployment data
- **PAPER 3 SUBMISSION:** "Health Economics Analysis and Scalability of AI-Guided Rare Disease Diagnosis"
  - Venue: JAMA, Health Affairs, Value in Health
  - Timeline: Submit by December 2029

**Dec 2029 - Mar 2030**
- **Thesis Manuscript Preparation:** Integrate all three papers into dissertation document
- **Committee Engagement:** Final committee meetings for feedback
- **FINAL ORAL EXAMINATION:** Schedule final defense
  - Committee: 4+ members (at least 3 BICB faculty from â‰¥2 budgetary units, advisor as member but not chair, 1 minor if applicable)
  - Presentation: Research summary and defense of dissertation

**April 2030**
- **DEFENSE:** Final oral examination
- Expected: PASS
- Complete degree clearance steps with Graduate School

**May 2030**
- **PhD COMPLETION:** Graduate with PhD in Bioinformatics and Computational Biology

---

## III. MILESTONE SUMMARY

| Milestone                                 | Target Date      | Status             |
| ----------------------------------------- | ---------------- | ------------------ |
| PhD Application Submission                | Dec 2025         | Pending            |
| PhD Admission                             | Apr 2026         | Pending            |
| **Program Start**                         | **Aug/Sep 2026** | **~9 months away** |
| Retrospective Validation Complete (Aim 1) | May 2027         | Year 1             |
| **Paper 1 Submission**                    | **May 2027**     | **Year 1**         |
| Prelim Written Exam                       | Spring 2027      | Year 1             |
| Prelim Oral Exam                          | June 2027        | Year 1             |
| IRB Approval (Prospective Study)          | Aug 2027         | Year 1             |
| Prospective Enrollment Launch             | Sep 2027         | Year 2             |
| 50% Enrollment Target                     | Feb 2028         | Year 2             |
| **Paper 2 Submission**                    | **Jan-Nov 2028** | **Year 2-3**       |
| Full Enrollment Complete                  | Sep 2028         | Year 3             |
| Multi-Site Expansion (Aim 3)              | Dec 2028         | Year 3             |
| **Paper 3 Submission**                    | **Dec 2029**     | **Year 4**         |
| Final Defense                             | Apr 2030         | Year 4             |
| **PhD Graduation**                        | **May 2030**     | **Year 4**         |

---

## IV. FUNDING STRATEGY

**Recommended Funding Applications:**

1. **NSF GRFP** (Apply Fall 2026 as 1st year student)
   - Higher success rate with preliminary data
   - $37K/year for 3 years
   - Application deadline: October 2026

2. **NIH F31** (Backup if NSF unsuccessful, Year 2)
   - Requires mentor/lab sponsorship
   - $25K/year + tuition coverage

3. **ARPA-H RAPID Program** (When eligible)
   - Explicitly funds "Rare Disease AI/ML for Precision Integrated Diagnostics"
   - Focus on clinical deployment (aligns with prospective trial)
   - Announced December 2024

4. **NIH U01:** "Demonstrating Clinical Utility of Genomic Diagnostic Technologies"
   - Multi-site collaboration potential
   - Supports prospective clinical trials

5. **Microsoft/NVIDIA Research Partnerships**
   - Compute resource access (reduce costs)
   - Joint publication opportunities

---

## V. PUBLICATION STRATEGY & VENUES

| Paper          | Aim                        | Timeline                | Venue                                | Impact                                              |
| -------------- | -------------------------- | ----------------------- | ------------------------------------ | --------------------------------------------------- |
| **Paper 1**    | Retrospective Validation   | Year 1 (May 2027)       | Nature Genetics, Genome Medicine     | Establishes technical methodology                   |
| **Paper 2**    | Prospective Clinical Trial | Year 2-3 (Jan-Nov 2028) | NEJM, Nature Medicine, NEJM Evidence | **Primary differentiator** - clinical utility proof |
| **Paper 3**    | Health Economics & Scaling | Year 4 (Dec 2029)       | JAMA, Health Affairs                 | Reimbursement case for adoption                     |
| **Additional** | Methods/Architecture       | Year 2-3                | NeurIPS, ICML, RECOMB                | Community impact                                    |

---

## VI. COMPETITIVE URGENCY

**Field Saturation Timeline:**
- DeepRare (June 2025) - multi-agent rare disease system published
- AlphaGenome (June 2025) - regulatory variant foundation model
- GenoMAS (July 2025) - multi-agent genomic analysis
- Google DeepMind, GeneDx, Broad Institute all advancing rapidly

**Competitive Window:** 12-18 months before field becomes saturated

**Strategy:** Target major prospective validation publication by end of 2028 (Year 3) to maintain differentiation and establish clinical validation gold standard before competitors.

---

## VII. RISK MITIGATION & CONTINGENCY

**If Prospective Trial Enrollment Delays:**
- Have retrospective cohort comparative analysis ready for parallel submission
- Interim results publication strategy (publish with 50-75 patients if needed)
- Partner expansion contingency (Broad, Stanford sites if Mayo alone insufficient)

**If Competitors Publish Prospective Validation First:**
- Emphasize unique aspects: multi-ancestry focus, isoform-specific predictions, health outcomes measurement
- Focus on multi-site/health economics differentiation

**If Paper 2 Rejected from Top Venue:**
- Secondary venues: Lancet Digital Health, JAMA Network Open, eLife
- Still publishable and fundable regardless of venue

---

## VIII. COMMITTEE & MENTORSHIP

**Recommended Committee:**
- **Advisor:** Eric Klee (Mayo) - rare disease diagnosis expertise, clinical infrastructure
- **Co-Advisor (if applicable):** Mayo informatics faculty with AI/ML expertise
- **Committee Member 2:** BICB faculty - genomics/genetics background
- **Committee Member 3:** BICB faculty - computational/ML methods expert
- **External Member (optional):** Clinical genetics collaborator (Broad, Baylor) for prospective validation perspective

---

**Prepared:** November 2025
**Next Review:** Monthly with advisor, formally in spring 2027 (prelim exam)