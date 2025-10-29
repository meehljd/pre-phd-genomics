# Week 1 Analysis Plan: ESM2 Attention Patterns

## Expected Biological Signals for ESM2

### Protein Structure Features:
1. **Secondary structure boundaries**
   - Alpha helices
   - Beta sheets
   - Loop regions

2. **Functional domains**
   - Active sites
   - Binding domains
   - Post-translational modification sites

3. **Conservation patterns**
   - Highly conserved residues (catalytic, structural)
   - Variable regions (surface, linkers)

### Variant-Specific Expectations:

**Pathogenic variants (e.g., BRCA1 R1699W):**
- High attention to variant position?
- Disrupted attention patterns compared to wild-type?
- Attention focused on functional domains affected by variant?

**Benign variants:**
- Attention patterns similar to wild-type?
- Located in low-attention (less functionally important) regions?

## Hypotheses to Test:

### H1: Layer Progression
- **Early layers:** Local sequence patterns (amino acid neighbors)
- **Middle layers:** Secondary structure elements
- **Late layers:** Global protein structure, functional domains

### H2: Pathogenic vs Benign
- Pathogenic variants disrupt normal attention patterns more than benign
- Pathogenic variants located in high-attention regions

### H3: Functional Relevance
- Positions receiving highest attention correspond to:
  - Known functional residues (from UniProt annotations)
  - Conserved positions (from multiple sequence alignments)
  - Disease-associated sites (from ClinVar)

## Validation Strategy:

1. **Domain annotations:** Compare attention peaks to UniProt domain boundaries
2. **Conservation scores:** Correlate attention weights with PhyloP/PhastCons
3. **Known pathogenic sites:** Check if known disease mutations have high attention

## Next Steps After Week 1:
- Test hypotheses with 10-variant systematic analysis (Week 2)
- Compare ESM2 patterns to DNABERT2 (if time permits)
- Begin SHAP analysis for mechanistic explanations