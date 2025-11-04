# Nucleotide Sequence Support for Variant Dataset

## Overview

The `variant_dataset.py` module has been upgraded to support nucleotide (CDS) sequence retrieval alongside protein sequences. This enables DNA-level analysis of variants using models like DNABERT2.

## What's New

### 1. New Function: `fetch_ensembl_cds()`
Fetches coding sequences (CDS) from Ensembl REST API.

```python
from variant_dataset import fetch_ensembl_cds

cds = fetch_ensembl_cds("ENST00000335295")  # HBB transcript
print(f"CDS length: {len(cds)} nucleotides")
```

### 2. New Function: `get_variant_nt_sequence()`
Retrieves wild-type or mutant nucleotide sequences with windowing support.

```python
from variant_dataset import PATHOGENIC_VARIANTS, get_variant_nt_sequence

variant = PATHOGENIC_VARIANTS[0]  # HBB E7V (sickle cell)

# Get wild-type CDS with 10-codon window (30 nt)
wt_cds = get_variant_nt_sequence(variant, window=10, version='wt')

# Get mutant CDS with same window
mut_cds = get_variant_nt_sequence(variant, window=10, version='mut')

print(f"WT:  {wt_cds}")
print(f"MUT: {mut_cds}")
```

### 3. Enhanced Variant Definitions
All variants now include `ensembl_transcript` field:

```python
{
    "gene": "HBB",
    "uniprot": "P68871",
    "ensembl_transcript": "ENST00000335295",  # NEW!
    "pos": 7,
    "wt": "E",
    "mut": "V",
    "disease": "Sickle cell disease",
    ...
}
```

### 4. Automatic Codon Mutation
The `get_variant_nt_sequence()` function automatically:
- Validates that the wild-type codon translates to the expected amino acid
- Finds a single-nucleotide change that produces the mutant amino acid
- Reports the codon change (e.g., GAG → GTG for E7V)

## Features

### Windowing
Specify window size in **codons** (not nucleotides):
- `window=5` → 33 nt total (5 upstream + variant + 5 downstream = 11 codons)
- `window=None` → full CDS

### Supported Mutations
- **Point mutations**: Automatically finds codon change (e.g., E→V)
- **Deletions**: Removes entire codon
- **Insertions**: Not yet implemented (will show warning)

### Validation
The genetic code is used to validate codon-to-amino-acid translation:

```python
from variant_dataset import validate_variant_dataset

validate_variant_dataset()
# ✓ Variant 1: HBB E7V (UniProt: P68871, Ensembl: ENST00000335295)
# ✓ Variant 2: CFTR F508del (UniProt: P13569, Ensembl: ENST00000003084)
# ...
```

## Example Usage

See `test_nt_sequences.py` for a complete demonstration:

```bash
python3 test_nt_sequences.py
```

This will show:
1. Protein sequences (wild-type vs mutant)
2. Nucleotide sequences (wild-type vs mutant)
3. Codon changes
4. Amino acid changes

## Use Cases

1. **DNA Language Model Analysis**: Feed CDS sequences to DNABERT2
2. **Codon Usage Studies**: Analyze synonymous vs non-synonymous changes
3. **Splice Site Analysis**: Extract sequences near exon boundaries
4. **Multi-modal Analysis**: Compare protein-level and DNA-level attention patterns

## API Reference

### `fetch_ensembl_cds(ensembl_transcript_id: str) -> str`
- **Args**: Ensembl transcript ID (e.g., "ENST00000335295")
- **Returns**: Full CDS as nucleotide string
- **Raises**: ValueError if transcript not found

### `get_variant_nt_sequence(variant_info: dict, window: Optional[int] = None, version: str = "wt") -> str`
- **Args**:
  - `variant_info`: Dict with keys: 'ensembl_transcript', 'pos', 'wt', 'mut'
  - `window`: Window size in codons (None = full sequence)
  - `version`: 'wt' for wild-type, 'mut' for mutant
- **Returns**: CDS sequence (full or windowed)
- **Raises**: ValueError if version is invalid

## Notes

- All Ensembl transcript IDs are canonical transcripts
- Codon positions are calculated from protein positions (1-indexed)
- The genetic code table is embedded in the function (standard code)
- API calls to Ensembl REST are made on-demand (no caching yet)
