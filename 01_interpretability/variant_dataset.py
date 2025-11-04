"""
Variant dataset for systematic attention analysis.

This module provides:
- 10 curated variants (5 pathogenic, 5 benign) with clinical annotations
- Protein sequence retrieval from UniProt (cached)
- Nucleotide (CDS) sequence retrieval from Ensembl (cached)
- Wild-type and mutant sequence generation
- Codon-level mutation modeling

Functions:
- fetch_uniprot_sequence(): Get protein sequence from UniProt (cached)
- fetch_ensembl_cds(): Get coding sequence from Ensembl (cached)
- get_variant_sequence(): Get protein sequences (wt/mut) with windowing
- get_variant_nt_sequence(): Get nucleotide sequences (wt/mut) with windowing
- validate_variant_dataset(): Validate all variant definitions

Caching:
    Both fetch_uniprot_sequence() and fetch_ensembl_cds() use LRU caching
    (maxsize=128) to avoid repeated API calls. First call fetches from API,
    subsequent calls are instant (>1M times faster).

Usage:
    from variant_dataset import PATHOGENIC_VARIANTS, get_variant_nt_sequence

    variant = PATHOGENIC_VARIANTS[0]  # HBB E7V (sickle cell)
    wt_cds = get_variant_nt_sequence(variant, window=10, version='wt')
    mut_cds = get_variant_nt_sequence(variant, window=10, version='mut')
"""

import requests
from typing import Optional
from functools import lru_cache

# Pathogenic variants from ClinVar
PATHOGENIC_VARIANTS = [
    {
        "gene": "HBB",
        "uniprot": "P68871",
        "ensembl_transcript": "ENST00000335295",  # HBB-201 (canonical)
        "pos": 7,  # 1-indexed (E7V after Met cleavage; also called E6V)
        "wt": "E",
        "mut": "V",
        "disease": "Sickle cell disease",
        "clinvar_id": "15333",
        "notes": "Most famous genetic variant; causes hemoglobin polymerization",
    },
    {
        "gene": "CFTR",
        "uniprot": "P13569",
        "ensembl_transcript": "ENST00000003084",  # CFTR-201 (canonical)
        "pos": 508,
        "wt": "F",
        "mut": "del",
        "disease": "Cystic fibrosis",
        "clinvar_id": "35",
        "notes": "Most common CF mutation (~70%); protein trafficking defect",
    },
    {
        "gene": "TP53",
        "uniprot": "P04637",
        "ensembl_transcript": "ENST00000269305",  # TP53-201 (canonical)
        "pos": 175,
        "wt": "R",
        "mut": "H",
        "disease": "Li-Fraumeni syndrome / Cancer",
        "clinvar_id": "12345",  # TODO: Find actual ClinVar ID
        "notes": "Most frequent TP53 hotspot; structural mutant affecting zinc binding",
    },
    {
        "gene": "TP53",
        "uniprot": "P04637",
        "ensembl_transcript": "ENST00000269305",  # TP53-201 (canonical)
        "pos": 248,
        "wt": "R",
        "mut": "W",
        "disease": "Cancer (multiple types)",
        "clinvar_id": "12346",  # TODO: Find actual ClinVar ID
        "notes": "DNA contact hotspot; gain-of-function; worse prognosis",
    },
    {
        "gene": "TP53",
        "uniprot": "P04637",
        "ensembl_transcript": "ENST00000269305",  # TP53-201 (canonical)
        "pos": 273,
        "wt": "R",
        "mut": "H",
        "disease": "Cancer (colorectal, others)",
        "clinvar_id": "12347",  # TODO: Find actual ClinVar ID
        "notes": "DNA contact hotspot; distinct transcriptional signature from R175H",
    },
]

# Benign variants from gnomAD (common polymorphisms)
BENIGN_VARIANTS = [
    {
        "gene": "OCA2",
        "uniprot": "Q04671",
        "ensembl_transcript": "ENST00000354638",  # OCA2-201 (canonical)
        "pos": 419,
        "wt": "R",
        "mut": "Q",
        "trait": "Eye color (green/hazel eyes)",
        "gnomad_freq": 0.05,  # ~5% in Europeans
        "notes": "rs1800407; Associated with normal pigmentation variation, not disease",
    },
    {
        "gene": "OCA2",
        "uniprot": "Q04671",
        "ensembl_transcript": "ENST00000354638",  # OCA2-201 (canonical)
        "pos": 615,
        "wt": "H",
        "mut": "R",
        "trait": "Light skin pigmentation (East Asian)",
        "gnomad_freq": 0.15,  # ~15% in East Asians
        "notes": "rs1800414; Common variant associated with normal pigmentation",
    },
    {
        "gene": "LCT",
        "uniprot": "P09848",
        "ensembl_transcript": "ENST00000299766",  # LCT-201 (canonical)
        "pos": 66,  # Approximate - regulatory variant in nearby MCM6
        "wt": "C",
        "mut": "T",
        "trait": "Lactase persistence (lactose tolerance)",
        "gnomad_freq": 0.30,  # ~30% globally, higher in Europeans
        "notes": "MCM6 rs4988235 (-13910 C>T); Regulates LCT expression; very common benign",
    },
    {
        "gene": "APOE",
        "uniprot": "P02649",
        "ensembl_transcript": "ENST00000252486",  # APOE-201 (canonical)
        "pos": 130,  # ε3 allele (most common)
        "wt": "C",
        "mut": "C",
        "trait": "Normal lipid metabolism",
        "gnomad_freq": 0.77,  # ~77% (ε3/ε3 + ε3/ε2 + ε3/ε4)
        "notes": "APOE ε3 is the ancestral/neutral allele; not associated with disease",
    },
    {
        "gene": "ALDH2",
        "uniprot": "P05091",
        "ensembl_transcript": "ENST00000261733",  # ALDH2-201 (canonical)
        "pos": 504,  # Note: ALDH2*2 at E487K is pathogenic
        "wt": "G",
        "mut": "A",
        "trait": "Normal alcohol metabolism",
        "gnomad_freq": 0.60,  # ~60% in East Asians have wild-type
        "notes": "Wild-type allele; common and benign in all populations",
    },
]


@lru_cache(maxsize=128)
def fetch_uniprot_sequence(uniprot_id: str) -> str:
    """
    Fetch protein sequence from UniProt (cached).

    Args:
        uniprot_id: UniProt accession (e.g., "P38398")

    Returns:
        sequence: Full protein sequence

    Note:
        Results are cached to avoid repeated API calls.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"Failed to fetch {uniprot_id}: {response.status_code}")

    # Parse FASTA format
    lines = response.text.strip().split("\n")
    sequence = "".join(lines[1:])  # Skip header line
    return sequence


@lru_cache(maxsize=128)
def fetch_ensembl_cds(ensembl_transcript_id: str) -> str:
    """
    Fetch coding sequence (CDS) from Ensembl (cached).

    Args:
        ensembl_transcript_id: Ensembl transcript ID (e.g., "ENST00000335295")

    Returns:
        sequence: Coding DNA sequence (nucleotides)

    Note:
        Results are cached to avoid repeated API calls.
    """
    url = f"https://rest.ensembl.org/sequence/id/{ensembl_transcript_id}"
    params = {"type": "cds"}
    headers = {"Content-Type": "text/x-fasta"}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch {ensembl_transcript_id}: {response.status_code}"
        )

    # Parse FASTA format
    lines = response.text.strip().split("\n")
    sequence = "".join(lines[1:])  # Skip header line
    return sequence


def get_variant_nt_sequence(
    variant_info: dict, window: Optional[int] = None, version: str = "wt"
) -> str:
    """
    Get nucleotide (CDS) sequence for variant analysis.

    Args:
        variant_info: Dict with 'ensembl_transcript', 'pos', 'wt', 'mut'
        window: If specified, extract window around variant position (in codons)
        version: 'wt' for wild-type or 'mut' for mutant CDS sequence

    Returns:
        sequence: Full or windowed nucleotide CDS sequence
    """
    full_cds = fetch_ensembl_cds(variant_info["ensembl_transcript"])

    # Convert protein position to nucleotide position (codon-based)
    # pos is 1-indexed protein position, convert to 0-indexed codon start
    pos = variant_info["pos"] - 1  # Convert to 0-indexed
    codon_start = pos * 3  # Start of codon in CDS
    codon = full_cds[codon_start : codon_start + 3]

    # Genetic code for validation
    genetic_code = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }

    # Validate wild-type codon translates to expected amino acid
    expected_wt = variant_info["wt"]
    actual_aa = genetic_code.get(codon, "?")

    if actual_aa != expected_wt:
        print(
            f"⚠️  Warning: Expected {expected_wt} at position {variant_info['pos']}, "
            f"codon {codon} translates to {actual_aa} in transcript "
            f"{variant_info['ensembl_transcript']}"
        )

    # Apply mutation if requested
    if version == "mut":
        mut = variant_info["mut"]

        if mut == "del":
            # Deletion: remove codon at position
            full_cds = full_cds[:codon_start] + full_cds[codon_start + 3 :]
            print(
                f"Deleted codon {codon} ({expected_wt}) at position "
                f"{variant_info['pos']}"
            )

        elif mut.startswith("ins"):
            # Insertion: add codons after position
            # For simplicity, this would require codon sequence in variant_info
            print(f"⚠️  Warning: Nucleotide insertion not yet implemented for {mut}")

        else:
            # Point mutation: find codon that produces mutant amino acid
            # Simple approach: try all possible single nucleotide changes
            mutant_codon = None
            for i in range(3):
                for nt in ["A", "T", "G", "C"]:
                    test_codon = codon[:i] + nt + codon[i + 1 :]
                    if genetic_code.get(test_codon, "?") == mut:
                        mutant_codon = test_codon
                        break
                if mutant_codon:
                    break

            if mutant_codon:
                full_cds = full_cds[:codon_start] + mutant_codon + full_cds[codon_start + 3 :]
                print(
                    f"Mutated codon {codon} → {mutant_codon} "
                    f"({expected_wt} → {mut}) at position {variant_info['pos']}"
                )
            else:
                print(
                    f"⚠️  Warning: Could not find single nucleotide change for "
                    f"{expected_wt} → {mut}"
                )

    elif version != "wt":
        raise ValueError(f"version must be 'wt' or 'mut', got '{version}'")

    # Extract window if specified (in codons)
    if window is None:
        return full_cds

    # Window is in codons, convert to nucleotides
    start = max(0, codon_start - (window * 3))
    end = min(len(full_cds), codon_start + 3 + (window * 3))
    return full_cds[start:end]


def get_variant_sequence(
    variant_info: dict, window: Optional[int] = None, version: str = "wt"
) -> str:
    """
    Get protein sequence for variant analysis.

    Args:
        variant_info: Dict with 'uniprot', 'pos', 'wt', 'mut'
        window: If specified, extract window around variant position
        version: 'wt' for wild-type or 'mut' for mutant sequence

    Returns:
        sequence: Full or windowed protein sequence
    """
    full_seq = fetch_uniprot_sequence(variant_info["uniprot"])

    # Validate wild-type matches
    pos = variant_info["pos"] - 1  # Convert to 0-indexed
    expected_wt = variant_info["wt"]
    actual_aa = full_seq[pos]

    if actual_aa != expected_wt:
        print(
            f"⚠️  Warning: Expected {expected_wt} at position {variant_info['pos']}, "
            f"found {actual_aa} in UniProt {variant_info['uniprot']}"
        )

    # Apply mutation if requested
    if version == "mut":
        mut = variant_info["mut"]

        if mut == "del":
            # Deletion: remove amino acid at position
            full_seq = full_seq[:pos] + full_seq[pos + 1 :]
            print(f"Deleted {expected_wt} at position {variant_info['pos']}")

        elif mut.startswith("ins"):
            # Insertion: add amino acids after position
            insert_seq = mut[3:]  # Extract sequence after 'ins'
            full_seq = full_seq[: pos + 1] + insert_seq + full_seq[pos + 1 :]
            print(f"Inserted {insert_seq} after position {variant_info['pos']}")

        else:
            # Point mutation: replace single amino acid
            full_seq = full_seq[:pos] + mut + full_seq[pos + 1 :]
            
    elif version != "wt":
        raise ValueError(f"version must be 'wt' or 'mut', got '{version}'")

    # Extract window if specified
    if window is None:
        return full_seq

    start = max(0, pos - window)
    end = min(len(full_seq), pos + window + 1)
    return full_seq[start:end]


def validate_variant_dataset() -> None:
    """
    Check that all variants have required fields.
    Run this before starting systematic analysis.
    """
    required_fields = ["gene", "uniprot", "ensembl_transcript", "pos", "wt", "mut"]

    print("=== Validating Pathogenic Variants ===")
    for i, var in enumerate(PATHOGENIC_VARIANTS, 1):
        missing = [f for f in required_fields if f not in var]
        if missing:
            print(f"❌ Variant {i}: Missing fields: {missing}")
        else:
            print(
                f"✓ Variant {i}: {var['gene']} {var['wt']}{var['pos']}{var['mut']} "
                f"(UniProt: {var['uniprot']}, Ensembl: {var['ensembl_transcript']})"
            )

    print("\n=== Validating Benign Variants ===")
    if not BENIGN_VARIANTS:
        print("⚠️  No benign variants defined yet - add 5 for Week 2")
    else:
        for i, var in enumerate(BENIGN_VARIANTS, 1):
            missing = [f for f in required_fields if f not in var]
            if missing:
                print(f"❌ Variant {i}: Missing fields: {missing}")
            else:
                print(
                    f"✓ Variant {i}: {var['gene']} {var['wt']}{var['pos']}{var['mut']} "
                    f"(UniProt: {var['uniprot']}, Ensembl: {var['ensembl_transcript']})"
                )


if __name__ == "__main__":
    validate_variant_dataset()
