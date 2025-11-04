"""
Example usage of nucleotide sequence retrieval for variants.

This demonstrates how to use the new NT sequence functionality
in variant_dataset.py for both pathogenic and benign variants.
"""

from variant_dataset import (
    PATHOGENIC_VARIANTS,
    BENIGN_VARIANTS,
    fetch_ensembl_cds,
    get_variant_nt_sequence,
    get_variant_sequence,
)


def demo_variant_sequences(variant_info: dict):
    """
    Demonstrate both protein and nucleotide sequence retrieval.
    """
    print(f"\n{'='*70}")
    print(f"Variant: {variant_info['gene']} {variant_info['wt']}{variant_info['pos']}{variant_info['mut']}")
    print(f"UniProt: {variant_info['uniprot']}")
    print(f"Ensembl: {variant_info['ensembl_transcript']}")
    print(f"{'='*70}\n")

    # 1. Get protein sequences (11 AA window)
    print("PROTEIN SEQUENCES (11 AA window):")
    print("-" * 70)
    wt_protein = get_variant_sequence(variant_info, window=5, version="wt")
    mut_protein = get_variant_sequence(variant_info, window=5, version="mut")
    print(f"WT:  {wt_protein}")
    print(f"MUT: {mut_protein}")

    # 2. Get nucleotide sequences (11 codon window = 33 nt)
    print("\nNUCLEOTIDE SEQUENCES (5 codon window = 33 nt):")
    print("-" * 70)
    wt_nt = get_variant_nt_sequence(variant_info, window=5, version="wt")
    mut_nt = get_variant_nt_sequence(variant_info, window=5, version="mut")
    print(f"WT:  {wt_nt}")
    print(f"MUT: {mut_nt}")

    # 3. Show the actual codon change
    pos = variant_info["pos"] - 1
    wt_codon_idx = 15  # Position in 33nt window (5*3 = 15)
    wt_codon = wt_nt[wt_codon_idx : wt_codon_idx + 3]
    mut_codon = mut_nt[wt_codon_idx : wt_codon_idx + 3]
    print(f"\nCodon change: {wt_codon} → {mut_codon}")
    print(f"Amino acid change: {variant_info['wt']} → {variant_info['mut']}")


def main():
    """
    Run examples for a few key variants.
    """
    print("\n" + "=" * 70)
    print("NUCLEOTIDE SEQUENCE FUNCTIONALITY DEMO")
    print("=" * 70)

    # Example 1: HBB E7V (Sickle cell)
    print("\n### EXAMPLE 1: Pathogenic variant (HBB E7V - Sickle cell)")
    demo_variant_sequences(PATHOGENIC_VARIANTS[0])

    # Example 2: TP53 R175H
    print("\n### EXAMPLE 2: Pathogenic variant (TP53 R175H)")
    demo_variant_sequences(PATHOGENIC_VARIANTS[2])

    # Example 3: Benign variant
    print("\n### EXAMPLE 3: Benign variant (OCA2 R419Q - eye color)")
    demo_variant_sequences(BENIGN_VARIANTS[0])

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print("✓ Added fetch_ensembl_cds() to retrieve CDS from Ensembl")
    print("✓ Added get_variant_nt_sequence() for NT sequence retrieval")
    print("✓ Added ensembl_transcript field to all variants")
    print("✓ Updated validation to check ensembl_transcript field")
    print("✓ Supports both wild-type and mutant NT sequences")
    print("✓ Automatic codon mutation based on amino acid change")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
