"""
Variant dataset for systematic attention analysis.
Week 1: Define 10 variants (5 pathogenic, 5 benign)
"""

# Pathogenic variants from ClinVar
PATHOGENIC_VARIANTS = [
    {
        "gene": "HBB",
        "uniprot": "P68871",
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
        "pos": 504,  # Note: ALDH2*2 at E487K is pathogenic
        "wt": "G",
        "mut": "A",
        "trait": "Normal alcohol metabolism",
        "gnomad_freq": 0.60,  # ~60% in East Asians have wild-type
        "notes": "Wild-type allele; common and benign in all populations",
    },
]


def get_variant_sequence(variant_info: dict, window: int = 100) -> str:
    """
    Get protein sequence for variant with surrounding context.

    Args:
        variant_info: Dictionary with gene, uniprot, pos, wt, mut
        window: Residues on each side of variant

    Returns:
        sequence: Protein sequence string

    Note: For now, returns placeholder. Will integrate with UniProt in Week 2.
    """
    # TODO: Placeholder for now
    raise NotImplementedError("UniProt integration coming in Week 2")


def validate_variant_dataset() -> None:
    """
    Check that all variants have required fields.
    Run this before starting systematic analysis.
    """
    required_fields = ["gene", "uniprot", "pos", "wt", "mut"]

    print("=== Validating Pathogenic Variants ===")
    for i, var in enumerate(PATHOGENIC_VARIANTS, 1):
        missing = [f for f in required_fields if f not in var]
        if missing:
            print(f"❌ Variant {i}: Missing fields: {missing}")
        else:
            print(f"✓ Variant {i}: {var['gene']} {var['wt']}{var['pos']}{var['mut']}")

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
                    f"✓ Variant {i}: {var['gene']} {var['wt']}{var['pos']}{var['mut']}"
                )


if __name__ == "__main__":
    validate_variant_dataset()
