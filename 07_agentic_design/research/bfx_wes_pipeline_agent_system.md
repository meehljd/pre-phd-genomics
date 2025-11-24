# Multi-Source WES Pipeline Automation: Agent System Implementation

## Executive Summary

This report details an agent-based system for automating your existing Mayo Tapestry WES workflow and extending it to handle Mayo BioBank (Regeneron) and UK Biobank data. The system uses **specialized bioinformatics agents** that orchestrate bcftools, plink2, and Python tools while ensuring data quality, reproducibility, and proper handling of batch effects and population structure.

**Key innovation**: Source-aware processing with automated QC gates, statistical validation, and harmonization across vendors before ML ingestion.

---

## Architecture Overview

### Agent topology

**Single supervisor, 5 specialized workers** (simpler than general co-scientist, focused on your pipeline):

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Supervisor Agent                     │
│  - Routes by data source & processing stage                      │
│  - Tracks lineage (source → intermediate → H5)                   │
│  - Enforces QC gates (blocks downstream if QC fails)             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┬──────────────┬────────────┐
       ↓               ↓               ↓              ↓            ↓
┌─────────────┐ ┌─────────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐
│Source Intake│ │  QC/Validate│ │Harmonize │ │Batch Norm │ │H5 Convert│
│   Agent     │ │   Agent     │ │  Agent   │ │  Agent    │ │  Agent   │
└─────────────┘ └─────────────┘ └──────────┘ └───────────┘ └──────────┘
     │                │               │             │             │
     ↓                ↓               ↓             ↓             ↓
bcftools view    bcftools stats  bcftools norm  plink2 --pca   h5py
bcftools query   vcftools        plink2 --maf   Python PCA    PyTorch
bcftools filter  GATK QC         ref matching   batch correct  tensors
```

### Why this architecture?

**Pipeline-centric** rather than research-centric. Each agent = stage in your existing Tapestry workflow, plus new stages for multi-source handling. Agents are **deterministic** (same input → same output) with statistical validation at each gate.

---

## Source-Specific Characteristics

### Mayo Tapestry (Helix)

**Sequencing details**:
- Platform: Illumina NovaSeq (likely)
- Coverage: 20-30X exome
- Capture kit: Exome Research Panel v1 or v2 (Helix custom)
- Variant caller: GATK HaplotypeCaller (standard)
- Reference: GRCh38 (confirm - older data may be GRCh37)

**Known quirks**:
- Helix uses proprietary QC pipeline
- May have systematic coverage bias in certain genes (e.g., homologous regions)
- Population: Mayo patient population (likely Midwest US, European ancestry enriched)

**Your existing workflow** (baseline):
```bash
# Tapestry processing (your current pipeline)
bcftools view -S sample_list.txt input.vcf.gz | \
bcftools norm -m- --fasta-ref GRCh38.fa | \
bcftools filter -e 'QUAL<30 || DP<10' | \
bcftools annotate -x INFO,FORMAT/GT,FORMAT/DP,FORMAT/GQ | \
python convert_to_h5.py --output tapestry.h5
```

### Mayo BioBank (Regeneron)

**Sequencing details**:
- Platform: Illumina sequencing (part of Regeneron Genetics Center collaboration)
- Coverage: Likely 30-40X (higher than Tapestry)
- Capture kit: Regeneron standard exome panel (different from Helix)
- Variant caller: Regeneron custom pipeline (based on GATK but tuned)
- Reference: GRCh38 (consistent with recent RGC projects)

**Expected differences**:
- **Different exome boundaries**: RGC panel ≠ Helix panel (~5-10% non-overlapping regions)
- **Higher DP**: Will need DP thresholds adjusted
- **Allele frequency filters**: RGC filters at different AF thresholds
- **Annotation differences**: Different VEP/ANNOVAR versions

### UK Biobank

**Sequencing details**:
- Platform: Multiple vendors over time (Illumina HiSeq, NovaSeq)
- Coverage: 20X target (lower than your Mayo sources)
- Capture kit: Mixed (early: Illumina TruSeq, later: IDT xGen)
- Variant caller: Multi-stage pipeline (joint calling across 500K samples)
- Reference: GRCh38 (consistent)

**Challenges**:
- **Massive batch effects**: 500K samples sequenced over years, multiple sequencing centers
- **Already has joint calling artifacts**: Need to be aware of reference panel biases
- **Population structure**: Far more diverse than Mayo (global recruitment)
- **Data access**: Requires RAP (Research Analysis Platform) or download - different data format

---

## Agent 1: Source Intake Agent

**Purpose**: Standardize input, extract metadata, validate file integrity

### Implementation

```python
from google.adk import tools, agents
from typing import Literal
import subprocess
import json

@tools.tool
def identify_source(vcf_path: str) -> dict:
    """
    Identify VCF source from metadata
    Returns: {source: str, n_samples: int, n_variants: int, build: str}
    """
    # Extract header
    header = subprocess.run(
        ["bcftools", "view", "-h", vcf_path],
        capture_output=True, text=True
    ).stdout
    
    # Parse metadata
    source = "unknown"
    if "Helix" in header or "tapestry" in vcf_path.lower():
        source = "mayo_tapestry_helix"
    elif "regeneron" in header.lower() or "biobank" in vcf_path.lower():
        source = "mayo_biobank_regeneron"
    elif "ukbiobank" in vcf_path.lower() or "ukb" in vcf_path.lower():
        source = "ukbiobank"
    
    # Get counts
    stats = subprocess.run(
        ["bcftools", "stats", vcf_path],
        capture_output=True, text=True
    ).stdout
    
    n_samples = int([l for l in stats.split('\n') if l.startswith('SN')][0].split('\t')[3])
    n_variants = int([l for l in stats.split('\n') if 'number of records' in l][0].split('\t')[3])
    
    # Detect reference build
    if "GRCh38" in header or "hg38" in header:
        build = "GRCh38"
    elif "GRCh37" in header or "hg19" in header:
        build = "GRCh37"
    else:
        build = "unknown"
    
    return {
        "source": source,
        "n_samples": n_samples,
        "n_variants": n_variants,
        "build": build,
        "vcf_path": vcf_path
    }

@tools.tool
def validate_vcf_integrity(vcf_path: str) -> dict:
    """
    Check VCF file integrity (format compliance, index, corruption)
    """
    checks = {}
    
    # 1. Check if indexed
    import os
    checks["has_index"] = os.path.exists(f"{vcf_path}.tbi") or os.path.exists(f"{vcf_path}.csi")
    
    # 2. Check format compliance
    try:
        result = subprocess.run(
            ["bcftools", "view", "-h", vcf_path],
            capture_output=True, text=True, check=True, timeout=60
        )
        checks["format_valid"] = True
    except:
        checks["format_valid"] = False
        checks["error"] = "VCF format invalid or file corrupted"
        return checks
    
    # 3. Check required fields
    header = result.stdout
    required_fields = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    checks["has_required_fields"] = all(f in header for f in required_fields)
    
    # 4. Sample corruption check (random sample 1000 variants)
    try:
        test = subprocess.run(
            ["bcftools", "view", "-H", vcf_path, "|", "head", "-1000"],
            shell=True, capture_output=True, text=True, check=True, timeout=60
        )
        checks["sample_valid"] = len(test.stdout.strip().split('\n')) > 0
    except:
        checks["sample_valid"] = False
    
    checks["passed"] = all([
        checks["has_index"],
        checks["format_valid"],
        checks["has_required_fields"],
        checks["sample_valid"]
    ])
    
    return checks

@tools.tool
def extract_source_metadata(vcf_path: str, source: str) -> dict:
    """
    Extract source-specific metadata for downstream normalization
    """
    # Get sample-level statistics
    stats_out = subprocess.run(
        ["bcftools", "stats", "-s", "-", vcf_path],
        capture_output=True, text=True
    ).stdout
    
    # Parse per-sample depth, quality
    sample_stats = {}
    for line in stats_out.split('\n'):
        if line.startswith('PSC'):  # Per-sample counts
            fields = line.split('\t')
            sample_id = fields[2]
            sample_stats[sample_id] = {
                "n_variants": int(fields[3]),
                "n_snps": int(fields[4]),
                "n_indels": int(fields[5]),
                "n_singletons": int(fields[6])
            }
    
    # Source-specific metadata
    if source == "mayo_tapestry_helix":
        capture_kit = "helix_exome_v2"
        expected_depth_range = (20, 30)
        expected_ti_tv = (2.8, 3.2)  # Typical exome Ti/Tv
    elif source == "mayo_biobank_regeneron":
        capture_kit = "regeneron_exome"
        expected_depth_range = (30, 40)
        expected_ti_tv = (2.8, 3.2)
    elif source == "ukbiobank":
        capture_kit = "mixed_illumina_idt"
        expected_depth_range = (15, 25)  # Lower, more variable
        expected_ti_tv = (2.7, 3.3)  # More variable due to batches
    else:
        capture_kit = "unknown"
        expected_depth_range = (10, 50)
        expected_ti_tv = (2.5, 3.5)
    
    return {
        "source": source,
        "capture_kit": capture_kit,
        "expected_depth_range": expected_depth_range,
        "expected_ti_tv": expected_ti_tv,
        "sample_stats": sample_stats,
        "n_samples": len(sample_stats)
    }

# Agent definition
intake_agent = agents.Agent(
    model="gemini-2.5-flash",  # Fast for orchestration
    tools=[identify_source, validate_vcf_integrity, extract_source_metadata],
    system_prompt="""You are a VCF intake specialist. Your job:
    1. Identify the data source (Mayo Tapestry/Helix, Mayo BioBank/Regeneron, UK Biobank)
    2. Validate file integrity and format compliance
    3. Extract source-specific metadata for downstream processing
    4. Flag any anomalies or quality concerns
    
    Always run all three tools in sequence. If validation fails, STOP and report errors.
    """
)
```

### Validation gates

**Automatic rejection criteria**:
- VCF format invalid
- Missing index
- No samples in VCF
- Reference build mismatch with expected (e.g., GRCh37 when expecting GRCh38)

**Human review triggers**:
- Unknown source detected
- Sample count far from expected (e.g., expect 1000, got 100)
- Unusual Ti/Tv ratio (outside expected range for source)

---

## Agent 2: QC/Validation Agent

**Purpose**: Comprehensive quality control before processing. Implements best practices from GATK, TOPMed, gnomAD.

### QC metrics hierarchy

**Sample-level** (fail samples, not entire batch):
- **Depth**: Mean DP per sample (source-specific thresholds)
- **Missingness**: Proportion of missing genotypes
- **Heterozygosity**: Expected ~0.001 for exomes (detect contamination)
- **Ti/Tv ratio**: Should be 2.8-3.2 for exomes
- **Sex check**: Heterozygosity on chrX vs expected sex

**Variant-level** (filter variants):
- **Quality score**: QUAL > threshold (source-dependent)
- **Depth**: DP > min threshold
- **Allele balance**: 0.25-0.75 for hets (detect strand bias)
- **Hardy-Weinberg**: HWE p > 1e-6 in controls
- **Missing rate**: <10% across samples

**Cohort-level** (batch QC):
- **Principal components**: Detect population stratification
- **Relatedness**: IBD > 0.25 (1st/2nd degree relatives)
- **Batch clustering**: PCA should NOT separate by sequencing batch

### Implementation

```python
@tools.tool
def compute_sample_qc(vcf_path: str, source: str) -> dict:
    """
    Compute per-sample QC metrics using bcftools + plink2
    """
    import tempfile
    import pandas as pd
    
    temp_prefix = tempfile.mktemp(prefix="sample_qc_")
    
    # Use plink2 for comprehensive QC
    subprocess.run([
        "plink2",
        "--vcf", vcf_path,
        "--make-bed",
        "--out", temp_prefix,
        "--max-alleles", "2",  # Biallelic only for QC
    ], check=True)
    
    # Sample-level missingness
    subprocess.run([
        "plink2",
        "--bfile", temp_prefix,
        "--missing", "sample",
        "--out", temp_prefix
    ], check=True)
    
    smiss = pd.read_csv(f"{temp_prefix}.smiss", delim_whitespace=True)
    
    # Heterozygosity
    subprocess.run([
        "plink2",
        "--bfile", temp_prefix,
        "--het",
        "--out", temp_prefix
    ], check=True)
    
    het = pd.read_csv(f"{temp_prefix}.het", delim_whitespace=True)
    het['F'] = (het['OBS_CT'] - het['HOM_CT']) / het['OBS_CT']  # Inbreeding coef
    
    # Merge QC metrics
    qc = smiss.merge(het[['IID', 'F']], on='IID')
    
    # Source-specific thresholds
    thresholds = {
        "mayo_tapestry_helix": {"miss_thresh": 0.05, "f_min": -0.15, "f_max": 0.15},
        "mayo_biobank_regeneron": {"miss_thresh": 0.03, "f_min": -0.15, "f_max": 0.15},
        "ukbiobank": {"miss_thresh": 0.10, "f_min": -0.20, "f_max": 0.20}  # More permissive
    }
    
    thresh = thresholds.get(source, {"miss_thresh": 0.05, "f_min": -0.15, "f_max": 0.15})
    
    # Flag failing samples
    qc['fail_missing'] = qc['F_MISS'] > thresh['miss_thresh']
    qc['fail_het'] = (qc['F'] < thresh['f_min']) | (qc['F'] > thresh['f_max'])
    qc['qc_pass'] = ~(qc['fail_missing'] | qc['fail_het'])
    
    # Cleanup
    import os
    for ext in ['.bed', '.bim', '.fam', '.smiss', '.het']:
        try: os.remove(f"{temp_prefix}{ext}")
        except: pass
    
    return {
        "n_samples": len(qc),
        "n_pass": qc['qc_pass'].sum(),
        "n_fail": (~qc['qc_pass']).sum(),
        "fail_samples": qc[~qc['qc_pass']]['IID'].tolist(),
        "summary_stats": {
            "mean_missingness": qc['F_MISS'].mean(),
            "mean_het_f": qc['F'].mean(),
            "std_het_f": qc['F'].std()
        }
    }

@tools.tool
def compute_variant_qc(vcf_path: str, source: str) -> dict:
    """
    Compute per-variant QC metrics
    """
    import tempfile
    
    # bcftools stats for Ti/Tv, quality distribution
    stats = subprocess.run(
        ["bcftools", "stats", vcf_path],
        capture_output=True, text=True
    ).stdout
    
    # Extract Ti/Tv
    titv_line = [l for l in stats.split('\n') if l.startswith('TSTV')][0]
    titv = float(titv_line.split('\t')[4])  # ts/tv ratio
    
    # Extract quality distribution
    qual_lines = [l for l in stats.split('\n') if l.startswith('QUAL')]
    # Parse quality distribution (skip for brevity, but important)
    
    # Variant-level stats with plink2
    temp_prefix = tempfile.mktemp(prefix="var_qc_")
    
    subprocess.run([
        "plink2",
        "--vcf", vcf_path,
        "--make-bed",
        "--out", temp_prefix,
        "--max-alleles", "2"
    ], check=True)
    
    # Missing rate per variant
    subprocess.run([
        "plink2",
        "--bfile", temp_prefix,
        "--missing", "variant",
        "--out", temp_prefix
    ], check=True)
    
    # Hardy-Weinberg equilibrium
    subprocess.run([
        "plink2",
        "--bfile", temp_prefix,
        "--hardy",
        "--out", temp_prefix
    ], check=True)
    
    # MAF
    subprocess.run([
        "plink2",
        "--bfile", temp_prefix,
        "--freq",
        "--out", temp_prefix
    ], check=True)
    
    import pandas as pd
    vmiss = pd.read_csv(f"{temp_prefix}.vmiss", delim_whitespace=True)
    hardy = pd.read_csv(f"{temp_prefix}.hardy", delim_whitespace=True)
    freq = pd.read_csv(f"{temp_prefix}.afreq", delim_whitespace=True)
    
    # Merge
    var_qc = vmiss.merge(hardy[['ID', 'P']], on='ID').merge(freq[['ID', 'ALT_FREQS']], on='ID')
    var_qc.columns = ['ID', 'CHROM', 'POS', 'MISSING_CT', 'OBS_CT', 'F_MISS', 'HWE_P', 'MAF']
    
    # Filter criteria (source-specific)
    if source == "mayo_tapestry_helix":
        filters = {"miss_max": 0.05, "hwe_min": 1e-6, "maf_min": 0.0001}
    elif source == "mayo_biobank_regeneron":
        filters = {"miss_max": 0.03, "hwe_min": 1e-6, "maf_min": 0.0001}
    elif source == "ukbiobank":
        filters = {"miss_max": 0.10, "hwe_min": 1e-8, "maf_min": 0.00001}  # More permissive for rare variants
    else:
        filters = {"miss_max": 0.05, "hwe_min": 1e-6, "maf_min": 0.0001}
    
    var_qc['pass'] = (
        (var_qc['F_MISS'] < filters['miss_max']) &
        (var_qc['HWE_P'] > filters['hwe_min'])
        # Note: Not filtering on MAF here (keep rare variants for ML)
    )
    
    # Cleanup
    import os
    for ext in ['.bed', '.bim', '.fam', '.vmiss', '.hardy', '.afreq']:
        try: os.remove(f"{temp_prefix}{ext}")
        except: pass
    
    return {
        "n_variants": len(var_qc),
        "n_pass": var_qc['pass'].sum(),
        "n_fail": (~var_qc['pass']).sum(),
        "ti_tv_ratio": titv,
        "summary_stats": {
            "mean_missingness": var_qc['F_MISS'].mean(),
            "median_maf": var_qc['MAF'].median(),
            "n_rare_variants": (var_qc['MAF'] < 0.01).sum()
        }
    }

@tools.tool
def check_sex_concordance(vcf_path: str, sample_manifest_path: str) -> dict:
    """
    Check reported sex vs genetic sex (F-statistic on chrX)
    Critical for detecting sample swaps
    """
    import tempfile
    import pandas as pd
    
    # Extract chrX
    temp_vcf = tempfile.mktemp(suffix=".vcf.gz")
    subprocess.run([
        "bcftools", "view", "-r", "chrX",
        "-Oz", "-o", temp_vcf, vcf_path
    ], check=True)
    
    subprocess.run(["bcftools", "index", temp_vcf], check=True)
    
    # plink2 sex check
    temp_prefix = tempfile.mktemp(prefix="sex_check_")
    subprocess.run([
        "plink2",
        "--vcf", temp_vcf,
        "--make-bed",
        "--out", temp_prefix
    ], check=True)
    
    subprocess.run([
        "plink2",
        "--bfile", temp_prefix,
        "--check-sex",
        "--out", temp_prefix
    ], check=True)
    
    sex_check = pd.read_csv(f"{temp_prefix}.sexcheck", delim_whitespace=True)
    
    # Load reported sex from manifest
    manifest = pd.read_csv(sample_manifest_path, sep='\t')
    sex_check = sex_check.merge(
        manifest[['sample_id', 'reported_sex']], 
        left_on='IID', 
        right_on='sample_id'
    )
    
    # Check concordance
    sex_check['concordant'] = (
        ((sex_check['SNPSEX'] == 1) & (sex_check['reported_sex'] == 'M')) |
        ((sex_check['SNPSEX'] == 2) & (sex_check['reported_sex'] == 'F'))
    )
    
    # Cleanup
    import os
    os.remove(temp_vcf)
    for ext in ['.bed', '.bim', '.fam', '.sexcheck']:
        try: os.remove(f"{temp_prefix}{ext}")
        except: pass
    
    return {
        "n_samples": len(sex_check),
        "n_concordant": sex_check['concordant'].sum(),
        "n_discordant": (~sex_check['concordant']).sum(),
        "discordant_samples": sex_check[~sex_check['concordant']]['IID'].tolist()
    }

@tools.tool
def generate_qc_report(
    sample_qc: dict, 
    variant_qc: dict, 
    sex_check: dict,
    source: str,
    output_path: str
) -> str:
    """
    Generate comprehensive QC report with visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'WES QC Report - {source}\n{datetime.now().strftime("%Y-%m-%d")}', 
                 fontsize=16)
    
    # Sample missingness distribution
    # (Would plot actual distributions here if we had the raw data)
    # For now, summary text
    axes[0, 0].text(0.1, 0.5, f"""Sample QC Summary:
    Total samples: {sample_qc['n_samples']}
    Pass: {sample_qc['n_pass']} ({100*sample_qc['n_pass']/sample_qc['n_samples']:.1f}%)
    Fail: {sample_qc['n_fail']}
    
    Mean missingness: {sample_qc['summary_stats']['mean_missingness']:.4f}
    Mean F (het): {sample_qc['summary_stats']['mean_het_f']:.4f}
    """, fontsize=10, verticalalignment='center')
    axes[0, 0].set_title('Sample QC')
    axes[0, 0].axis('off')
    
    # Variant QC
    axes[0, 1].text(0.1, 0.5, f"""Variant QC Summary:
    Total variants: {variant_qc['n_variants']:,}
    Pass: {variant_qc['n_pass']:,} ({100*variant_qc['n_pass']/variant_qc['n_variants']:.1f}%)
    Fail: {variant_qc['n_fail']:,}
    
    Ti/Tv ratio: {variant_qc['ti_tv_ratio']:.3f}
    Rare variants (MAF<1%): {variant_qc['summary_stats']['n_rare_variants']:,}
    """, fontsize=10, verticalalignment='center')
    axes[0, 1].set_title('Variant QC')
    axes[0, 1].axis('off')
    
    # Sex check
    axes[0, 2].text(0.1, 0.5, f"""Sex Concordance:
    Total samples: {sex_check['n_samples']}
    Concordant: {sex_check['n_concordant']}
    Discordant: {sex_check['n_discordant']}
    
    {'⚠️ REVIEW REQUIRED' if sex_check['n_discordant'] > 0 else '✓ All concordant'}
    """, fontsize=10, verticalalignment='center')
    axes[0, 2].set_title('Sex Check')
    axes[0, 2].axis('off')
    
    # Decision summary
    overall_pass = (
        sample_qc['n_fail'] / sample_qc['n_samples'] < 0.05 and
        variant_qc['ti_tv_ratio'] > 2.5 and 
        variant_qc['ti_tv_ratio'] < 3.5 and
        sex_check['n_discordant'] == 0
    )
    
    axes[1, 1].text(0.5, 0.5, 
                    '✓ PASS - Proceed to harmonization' if overall_pass 
                    else '✗ FAIL - Manual review required',
                    fontsize=14, fontweight='bold',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='green' if overall_pass else 'red')
    axes[1, 1].axis('off')
    
    # Hide unused subplots
    axes[1, 0].axis('off')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

# QC Agent
qc_agent = agents.Agent(
    model="gemini-2.5-flash",
    tools=[
        compute_sample_qc,
        compute_variant_qc,
        check_sex_concordance,
        generate_qc_report
    ],
    system_prompt="""You are a genomics QC specialist. Execute comprehensive quality control:
    
    1. Sample-level QC: missingness, heterozygosity, Ti/Tv
    2. Variant-level QC: missing rate, HWE, MAF distribution
    3. Sex concordance check (critical for sample tracking)
    4. Generate visual QC report
    
    Apply source-specific thresholds. Flag failures and recommend actions.
    If >5% samples fail OR Ti/Tv outside 2.5-3.5 OR any sex discordance, trigger manual review.
    """
)
```

---

## Agent 3: Harmonization Agent

**Purpose**: Normalize across sources - handle reference differences, variant representation, exome boundaries.

### Key challenges

**Problem 1: Different exome capture regions**
- Helix panel: ~45Mb
- Regeneron panel: ~46Mb  
- UKB mixed: ~40-45Mb

**Solution**: Intersect to **common exome regions** OR expand to **union** (keep all, impute missing).

**Problem 2: Variant representation**
- Indels may be left-aligned differently
- Multi-allelic sites split inconsistently
- Reference alleles might differ (GRCh38 patches)

**Solution**: Normalize with `bcftools norm` against same reference.

**Problem 3: Annotation inconsistencies**
- Different VEP versions, databases, consequences
- Some sources have rsIDs, others don't

**Solution**: Re-annotate all sources uniformly OR strip to minimal (CHROM/POS/REF/ALT/GT only).

### Implementation

```python
@tools.tool
def normalize_variants(vcf_path: str, reference_fasta: str, source: str) -> str:
    """
    Normalize variant representation using bcftools norm
    - Left-align indels
    - Split multi-allelic sites
    - Match reference alleles
    """
    import tempfile
    
    output_vcf = vcf_path.replace('.vcf.gz', '.normalized.vcf.gz')
    
    # Normalize pipeline
    subprocess.run([
        "bcftools", "norm",
        "--fasta-ref", reference_fasta,
        "-m-both",  # Split multi-allelics to biallelic records
        "--check-ref", "w",  # Warn on REF mismatches
        "-Oz", "-o", output_vcf,
        vcf_path
    ], check=True)
    
    # Index
    subprocess.run(["bcftools", "index", "-t", output_vcf], check=True)
    
    # Log normalization stats
    stats = subprocess.run(
        ["bcftools", "stats", output_vcf],
        capture_output=True, text=True
    ).stdout
    
    n_variants = int([l for l in stats.split('\n') 
                      if 'number of records' in l][0].split('\t')[3])
    
    return {
        "normalized_vcf": output_vcf,
        "n_variants": n_variants,
        "source": source
    }

@tools.tool
def intersect_exome_regions(
    vcf_path: str, 
    bed_path: str,
    operation: Literal["intersect", "exclude"] = "intersect"
) -> str:
    """
    Filter VCF to common exome regions using BED file
    """
    output_vcf = vcf_path.replace('.vcf.gz', '.intersected.vcf.gz')
    
    if operation == "intersect":
        # Keep only variants in BED regions
        subprocess.run([
            "bcftools", "view",
            "-R", bed_path,  # Regions from BED
            "-Oz", "-o", output_vcf,
            vcf_path
        ], check=True)
    else:  # exclude
        # Keep variants NOT in BED regions
        subprocess.run([
            "bcftools", "view",
            "-T", f"^{bed_path}",  # Exclude regions
            "-Oz", "-o", output_vcf,
            vcf_path
        ], check=True)
    
    subprocess.run(["bcftools", "index", "-t", output_vcf], check=True)
    
    return output_vcf

@tools.tool
def generate_common_exome_bed(
    tapestry_vcf: str,
    biobank_vcf: str,
    ukb_vcf: str,
    output_bed: str
) -> str:
    """
    Generate BED file of common exome regions across all sources
    Strategy: intersect variant positions from all 3 sources
    """
    import tempfile
    import pandas as pd
    
    # Extract positions from each VCF
    positions = {}
    for name, vcf in [("tapestry", tapestry_vcf), 
                       ("biobank", biobank_vcf), 
                       ("ukb", ukb_vcf)]:
        result = subprocess.run([
            "bcftools", "query",
            "-f", "%CHROM\t%POS\n",
            vcf
        ], capture_output=True, text=True)
        
        df = pd.DataFrame([l.split('\t') for l in result.stdout.strip().split('\n')],
                          columns=['CHROM', 'POS'])
        df['POS'] = df['POS'].astype(int)
        positions[name] = set(zip(df['CHROM'], df['POS']))
    
    # Intersect
    common = positions['tapestry'] & positions['biobank'] & positions['ukb']
    
    # Convert to BED (merge adjacent positions into regions)
    common_df = pd.DataFrame(list(common), columns=['CHROM', 'POS'])
    common_df = common_df.sort_values(['CHROM', 'POS'])
    
    # Merge adjacent positions into regions (± 10bp window)
    regions = []
    current_chrom = None
    current_start = None
    current_end = None
    
    for _, row in common_df.iterrows():
        chrom, pos = row['CHROM'], row['POS']
        
        if chrom != current_chrom or (current_end and pos > current_end + 10):
            # Start new region
            if current_chrom:
                regions.append([current_chrom, current_start, current_end])
            current_chrom = chrom
            current_start = pos
            current_end = pos
        else:
            # Extend region
            current_end = pos
    
    # Add last region
    if current_chrom:
        regions.append([current_chrom, current_start, current_end])
    
    # Write BED
    bed_df = pd.DataFrame(regions, columns=['CHROM', 'START', 'END'])
    bed_df.to_csv(output_bed, sep='\t', header=False, index=False)
    
    return output_bed

@tools.tool
def strip_to_minimal_fields(vcf_path: str) -> str:
    """
    Remove all INFO and FORMAT fields except GT, DP, GQ
    Makes downstream processing simpler and reduces file size
    """
    output_vcf = vcf_path.replace('.vcf.gz', '.minimal.vcf.gz')
    
    subprocess.run([
        "bcftools", "annotate",
        "-x", "INFO,^FORMAT/GT,^FORMAT/DP,^FORMAT/GQ",
        "-Oz", "-o", output_vcf,
        vcf_path
    ], check=True)
    
    subprocess.run(["bcftools", "index", "-t", output_vcf], check=True)
    
    return output_vcf

# Harmonization agent
harmonization_agent = agents.Agent(
    model="gemini-2.5-flash",
    tools=[
        normalize_variants,
        intersect_exome_regions,
        generate_common_exome_bed,
        strip_to_minimal_fields
    ],
    system_prompt="""You harmonize multi-source WES data:
    
    1. Normalize variant representation (left-align, split multi-allelics)
    2. Determine common exome regions across sources OR expand to union
    3. Strip to minimal fields (GT, DP, GQ only)
    
    Your output: harmonized VCF ready for batch correction.
    """
)
```

---

## Agent 4: Batch Normalization Agent

**Purpose**: Correct for technical batch effects while preserving biological signal (especially population structure).

### Batch effect sources

**Sequencing center**: Different sites have systematic coverage/quality differences
**Time**: Technology improves → newer batches have better quality
**Capture kit**: Different enrichment efficiency
**Library prep**: Technical variation

### Strategy

Use **statistical correction** AFTER quality filtering, BEFORE ML training:

1. **PCA-based correction** (for continuous batch effects)
2. **ComBat** (for discrete batches)
3. **Limma removeBatchEffect** (if you have covariates)

**Critical**: Do NOT remove population structure! That's biological signal.

### Implementation

```python
@tools.tool
def compute_pca(plink_prefix: str, n_pcs: int = 20) -> str:
    """
    Compute principal components for batch detection and correction
    """
    # LD prune first (standard for PCA)
    subprocess.run([
        "plink2",
        "--bfile", plink_prefix,
        "--indep-pairwise", "50", "5", "0.2",  # Window, step, r^2
        "--out", f"{plink_prefix}.prune"
    ], check=True)
    
    # Compute PCA on pruned variants
    subprocess.run([
        "plink2",
        "--bfile", plink_prefix,
        "--extract", f"{plink_prefix}.prune.prune.in",
        "--pca", str(n_pcs),
        "--out", f"{plink_prefix}.pca"
    ], check=True)
    
    return f"{plink_prefix}.pca.eigenvec"

@tools.tool
def detect_batch_effects(
    pca_file: str,
    sample_metadata: str,  # TSV with sample_id, source, batch, etc.
    output_plot: str
) -> dict:
    """
    Visualize PCs colored by batch/source to detect batch effects
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load PCA
    pca = pd.read_csv(pca_file, delim_whitespace=True, header=None,
                      names=['FID', 'IID'] + [f'PC{i}' for i in range(1, 21)])
    
    # Load metadata
    meta = pd.read_csv(sample_metadata, sep='\t')
    
    # Merge
    pca_meta = pca.merge(meta, left_on='IID', right_on='sample_id')
    
    # Plot PC1 vs PC2 colored by source
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # By source
    for source in pca_meta['source'].unique():
        subset = pca_meta[pca_meta['source'] == source]
        axes[0].scatter(subset['PC1'], subset['PC2'], 
                        label=source, alpha=0.6, s=10)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()
    axes[0].set_title('PCA colored by data source')
    
    # By batch (if available)
    if 'batch' in pca_meta.columns:
        for batch in pca_meta['batch'].unique():
            subset = pca_meta[pca_meta['batch'] == batch]
            axes[1].scatter(subset['PC1'], subset['PC2'],
                            label=f'Batch {batch}', alpha=0.6, s=10)
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].legend()
        axes[1].set_title('PCA colored by sequencing batch')
    else:
        axes[1].text(0.5, 0.5, 'No batch info available',
                     ha='center', va='center')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    
    # Quantify batch effects using ANOVA on PC1
    from scipy import stats
    
    if 'batch' in pca_meta.columns:
        batch_groups = [pca_meta[pca_meta['batch'] == b]['PC1'].values
                        for b in pca_meta['batch'].unique()]
        f_stat, p_value = stats.f_oneway(*batch_groups)
        batch_effect_detected = p_value < 0.05
    else:
        f_stat, p_value = None, None
        batch_effect_detected = False
    
    # Check source effects
    source_groups = [pca_meta[pca_meta['source'] == s]['PC1'].values
                     for s in pca_meta['source'].unique()]
    f_stat_source, p_value_source = stats.f_oneway(*source_groups)
    source_effect_detected = p_value_source < 0.05
    
    return {
        "plot_path": output_plot,
        "batch_effect_detected": batch_effect_detected,
        "batch_anova_p": p_value,
        "source_effect_detected": source_effect_detected,
        "source_anova_p": p_value_source,
        "recommendation": "Apply batch correction" if batch_effect_detected 
                          else "No batch correction needed"
    }

@tools.tool
def apply_batch_correction_python(
    vcf_path: str,
    pca_file: str,
    sample_metadata: str,
    batch_column: str,
    covariates: list[str],  # e.g., ['PC1', 'PC2', 'age', 'sex']
    output_h5: str
) -> str:
    """
    Apply batch correction and convert to H5 for PyTorch
    
    Strategy:
    1. Convert VCF to genotype matrix (samples x variants)
    2. Apply ComBat or linear model correction
    3. Save as H5 with metadata
    """
    import numpy as np
    import pandas as pd
    import h5py
    from combat.pycombat import pycombat  # pip install combat
    
    # 1. Load genotypes
    print("Loading genotypes from VCF...")
    # Use bcftools query for efficiency
    gt_result = subprocess.run([
        "bcftools", "query",
        "-f", "[%GT\t]\n",
        vcf_path
    ], capture_output=True, text=True)
    
    # Parse genotypes (0/0 → 0, 0/1 → 1, 1/1 → 2, ./. → -1)
    def parse_gt(gt_str):
        if '/' not in gt_str:
            return -1
        alleles = gt_str.split('/')
        if '.' in alleles:
            return -1
        return int(alleles[0]) + int(alleles[1])
    
    genotypes = []
    for line in gt_result.stdout.strip().split('\n'):
        gts = [parse_gt(gt.strip()) for gt in line.split('\t') if gt.strip()]
        genotypes.append(gts)
    
    genotypes = np.array(genotypes, dtype=np.int8)  # variants x samples
    genotypes = genotypes.T  # samples x variants
    
    print(f"Loaded genotypes: {genotypes.shape[0]} samples x {genotypes.shape[1]} variants")
    
    # 2. Load metadata
    meta = pd.read_csv(sample_metadata, sep='\t')
    pca = pd.read_csv(pca_file, delim_whitespace=True, header=None,
                      names=['FID', 'IID'] + [f'PC{i}' for i in range(1, 21)])
    
    meta = meta.merge(pca, left_on='sample_id', right_on='IID')
    
    # 3. Apply ComBat correction
    print(f"Applying batch correction on {batch_column}...")
    
    # ComBat expects: genes x samples (opposite of our layout)
    # Also, only correct AFTER removing missing (-1) variants
    valid_mask = (genotypes != -1).all(axis=0)  # Variants with no missing
    genotypes_valid = genotypes[:, valid_mask].T  # Now variants x samples
    
    # Prepare batch and covariate info
    batch = meta[batch_column].values
    
    if covariates:
        covariate_df = meta[covariates]
        # Ensure numeric
        for col in covariate_df.columns:
            if covariate_df[col].dtype == 'object':
                # Encode categoricals
                covariate_df[col] = pd.Categorical(covariate_df[col]).codes
    else:
        covariate_df = None
    
    # Run ComBat
    corrected = pycombat(
        genotypes_valid,
        batch,
        mod=covariate_df  # Preserve these effects
    )
    
    # Put back into full matrix
    genotypes_corrected = np.copy(genotypes).astype(np.float32)
    genotypes_corrected[:, valid_mask] = corrected.T
    
    # 4. Save as H5
    print(f"Saving to {output_h5}...")
    with h5py.File(output_h5, 'w') as f:
        # Genotypes
        f.create_dataset('genotypes', data=genotypes_corrected, 
                         compression='gzip', compression_opts=9)
        
        # Variant info
        var_result = subprocess.run([
            "bcftools", "query",
            "-f", "%CHROM\t%POS\t%ID\t%REF\t%ALT\n",
            vcf_path
        ], capture_output=True, text=True)
        
        variants = [l.split('\t') for l in var_result.stdout.strip().split('\n')]
        variants_df = pd.DataFrame(variants, 
                                    columns=['CHROM', 'POS', 'ID', 'REF', 'ALT'])
        
        f.create_dataset('variant_chrom', data=variants_df['CHROM'].values.astype('S'))
        f.create_dataset('variant_pos', data=variants_df['POS'].values.astype(int))
        f.create_dataset('variant_id', data=variants_df['ID'].values.astype('S'))
        
        # Sample info
        f.create_dataset('sample_ids', data=meta['sample_id'].values.astype('S'))
        f.create_dataset('sample_source', data=meta['source'].values.astype('S'))
        
        # PCs (for downstream stratification)
        pcs = meta[[f'PC{i}' for i in range(1, 11)]].values
        f.create_dataset('pcs', data=pcs)
        
        # Metadata
        f.attrs['n_samples'] = genotypes_corrected.shape[0]
        f.attrs['n_variants'] = genotypes_corrected.shape[1]
        f.attrs['batch_corrected'] = True
        f.attrs['batch_column'] = batch_column
        f.attrs['covariates'] = ','.join(covariates) if covariates else ''
    
    print(f"✓ Saved batch-corrected data to {output_h5}")
    
    return output_h5

# Batch normalization agent
batch_norm_agent = agents.Agent(
    model="gemini-2.5-pro",  # Need reasoning for when to apply correction
    tools=[
        compute_pca,
        detect_batch_effects,
        apply_batch_correction_python
    ],
    system_prompt="""You correct batch effects in multi-source genomics data:
    
    1. Compute PCA to visualize population structure + batch effects
    2. Statistically test for batch effects (ANOVA on PCs)
    3. If detected, apply ComBat or linear model correction
    4. Preserve biological signal (population structure, phenotype associations)
    
    CRITICAL: Only correct if batch effects detected (p<0.05 on PC1).
    Do NOT over-correct - you'll remove true population differences.
    """
)
```

---

## Agent 5: H5 Conversion & Validation Agent

**Purpose**: Convert to PyTorch-ready format with validation.

### H5 file structure

```
genomics_data.h5
├── genotypes (samples × variants, int8 or float32)
├── variant_chrom (variant IDs, string)
├── variant_pos (positions, int64)
├── variant_id (rsIDs or chr:pos:ref:alt, string)
├── sample_ids (sample IDs, string)
├── sample_source (data source per sample, string)
├── pcs (PCA coordinates, float32, for stratification)
├── phenotypes (optional, if available)
└── metadata (attributes: n_samples, n_variants, build, etc.)
```

### Implementation

```python
@tools.tool
def convert_vcf_to_h5(
    vcf_path: str,
    output_h5: str,
    include_phenotypes: bool = False,
    phenotype_file: str = None
) -> dict:
    """
    Convert VCF to H5 format for PyTorch
    """
    import numpy as np
    import pandas as pd
    import h5py
    
    print(f"Converting {vcf_path} to HDF5...")
    
    # 1. Extract genotypes
    gt_result = subprocess.run([
        "bcftools", "query",
        "-f", "[%GT\t]\n",
        vcf_path
    ], capture_output=True, text=True)
    
    def parse_gt(gt_str):
        if '/' not in gt_str:
            return -1
        alleles = gt_str.split('/')
        if '.' in alleles:
            return -1
        return int(alleles[0]) + int(alleles[1])
    
    genotypes = []
    for line in gt_result.stdout.strip().split('\n'):
        gts = [parse_gt(gt.strip()) for gt in line.split('\t') if gt.strip()]
        genotypes.append(gts)
    
    genotypes = np.array(genotypes, dtype=np.int8).T  # Transpose: samples × variants
    
    # 2. Extract variant info
    var_result = subprocess.run([
        "bcftools", "query",
        "-f", "%CHROM\t%POS\t%ID\t%REF\t%ALT\n",
        vcf_path
    ], capture_output=True, text=True)
    
    variants = [l.split('\t') for l in var_result.stdout.strip().split('\n')]
    variants_df = pd.DataFrame(variants, columns=['CHROM', 'POS', 'ID', 'REF', 'ALT'])
    
    # 3. Extract sample IDs
    samples_result = subprocess.run([
        "bcftools", "query",
        "-l", vcf_path
    ], capture_output=True, text=True)
    
    sample_ids = samples_result.stdout.strip().split('\n')
    
    # 4. Write H5
    with h5py.File(output_h5, 'w') as f:
        # Genotypes
        f.create_dataset('genotypes', data=genotypes, 
                         compression='gzip', compression_opts=9)
        
        # Variants
        f.create_dataset('variant_chrom', 
                         data=variants_df['CHROM'].values.astype('S'))
        f.create_dataset('variant_pos', 
                         data=variants_df['POS'].values.astype(int))
        f.create_dataset('variant_id', 
                         data=variants_df['ID'].values.astype('S'))
        f.create_dataset('variant_ref', 
                         data=variants_df['REF'].values.astype('S'))
        f.create_dataset('variant_alt', 
                         data=variants_df['ALT'].values.astype('S'))
        
        # Samples
        f.create_dataset('sample_ids', 
                         data=np.array(sample_ids).astype('S'))
        
        # Phenotypes (if provided)
        if include_phenotypes and phenotype_file:
            pheno = pd.read_csv(phenotype_file, sep='\t')
            # Assume columns: sample_id, phenotype1, phenotype2, ...
            pheno = pheno.set_index('sample_id').loc[sample_ids]  # Reorder
            
            for col in pheno.columns:
                f.create_dataset(f'phenotype_{col}', 
                                 data=pheno[col].values)
        
        # Metadata
        f.attrs['n_samples'] = genotypes.shape[0]
        f.attrs['n_variants'] = genotypes.shape[1]
        f.attrs['source_vcf'] = vcf_path
        f.attrs['creation_date'] = pd.Timestamp.now().isoformat()
    
    return {
        "h5_path": output_h5,
        "n_samples": genotypes.shape[0],
        "n_variants": genotypes.shape[1],
        "file_size_mb": os.path.getsize(output_h5) / 1e6
    }

@tools.tool
def validate_h5_integrity(h5_path: str) -> dict:
    """
    Validate H5 file structure and data integrity
    """
    import h5py
    import numpy as np
    
    checks = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Check required datasets
        required = ['genotypes', 'variant_chrom', 'variant_pos', 
                    'variant_id', 'sample_ids']
        checks['has_required_datasets'] = all(k in f.keys() for k in required)
        
        # Check dimensions match
        n_samples = f['genotypes'].shape[0]
        n_variants = f['genotypes'].shape[1]
        
        checks['dims_consistent'] = (
            len(f['sample_ids']) == n_samples and
            len(f['variant_chrom']) == n_variants and
            len(f['variant_pos']) == n_variants
        )
        
        # Check genotype values valid
        genotypes = f['genotypes'][:]
        valid_values = set([-1, 0, 1, 2])  # Missing, hom ref, het, hom alt
        unique_vals = set(np.unique(genotypes))
        checks['genotypes_valid'] = unique_vals.issubset(valid_values)
        
        # Check missing rate
        missing_rate = (genotypes == -1).sum() / genotypes.size
        checks['missing_rate'] = missing_rate
        checks['missing_acceptable'] = missing_rate < 0.10
        
        # Check for all-missing variants
        all_missing_vars = (genotypes == -1).all(axis=0).sum()
        checks['n_all_missing_variants'] = all_missing_vars
        checks['no_all_missing'] = all_missing_vars == 0
        
        # Metadata
        checks['n_samples'] = n_samples
        checks['n_variants'] = n_variants
    
    checks['passed'] = all([
        checks['has_required_datasets'],
        checks['dims_consistent'],
        checks['genotypes_valid'],
        checks['missing_acceptable'],
        checks['no_all_missing']
    ])
    
    return checks

@tools.tool
def create_pytorch_dataloader_example(h5_path: str, output_script: str) -> str:
    """
    Generate example PyTorch DataLoader code for the H5 file
    """
    script = f"""
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GenomicsDataset(Dataset):
    \"\"\"
    PyTorch Dataset for genomics data from HDF5
    \"\"\"
    def __init__(self, h5_path: str, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        # Load metadata (keep file open for efficient access)
        self.h5_file = h5py.File(h5_path, 'r')
        self.genotypes = self.h5_file['genotypes']
        self.n_samples = self.genotypes.shape[0]
        self.n_variants = self.genotypes.shape[1]
        
        # Load sample/variant info into memory (small)
        self.sample_ids = self.h5_file['sample_ids'][:].astype(str)
        self.variant_ids = self.h5_file['variant_id'][:].astype(str)
        
        # If PCs available, load them
        if 'pcs' in self.h5_file:
            self.pcs = self.h5_file['pcs'][:]
        else:
            self.pcs = None
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Load genotypes for this sample
        genotypes = self.genotypes[idx, :]
        
        # Convert to tensor
        x = torch.from_numpy(genotypes).float()
        
        # Handle missing (-1) → encode as separate value or impute
        # Option 1: Replace -1 with 0 (common)
        x[x == -1] = 0
        
        # Option 2: One-hot encode including missing
        # x_onehot = torch.nn.functional.one_hot(x.long() + 1, num_classes=4)
        
        if self.transform:
            x = self.transform(x)
        
        # Return genotypes and sample ID
        return {{
            'genotypes': x,
            'sample_id': self.sample_ids[idx],
            'pcs': torch.from_numpy(self.pcs[idx]) if self.pcs is not None else None
        }}
    
    def __del__(self):
        self.h5_file.close()

# Usage example
if __name__ == "__main__":
    # Create dataset
    dataset = GenomicsDataset("{h5_path}")
    
    print(f"Dataset size: {{len(dataset)}} samples, {{dataset.n_variants}} variants")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # For GPU transfer
    )
    
    # Iterate
    for batch in dataloader:
        genotypes = batch['genotypes']  # Shape: (batch_size, n_variants)
        sample_ids = batch['sample_id']
        
        print(f"Batch shape: {{genotypes.shape}}")
        print(f"Sample IDs: {{sample_ids[:5]}}")
        
        # Your model training here
        # outputs = model(genotypes)
        # loss = criterion(outputs, labels)
        # ...
        
        break  # Just show one batch
"""
    
    with open(output_script, 'w') as f:
        f.write(script)
    
    return output_script

# H5 conversion agent
h5_agent = agents.Agent(
    model="gemini-2.5-flash",
    tools=[
        convert_vcf_to_h5,
        validate_h5_integrity,
        create_pytorch_dataloader_example
    ],
    system_prompt="""You convert genomics data to PyTorch-ready HDF5 format:
    
    1. Convert VCF → H5 with proper structure
    2. Validate data integrity (dimensions, missing rates, values)
    3. Generate PyTorch DataLoader example code
    
    Your output is production-ready for ML training.
    """
)
```

---

## Supervisor Agent: Orchestration

```python
from google.adk import agents
from typing import Dict, List

class WESPipelineSupervisor:
    """
    Supervisor orchestrates the 5-agent pipeline
    """
    def __init__(self):
        self.agents = {
            "intake": intake_agent,
            "qc": qc_agent,
            "harmonize": harmonization_agent,
            "batch_norm": batch_norm_agent,
            "h5_convert": h5_agent
        }
        
        self.state = {
            "current_stage": None,
            "vcf_path": None,
            "source": None,
            "qc_passed": False,
            "normalized_vcf": None,
            "batch_corrected": False,
            "output_h5": None,
            "checkpoints": []
        }
    
    async def process_vcf(
        self, 
        vcf_path: str,
        sample_manifest: str,
        reference_fasta: str,
        output_dir: str
    ) -> Dict:
        """
        Execute full pipeline: VCF → validated H5
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.state["vcf_path"] = vcf_path
        
        # Stage 1: Intake
        print("=== Stage 1: Source Intake ===")
        self.state["current_stage"] = "intake"
        
        intake_result = await self.agents["intake"].query(f"""
        Process this VCF file: {vcf_path}
        1. Identify source
        2. Validate integrity
        3. Extract metadata
        """)
        
        # Parse intake result (in production, structured output)
        self.state["source"] = intake_result.get("source", "unknown")
        self.state["checkpoints"].append({"stage": "intake", "result": intake_result})
        
        if not intake_result.get("valid", False):
            return {"status": "failed", "stage": "intake", "reason": "Invalid VCF"}
        
        # Stage 2: QC
        print("=== Stage 2: Quality Control ===")
        self.state["current_stage"] = "qc"
        
        qc_result = await self.agents["qc"].query(f"""
        Run comprehensive QC on: {vcf_path}
        Source: {self.state['source']}
        Sample manifest: {sample_manifest}
        
        1. Sample-level QC
        2. Variant-level QC
        3. Sex concordance check
        4. Generate QC report
        """)
        
        self.state["qc_passed"] = qc_result.get("passed", False)
        self.state["checkpoints"].append({"stage": "qc", "result": qc_result})
        
        if not self.state["qc_passed"]:
            # HITL checkpoint
            print("⚠️  QC FAILED - Manual review required")
            return {
                "status": "qc_failed",
                "qc_report": qc_result.get("report_path"),
                "failing_samples": qc_result.get("fail_samples", []),
                "action_required": "Review QC report and decide: proceed, reprocess, or exclude samples"
            }
        
        # Stage 3: Harmonization
        print("=== Stage 3: Harmonization ===")
        self.state["current_stage"] = "harmonize"
        
        harmonize_result = await self.agents["harmonize"].query(f"""
        Harmonize VCF: {vcf_path}
        Source: {self.state['source']}
        Reference: {reference_fasta}
        
        1. Normalize variants (left-align, split multi-allelics)
        2. Strip to minimal fields (GT, DP, GQ)
        3. Output normalized VCF
        """)
        
        self.state["normalized_vcf"] = harmonize_result.get("output_vcf")
        self.state["checkpoints"].append({"stage": "harmonize", "result": harmonize_result})
        
        # Stage 4: Batch normalization (conditional)
        print("=== Stage 4: Batch Effect Correction ===")
        self.state["current_stage"] = "batch_norm"
        
        batch_result = await self.agents["batch_norm"].query(f"""
        Analyze and correct batch effects:
        VCF: {self.state['normalized_vcf']}
        Sample metadata: {sample_manifest}
        
        1. Compute PCA
        2. Detect batch effects
        3. If detected, apply correction
        4. Output batch-corrected data
        """)
        
        self.state["batch_corrected"] = batch_result.get("corrected", False)
        self.state["checkpoints"].append({"stage": "batch_norm", "result": batch_result})
        
        # Stage 5: H5 conversion
        print("=== Stage 5: H5 Conversion ===")
        self.state["current_stage"] = "h5_convert"
        
        output_h5 = os.path.join(output_dir, f"{self.state['source']}.h5")
        
        h5_result = await self.agents["h5_convert"].query(f"""
        Convert to PyTorch-ready H5:
        Input VCF: {self.state['normalized_vcf']}
        Output H5: {output_h5}
        
        1. Convert VCF → H5
        2. Validate integrity
        3. Generate PyTorch DataLoader example
        """)
        
        self.state["output_h5"] = h5_result.get("h5_path")
        self.state["checkpoints"].append({"stage": "h5_convert", "result": h5_result})
        
        # Final validation
        if h5_result.get("validated", False):
            print("✓ Pipeline complete!")
            return {
                "status": "success",
                "output_h5": self.state["output_h5"],
                "source": self.state["source"],
                "n_samples": h5_result.get("n_samples"),
                "n_variants": h5_result.get("n_variants"),
                "batch_corrected": self.state["batch_corrected"],
                "checkpoints": self.state["checkpoints"]
            }
        else:
            return {
                "status": "validation_failed",
                "stage": "h5_convert",
                "reason": "H5 file failed validation"
            }

# Deploy supervisor
supervisor = WESPipelineSupervisor()

# Usage
result = await supervisor.process_vcf(
    vcf_path="gs://mayo-tapestry/batch_01.vcf.gz",
    sample_manifest="gs://mayo-tapestry/batch_01_manifest.tsv",
    reference_fasta="gs://references/GRCh38.fa",
    output_dir="gs://mayo-processed/tapestry"
)
```

---

## Deployment on GCP

### Infrastructure setup

```bash
# 1. Enable services
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  batch.googleapis.com

# 2. Create storage buckets
gsutil mb -l us-central1 gs://mayo-wes-raw
gsutil mb -l us-central1 gs://mayo-wes-processed
gsutil mb -l us-central1 gs://mayo-wes-references

# 3. Upload reference genome
gsutil cp GRCh38.fa gs://mayo-wes-references/
gsutil cp GRCh38.fa.fai gs://mayo-wes-references/

# 4. Create MCP server for bioinformatics tools
cd bioinformatics_mcp
gcloud run deploy bioinf-tools-mcp \
  --source . \
  --region us-central1 \
  --cpu 8 --memory 32Gi \
  --timeout 3600 \
  --max-instances 20 \
  --set-env-vars REFERENCE_FASTA=gs://mayo-wes-references/GRCh38.fa
```

### Agent deployment

```python
# deploy_agents.py
from google.adk import agents
from vertexai import agent_engines

# Deploy each agent
agents_config = [
    {"name": "intake", "agent": intake_agent, "cpu": 2, "memory": "8Gi"},
    {"name": "qc", "agent": qc_agent, "cpu": 4, "memory": "16Gi"},
    {"name": "harmonize", "agent": harmonization_agent, "cpu": 4, "memory": "16Gi"},
    {"name": "batch_norm", "agent": batch_norm_agent, "cpu": 8, "memory": "32Gi"},
    {"name": "h5_convert", "agent": h5_agent, "cpu": 4, "memory": "16Gi"}
]

deployed_agents = {}

for config in agents_config:
    print(f"Deploying {config['name']} agent...")
    
    remote_agent = agent_engines.create(
        config["agent"],
        display_name=f"wes-pipeline-{config['name']}",
        requirements=[
            "google-cloud-aiplatform[agent_engines]",
            "plink2",  # Installed via system packages
            "bcftools",
            "pandas",
            "numpy",
            "h5py",
            "matplotlib",
            "seaborn",
            "scipy",
            "combat"  # For batch correction
        ],
        system_packages=["plink2", "bcftools", "tabix"],
        resources={
            "cpu": config["cpu"],
            "memory": config["memory"]
        }
    )
    
    deployed_agents[config["name"]] = remote_agent
    print(f"✓ Deployed: {remote_agent.resource_name}")

# Deploy supervisor
supervisor_remote = agent_engines.create(
    supervisor,
    display_name="wes-pipeline-supervisor",
    requirements=["google-cloud-aiplatform[agent_engines]"],
    resources={"cpu": 2, "memory": "8Gi"}
)

print("✓ All agents deployed!")
```

### Batch processing script

```python
# process_all_sources.py
import asyncio
from pathlib import Path

# Define sources
sources = [
    {
        "name": "mayo_tapestry_helix",
        "vcf_path": "gs://mayo-wes-raw/tapestry/all_samples.vcf.gz",
        "manifest": "gs://mayo-wes-raw/tapestry/manifest.tsv"
    },
    {
        "name": "mayo_biobank_regeneron",
        "vcf_path": "gs://mayo-wes-raw/biobank/all_samples.vcf.gz",
        "manifest": "gs://mayo-wes-raw/biobank/manifest.tsv"
    },
    {
        "name": "ukbiobank",
        "vcf_path": "gs://mayo-wes-raw/ukb/all_samples.vcf.gz",
        "manifest": "gs://mayo-wes-raw/ukb/manifest.tsv"
    }
]

async def process_all():
    results = {}
    
    for source in sources:
        print(f"\n{'='*60}")
        print(f"Processing {source['name']}")
        print('='*60)
        
        result = await supervisor.process_vcf(
            vcf_path=source["vcf_path"],
            sample_manifest=source["manifest"],
            reference_fasta="gs://mayo-wes-references/GRCh38.fa",
            output_dir=f"gs://mayo-wes-processed/{source['name']}"
        )
        
        results[source['name']] = result
        
        if result['status'] == 'success':
            print(f"✓ {source['name']}: SUCCESS")
            print(f"  Output: {result['output_h5']}")
            print(f"  Samples: {result['n_samples']:,}")
            print(f"  Variants: {result['n_variants']:,}")
        else:
            print(f"✗ {source['name']}: FAILED")
            print(f"  Reason: {result.get('reason', 'Unknown')}")
    
    return results

# Run
results = asyncio.run(process_all())

# Generate summary report
print("\n" + "="*60)
print("PIPELINE SUMMARY")
print("="*60)

for source, result in results.items():
    print(f"\n{source}:")
    print(f"  Status: {result['status']}")
    if result['status'] == 'success':
        print(f"  H5 file: {result['output_h5']}")
        print(f"  Ready for PyTorch training")
```

---

## Validation & Best Practices

### Reproducibility checklist

- [ ] All tools versioned (bcftools 1.18, plink2 2.00, Python 3.11)
- [ ] Reference genome version documented (GRCh38 patch 14)
- [ ] Random seeds set for PCA, batch correction
- [ ] Environment containerized (Docker image with fixed versions)
- [ ] Pipeline execution logged (Cloud Logging)
- [ ] All intermediate files retained (lineage tracking)

### QC metrics to monitor

**Sample-level**:
- Missingness: <5% (Tapestry/Regeneron), <10% (UKB)
- Heterozygosity F: -0.15 to 0.15
- Ti/Tv: 2.8-3.2
- Sex concordance: 100%

**Variant-level**:
- Missing rate: <5%
- HWE p-value: >1e-6
- Batch-specific MAF consistency (no single-batch variants)

**Batch effects**:
- PC1 should NOT separate by sequencing batch (ANOVA p>0.05)
- PC1-2 should separate by population structure (expected)

### EDA for validation

```python
# eda_validation.py
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def validate_multi_source_h5(h5_paths: list, output_report: str):
    """
    Comprehensive EDA across all sources
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Multi-Source WES Data Validation', fontsize=16)
    
    all_data = []
    
    for h5_path in h5_paths:
        with h5py.File(h5_path, 'r') as f:
            source = h5_path.split('/')[-1].replace('.h5', '')
            
            genotypes = f['genotypes'][:]
            pcs = f['pcs'][:] if 'pcs' in f else None
            
            all_data.append({
                'source': source,
                'genotypes': genotypes,
                'pcs': pcs,
                'n_samples': genotypes.shape[0],
                'n_variants': genotypes.shape[1]
            })
    
    # Plot 1: Sample counts by source
    sources = [d['source'] for d in all_data]
    counts = [d['n_samples'] for d in all_data]
    axes[0, 0].bar(sources, counts)
    axes[0, 0].set_title('Sample Counts by Source')
    axes[0, 0].set_ylabel('N samples')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Variant counts
    var_counts = [d['n_variants'] for d in all_data]
    axes[0, 1].bar(sources, var_counts)
    axes[0, 1].set_title('Variant Counts by Source')
    axes[0, 1].set_ylabel('N variants')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Missing rate distribution
    for i, d in enumerate(all_data):
        missing_rate = (d['genotypes'] == -1).sum(axis=1) / d['n_variants']
        axes[0, 2].hist(missing_rate, bins=50, alpha=0.5, label=d['source'])
    axes[0, 2].set_title('Missing Rate Distribution')
    axes[0, 2].set_xlabel('Missing rate')
    axes[0, 2].legend()
    
    # Plot 4: MAF distribution
    for i, d in enumerate(all_data):
        # Compute MAF per variant
        gts = d['genotypes'].copy()
        gts[gts == -1] = 0  # Treat missing as ref
        maf = gts.sum(axis=0) / (2 * d['n_samples'])
        maf = np.minimum(maf, 1 - maf)  # Minor allele
        
        axes[1, 0].hist(np.log10(maf[maf > 0]), bins=50, 
                        alpha=0.5, label=d['source'])
    axes[1, 0].set_title('MAF Distribution (log scale)')
    axes[1, 0].set_xlabel('log10(MAF)')
    axes[1, 0].legend()
    
    # Plot 5: PCA comparison (PC1 vs PC2)
    for i, d in enumerate(all_data):
        if d['pcs'] is not None:
            axes[1, 1].scatter(d['pcs'][:, 0], d['pcs'][:, 1],
                               alpha=0.5, s=5, label=d['source'])
    axes[1, 1].set_title('PCA: PC1 vs PC2 (all sources)')
    axes[1, 1].set_xlabel('PC1')
    axes[1, 1].set_ylabel('PC2')
    axes[1, 1].legend()
    
    # Plot 6: PC1 distribution by source (check batch effects)
    pc1_data = []
    for d in all_data:
        if d['pcs'] is not None:
            pc1_data.append(pd.DataFrame({
                'PC1': d['pcs'][:, 0],
                'source': d['source']
            }))
    
    if pc1_data:
        pc1_df = pd.concat(pc1_data)
        sns.boxplot(data=pc1_df, x='source', y='PC1', ax=axes[1, 2])
        axes[1, 2].set_title('PC1 Distribution by Source')
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    # Plot 7: Heterozygosity distribution
    for i, d in enumerate(all_data):
        gts = d['genotypes'].copy()
        gts[gts == -1] = 0
        het_count = (gts == 1).sum(axis=1)
        het_rate = het_count / d['n_variants']
        axes[2, 0].hist(het_rate, bins=50, alpha=0.5, label=d['source'])
    axes[2, 0].set_title('Heterozygosity Rate Distribution')
    axes[2, 0].set_xlabel('Het rate')
    axes[2, 0].legend()
    
    # Plot 8: Ti/Tv estimate (approximate from genotype data)
    # (Would need variant annotations for exact Ti/Tv)
    axes[2, 1].text(0.5, 0.5, 'Ti/Tv calculated during QC\nSee QC reports',
                    ha='center', va='center')
    axes[2, 1].axis('off')
    
    # Plot 9: Summary statistics table
    summary_text = "Summary Statistics\n\n"
    for d in all_data:
        summary_text += f"{d['source']}:\n"
        summary_text += f"  Samples: {d['n_samples']:,}\n"
        summary_text += f"  Variants: {d['n_variants']:,}\n"
        missing_rate = (d['genotypes'] == -1).sum() / d['genotypes'].size
        summary_text += f"  Missing rate: {missing_rate:.4f}\n\n"
    
    axes[2, 2].text(0.05, 0.95, summary_text,
                    verticalalignment='top', fontsize=9,
                    fontfamily='monospace')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_report, dpi=150)
    plt.close()
    
    print(f"✓ Validation report saved: {output_report}")

# Run validation
h5_files = [
    "gs://mayo-wes-processed/mayo_tapestry_helix/mayo_tapestry_helix.h5",
    "gs://mayo-wes-processed/mayo_biobank_regeneron/mayo_biobank_regeneron.h5",
    "gs://mayo-wes-processed/ukbiobank/ukbiobank.h5"
]

validate_multi_source_h5(h5_files, "multi_source_validation.png")
```

---

## Cost & Timeline

### Cost estimate

**GCP services**:
- Agent Engine: ~$300/month (6 agents, 20% utilization)
- Cloud Run (MCP server): ~$50/month (10K tool calls)
- Cloud Storage: ~$100/month (5TB WES data)
- Vertex AI Training (PyTorch): Variable, ~$200/week if training actively

**Total**: ~$500-800/month for production pipeline.

### Implementation timeline

**Week 1-2**: Infrastructure + Agent 1 (Intake)
- GCP setup, bucket creation
- Deploy MCP server with bcftools/plink2
- Intake agent + validation tests

**Week 3-4**: Agent 2 (QC)
- Implement all QC metrics
- Test on Tapestry data (known good)
- Tune thresholds

**Week 5-6**: Agent 3 (Harmonization)
- Normalization pipeline
- Test on all 3 sources
- Generate common exome BED

**Week 7-8**: Agent 4 (Batch correction)
- PCA + batch detection
- ComBat integration
- Validate on mixed cohorts

**Week 9-10**: Agent 5 (H5 conversion)
- H5 writer with compression
- PyTorch DataLoader
- Validation suite

**Week 11-12**: Integration + Testing
- Supervisor orchestration
- End-to-end tests (all 3 sources)
- Production deployment

**Total: 12 weeks to production-ready system**

---

## Key Takeaways

1. **Agent architecture**: 5 specialized agents (Intake, QC, Harmonization, Batch Norm, H5 Convert) + supervisor. Each = pipeline stage.

2. **Source-aware processing**: Different thresholds for Tapestry (Helix), BioBank (Regeneron), UK Biobank due to different capture kits, coverage, variant calling.

3. **QC is critical**: Sample-level (missingness, het, Ti/Tv, sex check), variant-level (missing, HWE, MAF), cohort-level (PCA, batch effects). STOP pipeline if QC fails.

4. **Batch correction**: Use PCA + ComBat ONLY if batch effects detected (PC1 ANOVA p<0.05). Preserve population structure.

5. **H5 format**: Efficient for PyTorch. Includes genotypes, PCs, metadata. Validation built-in.

6. **Reproducibility**: Version everything, log all operations, retain intermediates, containerize environment.

7. **Deployment**: Agent Engine for agents, Cloud Run for tools (bcftools, plink2), Cloud Storage for data. Batch processing via Cloud Workflows.

8. **Timeline**: 12 weeks. Start with Tapestry (known workflow), extend to BioBank + UKB.

**Next step**: Implement Agent 1 (Intake) this week. Test on your existing Tapestry VCF. Validate against your current manual pipeline.
