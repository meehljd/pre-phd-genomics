#!/bin/bash
#==============================================================================
# PLINK2 Genome-wide QC and Ancestry PCA Pipeline
#==============================================================================
# Purpose: Process Mayo Tapestry WES cohort for Phase B1 ancestry debiasing
# Input: Per-chromosome VCF files with CDS regions
# Output: QC'd genome-wide pgen + ancestry PCs for stratified analysis
#
# Workflow:
#   1. Extract CDS variants per chromosome → pgen format
#   2. Deduplicate variants (multiallelic handling)
#   3. Merge chromosomes → genome-wide dataset
#   4. Sample/variant QC (missingness, heterozygosity, sex, relatedness)
#   5. LD pruning → independent SNP set for PCA
#   6. Ancestry PCA → genetic ancestry coordinates
#==============================================================================

# Working directory
cd /home/ext_meehl_joshua_mayo_edu/pre-phd-genomics/02_genomics_domain

#------------------------------------------------------------------------------
# PARAMETERS
#------------------------------------------------------------------------------
# Pipeline control flags
do_sex_infer=False
make_pgen=False      # Step 1: Convert VCF → pgen per chromosome
do_dedup=False       # Step 2: Remove duplicate variant IDs
do_merge=True       # Step 3: Merge chromosomes
do_qc=True           # Step 4: Sample/variant QC filters
do_missingness=True
do_het_filter=False
do_relatedness=True
do_ld_prune=True    # Step 5: LD-based SNP pruning
do_pca=True         # Step 6: Ancestry PCA
do_summary=True    # Pipeline summary output

# Chromosomes to process
chromosomes=({1..22} X Y) # Exclude M for QC and PCA.  Haploid; no LD structure.

# Computational resources
n_threads=104       # CPU threads (adjust for cluster/local)

# Sample QC thresholds
mind=0.05           # Max per-sample missingness (5% = remove samples with >5% missing genotypes)
het_sd=3            # Heterozygosity outlier threshold (±3 SD from mean F coefficient)
                    # Flags: contamination (high het), inbreeding/batch effects (low het)

# Variant QC thresholds (applied before LD pruning)
geno=0.02           # Max per-variant missingness (2% = exclude SNPs missing in >2% of samples)
maf=0.01            # Minor allele frequency filter (1% = common variants only)
hwe=1e-10           # Hardy-Weinberg equilibrium p-value (strict for PCA; 1e-6 for GWAS)
                    # Flags: genotyping errors, population stratification, selection

# Relatedness threshold
king_cutoff=0.0884  # Kinship coefficient threshold (0.0884 = 3rd degree relatives)
                    # Remove one from each pair with kinship > threshold

# LD pruning parameters (for PCA independence)
ld_window_kb=1000   # Window size in kb (1 Mb windows)
ld_step=50          # Step size in variant count (50 SNPs)
ld_r2=0.2           # r² threshold (remove SNP pairs with r² > 0.2)
                    # Result: ~100-200k independent SNPs genome-wide

# PCA parameters
n_pcs=20            # Number of principal components to compute

# Directory structure
per_chr_dir="data/plink/tapestry/per_chr"      # Per-chromosome intermediate files
genome_dir="data/plink/tapestry/genome_wide"   # Genome-wide final outputs
mkdir -p ${per_chr_dir} ${genome_dir}


# #------------------------------------------------------------------------------
# # STEP 0: Infer Sex from chrX data
# #------------------------------------------------------------------------------
if [ "$do_sex_infer" = True ] ; then

  # Temporary fix for chrX naming issue
  echo "Renaming chrX to chr25 for plink2 compatibility..."
  bcftools annotate --rename-chrs <(echo "chrX chr25") \
      data/tapestry/vcfs/05-merged-cohort/cohort.chrX.merged.vcf.gz \
      -Oz -o ${per_chr_dir}/chrX_temp.vcf.gz

  # Make pgen for chrX
  plink2 \
    --vcf ${per_chr_dir}/chrX_temp.vcf.gz \
    --make-pgen \
    --out ${per_chr_dir}/chrX_infer \
    --threads ${n_threads}

  # Calculate heterozygosity on chrX
  plink2 \
  --pfile ${per_chr_dir}/chrX_infer \
  --chr 25 \
  --het \
  --out ${genome_dir}/chrX_het \
  --threads ${n_threads}

  # Inspect F coefficient distribution
  awk 'NR>1 {print $5}' ${genome_dir}/chrX_het.het | \
  sort -n | \
  awk '{
      bin = int($1 / 0.1) * 0.1;
      count[bin]++
  }
  END {
      for (b in count) printf "%.1f\t%s\n", b, count[b]
  }' | sort -n

  # Males have high F (~1), females have low F (~0)
  echo "Creating sex_info.txt based on chrX heterozygosity..."
  awk 'NR>1 {
      sex = ($5 > 0.5) ? 1 : 2;  # Male if F>0.5, else Female
      print $1, sex
  }' ${genome_dir}/chrX_het.het > ${genome_dir}/sex_info.txt
  echo "✓ Step 0 complete: Sex info generated"

  # Print sex distribution
  echo "Sex distribution: - 1=male, 2=female"
  awk '{print $3}' ${genome_dir}/sex_info.txt | sort | uniq -c

fi

#------------------------------------------------------------------------------
# STEP 1: Convert VCF to PGEN (per chromosome)
#------------------------------------------------------------------------------
# Extract CDS variants from full WES VCFs using gene coordinates
# Output: chr{1-22,X,Y}_cds.{pgen,pvar,psam}
# Expected: ~500k-2.5M variants per autosome, 97,422 samples
#------------------------------------------------------------------------------
if [ "$make_pgen" = True ] ; then
  echo "========================================="
  echo "STEP 1: Creating per-chromosome pgen files"
  echo "========================================="
  for i in "${chromosomes[@]}"; do
    echo "[Chr ${i}] Extracting CDS variants..."
    # Special handling for chrX (requires --split-par)
    if [ "$i" = "X" ]; then
      # Haploid regions on chrX (PAR1, PAR2) need to be split for proper ploidy handling
      plink2 \
        --vcf data/tapestry/vcfs/05-merged-cohort/cohort.chr${i}.merged.vcf.gz \
        --update-sex ${genome_dir}/sex_info.txt \
        --split-par hg38 \
        --extract range data/plink/ref/gtf_parsing/HS_chr${i}_genes_gtf.tsv \
        --make-pgen \
        --out ${per_chr_dir}/chr${i}_cds \
        --threads ${n_threads}
    else
      # Standard processing for autosomes and chrY
      plink2 \
        --vcf data/tapestry/vcfs/05-merged-cohort/cohort.chr${i}.merged.vcf.gz \
        --extract range data/plink/ref/gtf_parsing/HS_chr${i}_genes_gtf.tsv \
        --make-pgen \
        --out ${per_chr_dir}/chr${i}_cds \
        --threads ${n_threads}
    fi
  done
  echo "✓ Step 1 complete: Per-chromosome pgen files created"
fi

#------------------------------------------------------------------------------
# STEP 2: Deduplicate variants
#------------------------------------------------------------------------------
# Handle multiallelic sites and long indels:
#   - Create unique IDs: CHR:POS:REF:ALT format
#   - Allow alleles up to 237bp (covers most CDS indels/frameshifts)
#   - Remove duplicate positions with inconsistent genotypes
# Output: chr{1-22,X,Y}_cds_unique.{pgen,pvar,psam}
# Expected loss: ~0.4% per chromosome (duplicates + ultra-long SVs)
#------------------------------------------------------------------------------
if [ "$do_dedup" = True ] ; then
  echo "========================================="
  echo "STEP 2: Deduplicating variants"
  echo "========================================="
  for i in "${chromosomes[@]}"; do
    echo "[Chr ${i}] Removing duplicates and setting unique IDs..."
    plink2 --pfile ${per_chr_dir}/chr${i}_cds \
      --set-all-var-ids @:#:\$r:\$a \
      --new-id-max-allele-len 237 missing \
      --rm-dup exclude-mismatch \
      --make-pgen --out ${per_chr_dir}/chr${i}_cds_unique \
      --threads ${n_threads}
  done
  echo "✓ Step 2 complete: Duplicates removed, unique IDs assigned"
fi

#------------------------------------------------------------------------------
# STEP 3: Merge chromosomes
#------------------------------------------------------------------------------
# Combine all chromosomes into single genome-wide dataset
# Output: genome_wide_raw.{pgen,pvar,psam}
# Expected: ~10-12M CDS variants, 97,422 samples
#------------------------------------------------------------------------------
if [ "$do_merge" = True ] ; then
  echo "========================================="
  echo "STEP 3: Merging chromosomes"
  echo "========================================="
  
  # Create merge list (all chromosomes except chr1, which serves as base)
  ls ${per_chr_dir}/chr*_cds_unique.pgen | \
    sed 's/.pgen//' | \
    grep -v chr1_ > ${genome_dir}/merge_list.txt
  
  echo "Merging $(wc -l < ${genome_dir}/merge_list.txt) chromosomes into chr1..."
  plink2 --pfile ${per_chr_dir}/chr1_cds_unique \
    --pmerge-list ${genome_dir}/merge_list.txt pfile \
    --update-sex ${genome_dir}/sex_info.txt \
    --make-pgen --out ${genome_dir}/genome_wide_raw \
    --threads ${n_threads}
  
  echo "✓ Step 3 complete: Genome-wide pgen created"
fi

#------------------------------------------------------------------------------
# STEP 4: Quality Control
#------------------------------------------------------------------------------
# Multi-stage sample and variant filtering:
#   4a. Remove high-missingness samples (>5% missing genotypes)
#   4b. Identify heterozygosity outliers (contamination/inbreeding)
#   4c. Infer genetic sex from X/Y chromosome heterozygosity
#   4d. Remove related individuals (keep one per family, kinship < 0.0884)
# 
# Expected removals (Mayo cohort):
#   - Missingness: ~100-500 samples
#   - Heterozygosity: ~200-500 samples  
#   - Relatedness: ~500 samples
#   - Final: ~96,000 unrelated samples for PCA
#------------------------------------------------------------------------------
if [ "$do_qc" = True ] ; then
  echo "========================================="
  echo "STEP 4: Quality Control"
  echo "========================================="
  
  #-----------------------------------
  # 4a. Sample call rate filter
  #-----------------------------------
  if [ "$do_missingness" = True ] ; then
    echo "[QC-1] Filtering samples with >${mind} missingness..."
    plink2 --pfile ${genome_dir}/genome_wide_raw \
        --mind ${mind} \
        --make-pgen --out ${genome_dir}/genome_qc \
        --threads ${n_threads}
  fi
  #-----------------------------------
  # 4b. Heterozygosity filter
  #-----------------------------------
  # NOTE: This doesn't work for CDS only sequences, since they are under selection.  Need neutral sites.
  if [ "$do_het_filter" = True ] ; then
    echo "[QC-2] Calculating inbreeding coefficient (F) per sample..."
      plink2 --pfile ${genome_dir}/genome_qc \
        --het \
        --out ${genome_dir}/het_check \
        --threads ${n_threads}
    
      # Compute mean and SD of F coefficient
      awk 'NR>1 {sum+=$6; sumsq+=$6*$6; n++} END {
        mean=sum/n; 
        sd=sqrt(sumsq/n - mean*mean);
        lower=mean-'${het_sd}'*sd;
        upper=mean+'${het_sd}'*sd;
        print "Mean F:", mean, "SD:", sd;
        print "Outlier thresholds: [" lower, ",", upper "]"
      }' ${genome_dir}/het_check.het
    
      echo "[QC-2] Flagging heterozygosity outliers (±${het_sd} SD)..."
      awk 'NR==1 {next} {
        sum+=$6; sumsq+=$6*$6; n++; f[n]=$6; id1[n]=$1; id2[n]=$2
      } END {
        mean=sum/n; sd=sqrt(sumsq/n - mean*mean);
        lower=mean-'${het_sd}'*sd; upper=mean+'${het_sd}'*sd;
        for(i=1; i<=n; i++) {
          if(f[i] < lower || f[i] > upper) print id1[i], id2[i]
        }
      }' ${genome_dir}/het_check.het > ${genome_dir}/het_outliers.txt
    
      n_het_outliers=$(wc -l < ${genome_dir}/het_outliers.txt)
      echo "  → Removing ${n_het_outliers} heterozygosity outliers"
    
      plink2 --pfile ${genome_dir}/genome_qc \
        --remove ${genome_dir}/het_outliers.txt \
        --make-pgen --out ${genome_dir}/genome_qc \
        --threads ${n_threads}
  fi

  #-----------------------------------
  # 4d. Relatedness filtering
  #-----------------------------------
  if [ "$do_relatedness" = True ] ; then
    echo "[QC-4] Identifying related individuals (kinship > ${king_cutoff})..."
    plink2 --pfile ${genome_dir}/genome_qc \
        --king-cutoff ${king_cutoff} \
        --out ${genome_dir}/related \
        --threads ${n_threads}
    # Output: related.king.cutoff.in.id (unrelated samples to keep)
    
    n_unrelated=$(wc -l < ${genome_dir}/related.king.cutoff.in.id)
    echo "  → ${n_unrelated} unrelated samples retained for PCA"
  fi
  echo "✓ Step 4 complete: QC filtering done"
fi

#------------------------------------------------------------------------------
# STEP 5: LD Pruning
#------------------------------------------------------------------------------
# Remove SNPs in high linkage disequilibrium to create independent variant set
# Purpose: Prevent over-weighting of LD blocks in PCA (ancestry inference)
# Method: Sliding window (1 Mb), step 50 SNPs, prune pairs with r² > 0.2
# Output: ld_pruned.prune.in (list of ~100-200k independent SNPs)
#------------------------------------------------------------------------------
if [ "$do_ld_prune" = True ] ; then
  echo "========================================="
  echo "STEP 5: LD-based SNP pruning"
  echo "========================================="
  echo "Parameters: window=${ld_window_kb}kb, step=${ld_step}, r²>${ld_r2}"
  
  plink2 --pfile ${genome_dir}/genome_qc \
    --maf ${maf} \
    --geno ${geno} \
    --hwe ${hwe} \
    --indep-pairwise ${ld_step} 5 ${ld_r2} \
    --out ${genome_dir}/ld_pruned \
    --threads ${n_threads}
  
  n_pruned=$(wc -l < ${genome_dir}/ld_pruned.prune.in)
  echo "  → ${n_pruned} independent SNPs retained for PCA"
  echo "✓ Step 5 complete: LD pruning done"
fi

#------------------------------------------------------------------------------
# STEP 6: Ancestry PCA
#------------------------------------------------------------------------------
# Compute principal components on unrelated, LD-pruned variants
# Purpose: Infer genetic ancestry for Phase B1 debiasing (EUR/AFR/EAS/SAS)
# Output: 
#   - ancestry_pca.eigenvec: PC1-PC20 coordinates per sample
#   - ancestry_pca.eigenval: Variance explained per PC
#   - ancestry_pca.acount: Allele counts (for gnomAD projection later)
# Usage: Plot PC1 vs PC2 to identify ancestry clusters → assign labels
#------------------------------------------------------------------------------
if [ "$do_pca" = True ] ; then
  echo "========================================="
  echo "STEP 6: Ancestry PCA"
  echo "========================================="
  echo "Computing ${n_pcs} PCs on unrelated, LD-pruned variants..."
  
  plink2 --pfile ${genome_dir}/genome_qc \
    --keep ${genome_dir}/related.king.cutoff.in.id \
    --extract ${genome_dir}/ld_pruned.prune.in \
    --freq counts \
    --pca ${n_pcs} approx \
    --out ${genome_dir}/ancestry_pca \
    --threads ${n_threads}
  
  echo "✓ Step 6 complete: PCA results written"
  echo ""
  echo "Next steps:"
  echo "  1. Plot PCs: python scripts/plot_ancestry_pca.py"
  echo "  2. Assign ancestry labels via clustering or gnomAD projection"
  echo "  3. Proceed to Phase B1 subspace removal debiasing"
fi

#------------------------------------------------------------------------------
# Pipeline Summary
#------------------------------------------------------------------------------
if [ "$do_summary" = True ] ; then
  echo "========================================="
  echo "PIPELINE COMPLETE"
  echo "========================================="
  echo "Key outputs:"
  echo "  - QC'd genome-wide pgen: ${genome_dir}/genome_wide_sexed.{pgen,pvar,psam}"
  echo "  - Unrelated sample list: ${genome_dir}/related.king.cutoff.in.id"
  echo "  - LD-pruned SNP list: ${genome_dir}/ld_pruned.prune.in"
  echo "  - Ancestry PCs: ${genome_dir}/ancestry_pca.eigenvec"
  echo ""
  echo "For Phase B1 Week 1-2 deliverables:"
  echo "  ✓ Genome-wide LD-pruned dataset ready"
  echo "  ✓ Genetic ancestry PCA computed"
  echo "  → Next: Ancestry stratification + subspace removal"
fi