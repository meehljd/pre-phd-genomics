# Week 2 Study Plan: Nov 3 - Nov 9, 2025
## Phase B Phase 1: LD-Pruning + Ancestry PCA + Subspace Removal

---

## OVERVIEW FOR WEEK 2

**Goal:** Build ancestry-aware debiasing infrastructure for fair genomic ML models

**Focus areas:**
1. Ancestry-stratified LD computation and pruning
2. Genetic ancestry PCA (continuous, not self-reported categories)
3. Subspace removal debiasing (project out ancestry-correlated embeddings)
4. Validation: Mutual Information reduction

**Time budget:**
- Weekday mornings (Mon-Fri): 1.5 hrs each = 7.5 hrs total
- Sunday: 2.5 hr block = 2.5 hrs
- **Total: ~10 hrs** (fits within Phase B Phase 1 budget of 25-35 hrs across 4 weeks)

**Deliverable by Sunday:** 
- Ancestry-stratified LD matrices computed
- LD-pruned variant list (removes LD-confounded variants)
- Genetic ancestry PCA embeddings (PC1-PC10)
- Subspace removal implementation working
- Validation: >80% MI reduction between ancestry and embeddings

**Phase context:**
- **Phase B Phase 1** = 4 weeks total (Nov 3025)
- **Week 1** (last week): Completed Track A Phase 1 (ESM2 attention)
- **Week 2** (this week): LD + ancestry PCA + subspace removal
- **Week 3** (next week): Sex handling + fairness validation matrices
- **Week 4**: Optional INLP/fairness constraints

---

## WEEKDAY MORNINGS (Mon Nov 3 - Fri Nov 7) - 1.5 hrs each

### Monday Nov 3 (1.5 hours) - Data Prep + Tool Setup

**What to do:**

1. **Set up directory structure (10 min)**
   ```bash
   cd ~/pre-phd-genomics
   mkdir -p 02_debiasing/{data,notebooks,scripts,figures,docs}
   mkdir -p 02_debiasing/data/ld_matrices
   
   # Create README
   echo "# Phase B Phase 1: Debiasing" > 02_debiasing/README.md
   echo "Week 2: LD-pruning + Ancestry PCA + Subspace removal" >> 02_debiasing/README.md
   ```

2. **Install Plink 2.0 (30 min)**
   ```bash
   # Plink 2.0 for LD computation and PCA
   cd ~/tools
   wget https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_x86_64_20241124.zip
   unzip plink2_linux_x86_64_20241124.zip
   sudo mv plink2 /usr/local/bin/
   plink2 --version
   
   # Verify installation
   plink2 --help | head -20
   
   # Document in environment setup
   echo "plink2==2.00a5" >> requirements.txt
   ```

3. **Access Mayo cohort data OR set up 1000 Genomes fallback (50 min)**
   
   **Option A: Mayo cohort (if available)**
   ```bash
   # Work with Eric Klee's lab to get VCF access
   # Expected location: /path/to/mayo_rare_disease_cohort.vcf.gz
   # Document access path in 02_debiasing/docs/data_access.md
   ```
   
   **Option B: 1000 Genomes Phase 3 (fallback)**
   ```bash
   # Download 1000 Genomes Phase 3 for testing
   cd 02_debiasing/data
   
   # Download chromosome 22 (small, fast for testing)
   wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
   wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz.tbi
   
   # Download population/ancestry labels
   wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
   
   # Create ancestry-specific VCFs (for LD computation)
   # EUR, AFR, EAS, SAS populations
   ```
   
   **Document data source in notebook:**
   ```python
   # File: 02_debiasing/docs/data_access.md
   
   # Data Source
   - **Primary**: Mayo Clinic rare disease cohort (pending access)
   - **Fallback**: 1000 Genomes Phase 3, chr22
   - **Ancestry labels**: Genetic PCA (computed), not self-reported
   
   # Expected format
   - VCF format (gzipped)
   - ~2,000-5,000 subjects
   - EUR, AFR, EAS, SAS ancestries represented
   ```

**Output:** 
- Directory structure created
- Plink 2.0 installed and working
- Data access confirmed (Mayo or 1000G)
- `02_debiasing/docs/data_access.md` documented

**If stuck:** If Mayo data not available, proceed with 1000G. Pipeline is VCF-agnostic.

---

### Tuesday Nov 4 (1.5 hours) - Ancestry-Stratified LD Computation

**What to do:**

1. **Split cohort by ancestry (30 min)**
   ```bash
   # File: 02_debiasing/scripts/split_by_ancestry.sh
   
   #!/bin/bash
   # Split VCF by ancestry for LD computation
   
   # If using 1000 Genomes, extract by superpopulation
   for ancestry in EUR AFR EAS SAS; do
     # Get sample IDs for this ancestry
     grep $ancestry integrated_call_samples_v3.20130502.ALL.panel | \
       cut -f1 > samples_${ancestry}.txt
     
     # Extract samples from VCF
     bcftools view -S samples_${ancestry}.txt \
       ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz \
       -Oz -o cohort_${ancestry}.vcf.gz
     
     # Index
     tabix -p vcf cohort_${ancestry}.vcf.gz
   done
   ```

2. **Compute LD matrices per ancestry (45 min setup + overnight run)**
   ```bash
   # File: 02_debiasing/scripts/compute_ld_by_ancestry.sh
   
   #!/bin/bash
   # Compute LD (r²) matrices for each ancestry
   
   mkdir -p 02_debiasing/data/ld_matrices
   
   for ancestry in EUR AFR EAS SAS; do
     echo "Computing LD for ${ancestry}..."
     
     plink2 --vcf data/cohort_${ancestry}.vcf.gz \
            --r2 \
            --ld-window-kb 1000 \
            --ld-window-r2 0.1 \
            --out data/ld_matrices/${ancestry}_ld \
            --threads 4 \
            --memory 8000
     
     echo "${ancestry} LD computation complete"
   done
   
   echo "All LD matrices computed!"
   ```
   
   **Run script:**
   ```bash
   chmod +x 02_debiasing/scripts/compute_ld_by_ancestry.sh
   ./02_debiasing/scripts/compute_ld_by_ancestry.sh
   ```
   
   **Expected runtime:** 
   - Chr22 only: ~5-10 min per ancestry
   - Whole genome: ~2-4 hours per ancestry (run overnight)

3. **Document LD parameters (15 min)**
   ```python
   # File: 02_debiasing/docs/ld_parameters.md
   
   # LD Computation Parameters
   
   ## Plink2 settings
   - `--r2`: Output pairwise LD (r²) values
   - `--ld-window-kb 1000`: Consider variants within 1Mb
   - `--ld-window-r2 0.1`: Only report r² > 0.1 (strong LD)
   - `--threads 4`: Parallelize across 4 cores
   
   ## Rationale
   - r² > 0.1 captures strong LD (tagged variants)
   - 1Mb window covers typical LD blocks in human genome
   - Ancestry-stratified: Different populations have different LD structure
   
   ## Expected output
   - EUR: ~500k variant pairs in LD
   - AFR: ~200k variant pairs (lower LD due to larger effective pop size)
   - EAS/SAS: ~400k variant pairs
   ```

**Output:** 
- LD matrices computed for each ancestry
- `02_debiasing/data/ld_matrices/{EUR,AFR,EAS,SAS}_ld.ld` files
- Parameters documented

**If slow:** Let LD computation run overnight Tue-Wed. Continue with Wed tasks using partial results.

---

### Wednesday Dec  4 (1.5 hours) - LD Pruning Implementation

**What to do:**

1. **Implement ancestry-stratified LD pruning (1 hour)**
   ```python
   # File: 02_debiasing/scripts/ld_prune_union.py
   
   """
   LD-prune variants across all ancestries.
   Strategy: Remove variants in LD in ANY ancestry (conservative).
   """
   
   import pandas as pd
   import numpy as np
   from pathlib import Path
   
   def load_ld_matrix(ld_file):
       """Load plink2 LD output"""
       ld = pd.read_csv(ld_file, delim_whitespace=True)
       # Columns: CHR_A, BP_A, SNP_A, CHR_B, BP_B, SNP_B, R2
       return ld
   
   def get_tagged_variants(ld_df, r2_threshold=0.1):
       """Identify variants in LD (tagged by others)"""
       # For each variant pair in LD, mark SNP_B as "tagged"
       # (Keep SNP_A as the "index" variant)
       tagged = ld_df[ld_df['R2'] > r2_threshold]['SNP_B'].unique()
       return set(tagged)
   
   def ld_prune_multi_ancestry(ld_dir, ancestries=['EUR', 'AFR', 'EAS', 'SAS']):
       """LD-prune across multiple ancestries (union)"""
       
       all_tagged = set()
       all_variants = set()
       
       for ancestry in ancestries:
           ld_file = Path(ld_dir) / f"{ancestry}_ld.ld"
           
           if not ld_file.exists():
               print(f"Warning: {ld_file} not found, skipping {ancestry}")
               continue
           
           print(f"Processing {ancestry}...")
           ld_df = load_ld_matrix(ld_file)
           
           # Get all variants in this ancestry
           variants_a = set(ld_df['SNP_A'].unique())
           variants_b = set(ld_df['SNP_B'].unique())
           all_variants.update(variants_a | variants_b)
           
           # Get tagged variants
           tagged = get_tagged_variants(ld_df, r2_threshold=0.1)
           all_tagged.update(tagged)
           
           print(f"  {ancestry}: {len(tagged):,} variants in LD")
       
       # Final variant list: all variants MINUS tagged variants
       final_variants = all_variants - all_tagged
       
       print(f"\nLD Pruning Summary:")
       print(f"  Total variants: {len(all_variants):,}")
       print(f"  Tagged (removed): {len(all_tagged):,}")
       print(f"  Final (kept): {len(final_variants):,}")
       print(f"  Retention rate: {100*len(final_variants)/len(all_variants):.1f}%")
       
       return final_variants, all_tagged
   
   if __name__ == "__main__":
       # Run LD pruning
       final_vars, tagged_vars = ld_prune_multi_ancestry(
           ld_dir='02_debiasing/data/ld_matrices'
       )
       
       # Save results
       with open('02_debiasing/data/final_variants_ldpruned.txt', 'w') as f:
           for var in sorted(final_vars):
               f.write(f"{var}\n")
       
       with open('02_debiasing/data/tagged_variants_removed.txt', 'w') as f:
           for var in sorted(tagged_vars):
               f.write(f"{var}\n")
       
       print("\nVariant lists saved!")
   ```
   
   **Run script:**
   ```bash
   cd ~/pre-phd-genomics
   python 02_debiasing/scripts/ld_prune_union.py
   ```

2. **Visualize LD structure (30 min)**
   ```python
   # File: 02_debiasing/notebooks/01_ld_visualization.ipynb
   
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Load LD data for one ancestry
   ld_eur = pd.read_csv('02_debiasing/data/ld_matrices/EUR_ld.ld', 
                        delim_whitespace=True)
   
   # Sample 1000 variant pairs for visualization
   ld_sample = ld_eur.sample(n=min(1000, len(ld_eur)))
   
   # Plot r² distribution
   plt.figure(figsize=(10, 5))
   plt.hist(ld_sample['R2'], bins=50, edgecolor='black', alpha=0.7)
   plt.xlabel('LD (r²)')
   plt.ylabel('Number of variant pairs')
   plt.title('LD Structure (EUR ancestry)')
   plt.axvline(0.1, color='red', linestyle='--', label='Pruning threshold')
   plt.legend()
   plt.savefig('02_debiasing/figures/ld_distribution_eur.png', dpi=150)
   plt.close()
   
   # Create LD decay plot (r² vs distance)
   ld_sample['distance_kb'] = (ld_sample['BP_B'] - ld_sample['BP_A']) / 1000
   
   plt.figure(figsize=(10, 5))
   plt.scatter(ld_sample['distance_kb'], ld_sample['R2'], 
               alpha=0.3, s=10)
   plt.xlabel('Distance (kb)')
   plt.ylabel('LD (r²)')
   plt.title('LD Decay (EUR ancestry)')
   plt.xlim(0, 1000)
   plt.savefig('02_debiasing/figures/ld_decay_eur.png', dpi=150)
   plt.close()
   
   print("LD visualizations saved!")
   ```

**Output:** 
- LD-pruned variant list saved
- LD structure visualizations (distribution + decay)
- Pruning metadata (how many variants removed per ancestry)

---

### Thursday Nov 6 (1.5 hours) - Ancestry PCA Computation

**What to do:**

1. **Relatedness pruning for PCA (30 min)**
   ```bash
   # File: 02_debiasing/scripts/compute_ancestry_pca.sh
   
   #!/bin/bash
   # Compute genetic ancestry PCAs
   
   # Step 1: Prune for relatedness (independent SNPs for PCA)
   # Use ~50k independent SNPs
   
   plink2 --vcf data/cohort_all.vcf.gz \
          --indep-pairwise 50 5 0.2 \
          --out data/cohort_pruned
   
   # Step 2: Extract pruned SNPs and convert to bed format
   plink2 --vcf data/cohort_all.vcf.gz \
          --extract data/cohort_pruned.prune.in \
          --make-bed \
          --out data/cohort_pruned
   
   # Step 3: Compute PCA (10 components)
   plink2 --bfile data/cohort_pruned \
          --pca 10 \
          --out data/ancestry_pca
   
   echo "Ancestry PCA complete!"
   echo "Output: data/ancestry_pca.eigenvec (PC1-PC10 per subject)"
   ```
   
   **Run script:**
   ```bash
   chmod +x 02_debiasing/scripts/compute_ancestry_pca.sh
   ./02_debiasing/scripts/compute_ancestry_pca.sh
   ```

2. **Load and visualize PCA results (1 hour)**
   ```python
   # File: 02_debiasing/notebooks/02_ancestry_pca_analysis.ipynb
   
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Load PCA results from plink2
   pca_df = pd.read_csv('02_debiasing/data/ancestry_pca.eigenvec', 
                        delim_whitespace=True, header=None)
   pca_df.columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, 11)]
   
   # Load population labels (if using 1000G)
   pop_df = pd.read_csv('02_debiasing/data/integrated_call_samples_v3.20130502.ALL.panel',
                        sep='\t')
   pop_df = pop_df.rename(columns={'sample': 'IID'})
   
   # Merge
   pca_df = pca_df.merge(pop_df[['IID', 'super_pop']], on='IID', how='left')
   
   # Visualize PC1 vs PC2
   plt.figure(figsize=(10, 8))
   
   for pop in ['EUR', 'AFR', 'EAS', 'SAS', 'AMR']:
       mask = pca_df['super_pop'] == pop
       plt.scatter(pca_df.loc[mask, 'PC1'], 
                   pca_df.loc[mask, 'PC2'],
                   label=pop, alpha=0.6, s=30)
   
   plt.xlabel('PC1')
   plt.ylabel('PC2')
   plt.title('Genetic Ancestry PCA')
   plt.legend()
   plt.grid(alpha=0.3)
   plt.savefig('02_debiasing/figures/ancestry_pca_pc1_pc2.png', dpi=150)
   plt.close()
   
   # Scree plot (variance explained)
   eigenval_df = pd.read_csv('02_debiasing/data/ancestry_pca.eigenval',
                             header=None, names=['eigenvalue'])
   
   var_explained = 100 * eigenval_df['eigenvalue'] / eigenval_df['eigenvalue'].sum()
   
   plt.figure(figsize=(10, 5))
   plt.bar(range(1, 11), var_explained[:10])
   plt.xlabel('Principal Component')
   plt.ylabel('Variance Explained (%)')
   plt.title('PCA Scree Plot')
   plt.xticks(range(1, 11))
   plt.savefig('02_debiasing/figures/ancestry_pca_scree.png', dpi=150)
   plt.close()
   
   print(f"PC1 explains {var_explained.iloc[0]:.1f}% of variance")
   print(f"PC2 explains {var_explained.iloc[1]:.1f}% of variance")
   print(f"Top 3 PCs explain {var_explained[:3].sum():.1f}% of variance")
   
   # Create decile bins for stratified sampling
   pca_df['PC1_decile'] = pd.qcut(pca_df['PC1'], q=10, labels=False, duplicates='drop')
   pca_df['PC2_decile'] = pd.qcut(pca_df['PC2'], q=10, labels=False, duplicates='drop')
   pca_df['ancestry_cell'] = pca_df['PC1_decile'] * 10 + pca_df['PC2_decile']
   
   # Save processed PCA data
   pca_df.to_csv('02_debiasing/data/ancestry_pcs_processed.csv', index=False)
   print("PCA data saved with decile bins!")
   ```

**Output:** 
- Ancestry PCA computed (PC1-PC10)
- PC1 vs PC2 visualization colored by population
- Scree plot (variance explained)
- Decile stratification for sampling
- `ancestry_pcs_processed.csv` saved

---

### Friday Nov 7 (1.5 hours) - Subspace Removal Setup

**What to do:**

1. **Train ancestry regression head (1 hour)**
   ```python
   # File: 02_debiasing/notebooks/03_subspace_removal.ipynb
   
   import torch
   import torch.nn as nn
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Load ancestry PCs
   pca_df = pd.read_csv('02_debiasing/data/ancestry_pcs_processed.csv')
   ancestry_pcs = torch.tensor(pca_df[['PC1', 'PC2', 'PC3']].values, 
                                dtype=torch.float32)
   
   # For now, create dummy encoder embeddings
   # (In Week 3-4, replace with actual model embeddings)
   n_samples = len(pca_df)
   encoder_dim = 512
   
   # Dummy embeddings: add some ancestry signal for testing
   encoder_output = torch.randn(n_samples, encoder_dim)
   
   # Inject ancestry signal into first few dimensions (for testing)
   encoder_output[:, :3] += ancestry_pcs * 0.5  # Weak correlation
   
   print(f"Encoder embeddings shape: {encoder_output.shape}")
   print(f"Ancestry PCs shape: {ancestry_pcs.shape}")
   
   # Define ancestry regression model
   class AncestryRegressor(nn.Module):
       def __init__(self, encoder_dim, n_pcs=3):
           super().__init__()
           self.linear = nn.Linear(encoder_dim, n_pcs)
       
       def forward(self, x):
           return self.linear(x)
   
   # Initialize and train
   model = AncestryRegressor(encoder_dim, n_pcs=3)
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   criterion = nn.MSELoss()
   
   # Training loop
   losses = []
   for epoch in range(200):
       pred_pcs = model(encoder_output)
       loss = criterion(pred_pcs, ancestry_pcs)
       
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       losses.append(loss.item())
       
       if epoch % 50 == 0:
           print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}")
   
   # Plot training curve
   plt.figure(figsize=(10, 5))
   plt.plot(losses)
   plt.xlabel('Epoch')
   plt.ylabel('MSE Loss')
   plt.title('Ancestry Regression Training')
   plt.savefig('02_debiasing/figures/ancestry_regression_training.png', dpi=150)
   plt.close()
   
   # Extract learned weight matrix
   W = model.linear.weight.detach()  # Shape: (3, 512)
   print(f"\nWeight matrix W shape: {W.shape}")
   print(f"W norm: {torch.norm(W).item():.4f}")
   
   # Save model
   torch.save({
       'model_state_dict': model.state_dict(),
       'W': W,
       'encoder_dim': encoder_dim,
       'n_pcs': 3
   }, '02_debiasing/data/ancestry_regressor.pt')
   
   print("\nAncestry regressor trained and saved!")
   ```

2. **Implement subspace projection (30 min)**
   ```python
   # Continue in 03_subspace_removal.ipynb
   
   def debias_embeddings(embeddings, W):
       """
       Remove ancestry-correlated subspace from embeddings.
       
       Args:
           embeddings: (n_samples, encoder_dim) tensor
           W: (n_pcs, encoder_dim) weight matrix from ancestry regressor
       
       Returns:
           debiased_embeddings: (n_samples, encoder_dim) tensor
       """
       # Compute projection matrix: P = W^T W
       projection = torch.mm(W.T, W)  # (encoder_dim, encoder_dim)
       
       # Project out ancestry subspace: e_debiased = e - P @ e
       debiased = embeddings - torch.mm(embeddings, projection)
       
       return debiased
   
   # Apply debiasing
   encoder_debiased = debias_embeddings(encoder_output, W)
   
   print(f"Original embeddings shape: {encoder_output.shape}")
   print(f"Debiased embeddings shape: {encoder_debiased.shape}")
   
   # Visualize embedding space before/after
   from sklearn.decomposition import PCA
   
   # Reduce to 2D for visualization
   pca_viz = PCA(n_components=2)
   embed_2d = pca_viz.fit_transform(encoder_output.numpy())
   embed_debiased_2d = pca_viz.transform(encoder_debiased.numpy())
   
   # Plot
   fig, axes = plt.subplots(1, 2, figsize=(15, 6))
   
   # Before debiasing
   scatter = axes[0].scatter(embed_2d[:, 0], embed_2d[:, 1], 
                             c=ancestry_pcs[:, 0], cmap='viridis', 
                             alpha=0.6, s=20)
   axes[0].set_title('Embeddings BEFORE Debiasing')
   axes[0].set_xlabel('Embedding PC1')
   axes[0].set_ylabel('Embedding PC2')
   plt.colorbar(scatter, ax=axes[0], label='Ancestry PC1')
   
   # After debiasing
   scatter = axes[1].scatter(embed_debiased_2d[:, 0], embed_debiased_2d[:, 1],
                             c=ancestry_pcs[:, 0], cmap='viridis',
                             alpha=0.6, s=20)
   axes[1].set_title('Embeddings AFTER Debiasing')
   axes[1].set_xlabel('Embedding PC1')
   axes[1].set_ylabel('Embedding PC2')
   plt.colorbar(scatter, ax=axes[1], label='Ancestry PC1')
   
   plt.tight_layout()
   plt.savefig('02_debiasing/figures/embeddings_before_after_debiasing.png', dpi=150)
   plt.close()
   
   # Save debiased embeddings
   torch.save({
       'encoder_output': encoder_output,
       'encoder_debiased': encoder_debiased,
       'ancestry_pcs': ancestry_pcs
   }, '02_debiasing/data/embeddings_debiased.pt')
   
   print("\nDebiasing complete! Embeddings saved.")
   ```

**Output:** 
- Ancestry regression head trained
- Weight matrix W saved
- Subspace removal projection implemented
- Before/after debiasing visualization
- Debiased embeddings saved

---

## SUNDAY Nov 9 - Extended Block (2.5 hours)

### Hour 1: Mutual Information Validation (1 hour)

**What to do:**

```python
# File: 02_debiasing/notebooks/04_validation_mi_reduction.ipynb

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

# Load data
data = torch.load('02_debiasing/data/embeddings_debiased.pt')
encoder_output = data['encoder_output'].numpy()
encoder_debiased = data['encoder_debiased'].numpy()
ancestry_pcs = data['ancestry_pcs'].numpy()

# Discretize ancestry PC1 for MI computation (10 bins)
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
ancestry_discrete = discretizer.fit_transform(ancestry_pcs[:, 0:1]).flatten()

print("Computing Mutual Information...")
print(f"Comparing {encoder_output.shape[1]} embedding dimensions vs ancestry PC1")

# Compute MI for each embedding dimension (before debiasing)
mi_before = []
for dim in range(encoder_output.shape[1]):
    mi = mutual_info_score(ancestry_discrete, 
                           KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
                           .fit_transform(encoder_output[:, dim:dim+1]).flatten())
    mi_before.append(mi)

# Compute MI for each embedding dimension (after debiasing)
mi_after = []
for dim in range(encoder_debiased.shape[1]):
    mi = mutual_info_score(ancestry_discrete,
                           KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
                           .fit_transform(encoder_debiased[:, dim:dim+1]).flatten())
    mi_after.append(mi)

mi_before = np.array(mi_before)
mi_after = np.array(mi_after)

# Calculate statistics
mean_mi_before = mi_before.mean()
mean_mi_after = mi_after.mean()
mi_reduction_pct = 100 * (1 - mean_mi_after / mean_mi_before)

print(f"\n{'='*60}")
print(f"MUTUAL INFORMATION RESULTS")
print(f"{'='*60}")
print(f"Mean MI (before):  {mean_mi_before:.6f}")
print(f"Mean MI (after):   {mean_mi_after:.6f}")
print(f"MI reduction:      {mi_reduction_pct:.1f}%")
print(f"{'='*60}")

# Success criterion
if mi_reduction_pct > 80:
    print("✅ SUCCESS: MI reduction > 80% (excellent debiasing)")
elif mi_reduction_pct > 60:
    print("⚠️  MODERATE: MI reduction 60-80% (acceptable, may need INLP in Week 4)")
else:
    print("❌ INSUFFICIENT: MI reduction < 60% (INLP required in Week 4)")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: MI per dimension (before)
axes[0, 0].bar(range(len(mi_before)), mi_before, alpha=0.7, color='red')
axes[0, 0].set_xlabel('Embedding Dimension')
axes[0, 0].set_ylabel('MI(ancestry, dimension)')
axes[0, 0].set_title('Mutual Information BEFORE Debiasing')
axes[0, 0].axhline(mean_mi_before, color='black', linestyle='--', 
                   label=f'Mean = {mean_mi_before:.4f}')
axes[0, 0].legend()

# Plot 2: MI per dimension (after)
axes[0, 1].bar(range(len(mi_after)), mi_after, alpha=0.7, color='green')
axes[0, 1].set_xlabel('Embedding Dimension')
axes[0, 1].set_ylabel('MI(ancestry, dimension)')
axes[0, 1].set_title('Mutual Information AFTER Debiasing')
axes[0, 1].axhline(mean_mi_after, color='black', linestyle='--',
                   label=f'Mean = {mean_mi_after:.4f}')
axes[0, 1].legend()

# Plot 3: Before vs After (overlay)
axes[1, 0].plot(mi_before, label='Before debiasing', alpha=0.7, linewidth=2)
axes[1, 0].plot(mi_after, label='After debiasing', alpha=0.7, linewidth=2)
axes[1, 0].set_xlabel('Embedding Dimension')
axes[1, 0].set_ylabel('MI(ancestry, dimension)')
axes[1, 0].set_title('MI Comparison: Before vs After')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: MI reduction per dimension
mi_reduction_per_dim = 100 * (1 - mi_after / (mi_before + 1e-8))
axes[1, 1].bar(range(len(mi_reduction_per_dim)), mi_reduction_per_dim, 
               alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Embedding Dimension')
axes[1, 1].set_ylabel('MI Reduction (%)')
axes[1, 1].set_title('MI Reduction per Dimension')
axes[1, 1].axhline(80, color='red', linestyle='--', label='80% target')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('02_debiasing/figures/mi_reduction_analysis.png', dpi=150)
plt.close()

print("\nVisualization saved: mi_reduction_analysis.png")

# Which dimensions had strongest ancestry signal?
top_ancestry_dims = np.argsort(mi_before)[-10:][::-1]
print(f"\nTop 10 dimensions with strongest ancestry signal (before debiasing):")
print(top_ancestry_dims)
print(f"MI values: {mi_before[top_ancestry_dims]}")

# Save validation results
validation_results = {
    'mean_mi_before': float(mean_mi_before),
    'mean_mi_after': float(mean_mi_after),
    'mi_reduction_pct': float(mi_reduction_pct),
    'mi_before': mi_before.tolist(),
    'mi_after': mi_after.tolist(),
    'top_ancestry_dims': top_ancestry_dims.tolist()
}

import json
with open('02_debiasing/data/validation_results.json', 'w') as f:
    json.dump(validation_results, f, indent=2)

print("\nValidation results saved!")
```

**Output:** 
- MI reduction quantified (target: >80%)
- Validation plots saved
- Results JSON saved
- Top ancestry-correlated dimensions identified

**Success criterion:** MI reduction > 80% → proceed to Week 3. If < 60%, plan INLP for Week 4.

---

### Hour 2-2.5: Week Summary + Week 3 Planning (1.5 hours)

**What to do:**

1. **Create comprehensive Week 2 summary (45 min)**
   ```markdown
   # File: 02_debiasing/WEEK2_SUMMARY.md
   
   # Week 2 Summary: LD-Pruning + Ancestry PCA + Subspace Removal
   
   ## Completed Tasks
   
   ### 1. LD Computation (Mon-Wed)
   - [x] Installed Plink 2.0
   - [x] Split cohort by ancestry (EUR, AFR, EAS, SAS)
   - [x] Computed LD matrices per ancestry
   - [x] LD parameters: r² > 0.1, window 1Mb
   
   **Results:**
   - EUR: XXX,XXX variant pairs in LD
   - AFR: XXX,XXX variant pairs in LD
   - EAS: XXX,XXX variant pairs in LD
   - SAS: XXX,XXX variant pairs in LD
   
   ### 2. LD Pruning (Wed)
   - [x] Implemented ancestry-stratified pruning
   - [x] Union strategy: remove if in LD in ANY ancestry
   - [x] Visualized LD structure and decay
   
   **Results:**
   - Started with: X,XXX variants
   - Removed (LD-tagged): X,XXX variants
   - Kept (final): X,XXX variants
   - Retention rate: XX%
   
   ### 3. Ancestry PCA (Thu)
   - [x] Relatedness pruning (~50k independent SNPs)
   - [x] Computed PC1-PC10
   - [x] Created decile bins for stratified sampling
   
   **Results:**
   - PC1 explains XX% variance
   - PC2 explains XX% variance
   - Top 3 PCs explain XX% variance
   - Clear ancestry separation visible in PC1 vs PC2
   
   ### 4. Subspace Removal (Fri-Sun)
   - [x] Trained ancestry regression head
   - [x] Implemented projection: e_debiased = e - (W^T W) @ e
   - [x] Validated MI reduction
   
   **Results:**
   - Mean MI before debiasing: X.XXXX
   - Mean MI after debiasing: X.XXXX
   - MI reduction: XX.X%
   - Status: [PASS/NEEDS INLP]
   
   ## Key Insights
   
   1. **LD structure varies by ancestry**: AFR has lower LD (more recombination), 
      EUR/EAS have higher LD. Ancestry-stratified pruning is essential.
   
   2. **Genetic PCA separates ancestries**: PC1 primarily captures EUR-AFR axis, 
      PC2 captures EAS-SAS. Continuous PCs better than discrete categories.
   
   3. **Subspace removal effective**: [XX%] MI reduction indicates [good/moderate] 
      decorrelation. First [X] embedding dimensions most ancestry-correlated.
   
   ## Challenges Encountered
   
   - [List any issues: data access delays, LD computation time, etc.]
   - [Solutions applied]
   
   ## Files Generated
   
   ### Data
   - `data/ld_matrices/{EUR,AFR,EAS,SAS}_ld.ld` - LD matrices
   - `data/final_variants_ldpruned.txt` - Final variant list
   - `data/ancestry_pcs_processed.csv` - PCA embeddings with deciles
   - `data/ancestry_regressor.pt` - Trained regression model
   - `data/embeddings_debiased.pt` - Debiased embeddings
   - `data/validation_results.json` - MI reduction stats
   
   ### Figures
   - `figures/ld_distribution_eur.png` - LD r² distribution
   - `figures/ld_decay_eur.png` - LD decay with distance
   - `figures/ancestry_pca_pc1_pc2.png` - Ancestry PCA plot
   - `figures/ancestry_pca_scree.png` - Variance explained
   - `figures/ancestry_regression_training.png` - Training curve
   - `figures/embeddings_before_after_debiasing.png` - Debiasing effect
   - `figures/mi_reduction_analysis.png` - MI validation plots
   
   ### Code
   - `scripts/compute_ld_by_ancestry.sh` - LD computation
   - `scripts/ld_prune_union.py` - LD pruning logic
   - `scripts/compute_ancestry_pca.sh` - PCA pipeline
   - `notebooks/01_ld_visualization.ipynb` - LD analysis
   - `notebooks/02_ancestry_pca_analysis.ipynb` - PCA analysis
   - `notebooks/03_subspace_removal.ipynb` - Debiasing implementation
   - `notebooks/04_validation_mi_reduction.ipynb` - Validation
   
   ## Integration with Larger Project
   
   - **Phase B Phase 1 progress**: Week 2/4 complete (25% of debiasing phase)
   - **Next**: Week 3 - Sex handling + fairness matrices
   - **Timeline**: On track for 25-35 hr total Phase B Phase 1 budget
   
   ## Preparation for Week 3
   
   - Debiased embeddings ready for sex-stratified analysis
   - Ancestry PCs ready for fairness validation
   - Need to add: X-chromosome hemizygous encoding for males
   
   ---
   
   **Week 2 Status: ✅ COMPLETE**  
   **Ready for Week 3: YES**
   ```

2. **Plan Week 3 in detail (30 min)**
   ```markdown
   # File: docs/week3_plan.md
   
   # Week 3 Plan: Sex Handling + Fairness Validation
   
   ## Goals
   1. Implement X-chromosome sex-specific encoding (hemizygous males)
   2. Train sex-conditional model (sex as covariate, not removed)
   3. Compute fairness validation matrices (ancestry × sex)
   4. Validate stratified performance meets fairness criteria
   
   ## Time Budget
   - Mon-Fri mornings: 1.5 hrs × 5 = 7.5 hrs
   - Sunday: 2.5 hrs
   - Total: ~10 hrs
   
   ## Daily Breakdown (preliminary)
   
   ### Monday: X-chromosome encoding
   - Identify X-linked variants in cohort
   - Code males: 0/1 (hemizygous)
   - Code females: 0/1/2 (diploid)
   - Test on toy example
   
   ### Tuesday: Sex-conditional model
   - Add sex as model input (categorical)
   - Retrain ancestry regressor with sex conditioning
   - Verify: sex signal NOT removed (biological, not bias)
   
   ### Wednesday: Ablation by sex
   - Run variant ablation separately for males/females
   - Measure: Do X-linked variants have sex-specific effects?
   - Validate against biology (e.g., X-linked diseases)
   
   ### Thursday: Fairness matrix computation
   - Compute accuracy/sensitivity/specificity
   - Stratify by: ancestry (EUR/AFR/EAS/SAS) × sex (M/F)
   - Target: All cells within 3-5% of best group
   
   ### Friday: Visualization + validation
   - Heatmap: fairness metrics by ancestry × sex
   - Identify: Which groups underperform?
   - Document: Pass/fail fairness criteria
   
   ### Sunday: Week wrap-up
   - Summary document
   - Plan Week 4 (optional INLP or move to Track C)
   
   ## Success Criteria
   - [ ] X-chromosome encoding implemented correctly
   - [ ] Sex-conditional model trained
   - [ ] Fairness matrix computed (8 cells: 4 ancestry × 2 sex)
   - [ ] All fairness cells within 5% of best group
   
   ## Reading for Week 3
   - Martin et al. 2019 (ancestry bias in polygenic scores)
   - Popejoy & Fullerton 2016 (diversity in genomics)
   
   ---
   
   **Week 3 Preview Created**  
   **Ready to execute Monday Nov 10**
   ```

3. **Git commit and push (15 min)**
   ```bash
   cd ~/pre-phd-genomics
   
   # Stage all Week 2 work
   git add 02_debiasing/
   git add docs/week3_plan.md
   git add requirements.txt
   
   # Commit
   git commit -m "Week 2 complete: LD-pruning + ancestry PCA + subspace removal
   
   - Ancestry-stratified LD matrices computed (EUR/AFR/EAS/SAS)
   - LD-pruned variant list generated (XX% retention)
   - Genetic ancestry PCA (PC1-PC10) with decile stratification
   - Subspace removal debiasing implemented
   - MI reduction: XX.X% (target >80%)
   - All validation plots and results saved
   
   Ready for Week 3: sex handling + fairness matrices"
   
   # Push to remote
   git push origin main
   
   echo "Week 2 work committed and pushed!"
   ```

**Output:** 
- Comprehensive Week 2 summary documented
- Week 3 plan created
- All work committed to Git

---

## GITHUB COMMIT CHECKLIST (By Sunday Evening)

- [ ] Directory structure: `02_debiasing/{data,notebooks,scripts,figures,docs}/`
- [ ] Data files:
  - [ ] LD matrices per ancestry
  - [ ] LD-pruned variant list
  - [ ] Ancestry PCs with deciles
  - [ ] Ancestry regressor model
  - [ ] Debiased embeddings
  - [ ] Validation results JSON
- [ ] Scripts:
  - [ ] `scripts/compute_ld_by_ancestry.sh`
  - [ ] `scripts/ld_prune_union.py`
  - [ ] `scripts/compute_ancestry_pca.sh`
- [ ] Notebooks:
  - [ ] `notebooks/01_ld_visualization.ipynb`
  - [ ] `notebooks/02_ancestry_pca_analysis.ipynb`
  - [ ] `notebooks/03_subspace_removal.ipynb`
  - [ ] `notebooks/04_validation_mi_reduction.ipynb`
- [ ] Figures (all 7 visualizations)
- [ ] Documentation:
  - [ ] `WEEK2_SUMMARY.md`
  - [ ] `docs/data_access.md`
  - [ ] `docs/ld_parameters.md`
- [ ] Planning:
  - [ ] `docs/week3_plan.md`
- [ ] Requirements:
  - [ ] `plink2==2.00a5` in requirements.txt

---

## SUCCESS CRITERIA FOR WEEK 2

**Hard criteria (must have):**
- [ ] Plink 2.0 installed and working
- [ ] LD matrices computed for ≥2 ancestries
- [ ] LD-pruned variant list generated
- [ ] Ancestry PCA computed (PC1-PC10)
- [ ] Subspace removal implemented and tested
- [ ] MI reduction validated (documented, even if <80%)

**Soft criteria (nice to have):**
- [ ] All 4 ancestries (EUR, AFR, EAS, SAS) processed
- [ ] MI reduction >80% (excellent debiasing)
- [ ] Mayo cohort data accessed (vs 1000G fallback)
- [ ] All 7 visualizations generated

**If all hard criteria true → Week 2 success ✅**

---

## EXECUTION NOTES

### 1. Time Management
- **Strict 1.5 hr blocks**: Set timer, stop when time's up (unless debugging critical bug)
- **Monday is setup-heavy**: If tool installation takes >30 min, defer LD computation to Tuesday
- **LD computation can run overnight**: Start Tuesday evening, check Wednesday morning
- **Validation is critical**: Don't skip Sunday's MI analysis - it determines Week 4 plan

### 2. Data Contingencies
- **Mayo VCF not ready?** Use 1000 Genomes Phase 3 (chr22 for speed)
- **1000G download slow?** Use subset: EUR (CEU), AFR (YRI), EAS (CHB), SAS (GIH) only
- **Plink2 issues?** Fall back to plink 1.9 (slower but stable)

### 3. Computational Notes
- **LD computation time**: Chr22 only = ~10 min/ancestry; whole genome = ~2-4 hrs
- **PCA computation**: ~5-15 min depending on sample size
- **MI validation**: ~10-30 min depending on embedding dimensions

### 4. Git Discipline
- Commit daily, even if incomplete: "WIP: LD computation in progress"
- Push to remote daily (backup!)
- Tag Week 2 completion: `git tag week2-complete`

### 5. Ask for Help Early
- Plink2 installation issues? Check GitHub issues or Plink forums
- LD matrix format confusing? Look at Plink documentation: www.cog-genomics.org/plink/2.0/
- MI reduction < 60%? Document why, plan INLP for Week 4

### 6. Quality Over Speed
- Better to complete 4/5 days well than rush through all 5 poorly
- If behind schedule by Thursday, focus on: (1) PCA, (2) subspace removal, (3) MI validation
- Skip optional LD visualizations if time-constrained

---

## IF YOU GET AHEAD / BEHIND

**Ahead (all tasks done by Saturday):**
- Start Week 3 sex handling early (X-chromosome encoding)
- Or: Read Martin et al. 2019 (ancestry bias paper for Week 3)
- Or: Implement alternative debiasing (CORAL, quick to code)

**Behind (stuck on LD computation by Wednesday):**
- Use precomputed gnomAD LD matrices (if available)
- Or: Skip LD pruning for now, focus on PCA + subspace removal
- Plan to complete LD pruning early Week 3

**MI reduction < 60% (validation fails):**
- Don't panic! Document results clearly
- Plan Week 4 for INLP (iterative null-space projection)
- Week 2 is still successful if pipeline works, even if MI high

---

## LOOKING AHEAD: WEEK 3 PREVIEW

**Week 3 focus:** Sex handling + fairness validation
- X-chromosome hemizygous encoding (males 0/1, females 0/1/2)
- Sex as categorical covariate (not removed via subspace projection)
- Fairness matrices: accuracy/sensitivity/specificity by ancestry × sex
- Target: All 8 cells (4 ancestry × 2 sex) within 5% of best group

**Week 4 decision point:**
- If MI reduction >80% in Week 2 → skip INLP, move to Track C
- If MI reduction 60-80% → optional INLP iteration in Week 4
- If MI reduction <60% → INLP required in Week 4

---

## MOTIVATION

**Week 2 is foundational for fairness.** Without ancestry debiasing:
- Models learn population-specific LD patterns (confounding)
- Performance drops on underrepresented ancestries (AFR, SAS)
- Clinical deployment fails fairness audits
- Paper 1 lacks credibility

**Your Week 2 work enables:**
- Fair models that work across all ancestries
- Paper 1 supplement: "Fairness Analysis" section
- PhD defense: "We explicitly addressed ancestry bias"
- Aim 2 prospective trial: Equitable patient outcomes

**Take Week 2 seriously. This is PhD-level work, not just an exercise.**

Good luck! 🧬
