# Project 3: Patient-Scale Heterogeneous GNN for Rare Disease Diagnosis

**Status:** In Development (Jan-Apr 2026)
**Target Publication:** Bioinformatics or NPJ Digital Medicine (May 2026 submission)

## Overview

This project develops a heterogeneous graph neural network approach to rare disease diagnosis that:
1. Represents each patient as a multi-relational graph (PPI, regulatory, pathway, co-expression)
2. Learns patient-level embeddings via message passing across edge types
3. Predicts diagnostic category with interpretable subgraph explanations
4. Maintains fairness across ancestry groups

## Key Differentiators

| Aspect | Our Approach | Competitors |
|--------|--------------|-------------|
| **Network** | Per-patient heterogeneous graph | Global network or none |
| **Edge Types** | PPI, regulatory, pathway, co-expr | Usually single type |
| **Ancestry** | Explicit fairness constraints | Often ignored |
| **Interpretability** | Attention + GNNExplainer | Black box |

## Quick Start

```bash
# Install dependencies
pip install torch torch-geometric networkx pandas optuna wandb

# Build patient graphs (after data setup)
python src/graph/builder.py --config configs/data_configs.yaml

# Train HetGNN
python src/training/trainer.py --config configs/model_configs.yaml --model han

# Run evaluation
python src/evaluation/metrics.py --model_path results/han/best_model.pt
```

## Directory Structure

```
project3_hetgnn/
├── configs/           # Model and data configuration
├── data/              # Raw and processed data
│   ├── networks/      # STRING, Reactome, ENCODE, GTEx
│   ├── patients/      # VCF, HPO, ancestry
│   ├── graphs/        # Built PyG HeteroData objects
│   └── splits/        # Train/val/test splits
├── src/               # Source code
│   ├── graph/         # Graph construction
│   ├── models/        # R-GCN, HAN implementations
│   ├── baselines/     # ACMG, network propagation, simple GCN
│   ├── training/      # Training loop, samplers, losses
│   ├── evaluation/    # Metrics, fairness
│   └── interpretation/# Attention, GNNExplainer
├── notebooks/         # Exploration and analysis
├── results/           # Experiment outputs
├── figures/           # Publication figures
└── paper/             # Manuscript drafts
```

## Timeline

| Phase | Dates | Focus |
|-------|-------|-------|
| 1. Data Infrastructure | Jan 1-21 | Networks, patients, graphs |
| 2. Model Development | Jan 22 - Feb 18 | Baselines, HetGNN, fairness |
| 3. Experiments | Feb 19 - Mar 18 | Training, ablations, interpretation |
| 4. Paper Writing | Mar 19 - Apr 30 | Figures, manuscript, submission |

## Key Metrics

- **AUROC:** >0.80 (diagnostic prediction)
- **Fairness Gap:** <5% across ancestry groups
- **vs Baselines:** >5% improvement over ACMG-only

## References

See [implementation plan](../../docs/planning/projects/project3_implementation_plan.md) for detailed methodology and full reference list.

## Contact

- **Researcher:** Joshua Meehl
- **Advisor:** Eric Klee (Mayo Clinic)
