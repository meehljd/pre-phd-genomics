# Project 3: Platform Integration Plan

**Date:** 2026-01-09
**Status:** Planning
**Target Branch:** `feature/hetgnn-aggregator`

---

## 1. Architecture Exploration Summary

### Platform Structure

The genomic-foundation-platform follows a four-layer modular architecture:

```
┌─────────────────────────────────┐
│  gfm_eval (Evaluation)          │  ← Metrics, fairness, diagnostics
├─────────────────────────────────┤
│  gfm_tasks (Aggregators + Heads)│  ← HetGNN goes HERE
├─────────────────────────────────┤
│  gfm_foundation (FM Embeddings) │  ← Evo2, NT, DNABERT embeddings
├─────────────────────────────────┤
│  gfm_data (Data Layer)          │  ← VCF, cohorts, ancestry
└─────────────────────────────────┘
```

### Key Finding: HetGNN as Aggregator

HetGNN fits naturally as a **new Aggregator subclass** in `gfm_tasks/`, following ADR-012's modular pattern:

```python
class HeterogeneousGNNAggregator(Aggregator):
    """GNN-based aggregation for rare disease diagnosis."""

    def fit(self, gene_embeddings, labels=None): ...
    def transform(self, gene_embeddings) -> patient_embeddings: ...
    def get_gene_importance_weights(self) -> np.ndarray: ...
    def get_edge_attention_weights(self) -> Dict: ...
```

### Relevant Existing ADRs

| ADR | Title | Relevance |
|-----|-------|-----------|
| ADR-002 | Two-Stage Embedding Pipeline | HetGNN operates on cached embeddings (Stage 2) |
| ADR-010 | Foundation Model Integration | HetGNN consumes FM embeddings |
| ADR-012 | Task Layer and Classifier Training | **CORE** - Defines Aggregator + TaskHead pattern |
| ADR-020 | Extended Task Heads | HetGNN compatible with all head types |
| ADR-013 | Evaluation and Benchmarking | Evaluation framework for HetGNN |

### Current State

- **No existing GNN or graph code** in the platform
- 8 statistical aggregators exist (PCA variants, pooling)
- HetGNN would be the 9th aggregator (first neural network-based)

---

## 2. User Decisions

### Build Location
**Decision:** Build in `genomic-foundation-platform` (not pre-phd-genomics sandbox)

**Rationale:**
- Production-quality codebase for thesis
- Leverages existing infrastructure (evaluation, training, configs)
- Proper testing, CI/CD, documentation
- Publication-ready ("Available at github.com/...")

### ADR Structure
**Decision:** Create ADRs for this approach. Consider whether network loaders and interpretability need separate ADRs or can be combined.

**Options:**
1. **Single ADR (ADR-032):** Covers HetGNN aggregator + network loaders + interpretability
2. **Split ADRs:**
   - ADR-032: HetGNN Aggregator (core)
   - ADR-033: Gene Interaction Network Loaders
   - ADR-034: GNN Interpretability Tools

**Recommendation:** Single ADR with subsections (simpler, cohesive feature)

### Development Workflow
1. Create feature branch: `feature/hetgnn-aggregator`
2. Write ADR(s)
3. Draft implementation code
4. Have validation agents review
5. Test
6. Document
7. Commit/PR

---

## 3. Implementation Plan

### Phase 1: Setup & ADR (Day 1-2)

```bash
# Create feature branch
cd /root/genomic-foundation-platform
git checkout -b feature/hetgnn-aggregator
```

**Deliverables:**
- [ ] Feature branch created
- [ ] ADR-032: Heterogeneous GNN Aggregator
  - Context & problem statement
  - Design decisions
  - Architecture integration
  - Network loader subsection
  - Interpretability subsection
  - Implementation plan

### Phase 2: Core Implementation (Days 3-10)

#### 2.1 HetGNN Aggregator Module
**Location:** `gfm_tasks/gfm_tasks/heterogeneous_gnn.py`

```python
# Core classes to implement:
class HeterogeneousGNNAggregator(Aggregator):
    """Main aggregator following ADR-012 interface."""
    pass

class GATLayer(nn.Module):
    """Graph Attention layer."""
    pass

class RGCNLayer(nn.Module):
    """Relational GCN layer for heterogeneous edges."""
    pass
```

**Estimated:** ~600 LOC

#### 2.2 Network Loaders
**Location:** `gfm_data/gfm_data/network_loaders.py`

```python
class GeneInteractionNetworkLoader:
    @staticmethod
    def load_ppi_network(source="stringdb", confidence=0.7): ...
    @staticmethod
    def load_regulatory_network(source="encode"): ...
    @staticmethod
    def load_pathway_network(source="reactome"): ...
    @staticmethod
    def load_coexpression_network(source="gtex"): ...
    @staticmethod
    def merge_heterogeneous_networks(networks): ...
```

**Estimated:** ~400 LOC

#### 2.3 Interpretability Tools
**Location:** `gfm_eval/gfm_eval/diagnostics/gnn_interpretation.py`

```python
class GNNInterpreter:
    def extract_gene_importance(aggregator, embeddings): ...
    def extract_edge_attention(aggregator, embeddings): ...
    def visualize_subgraph(genes, edges, importance_weights): ...
    def generate_case_study_report(patient_id, aggregator, embeddings): ...
```

**Estimated:** ~300 LOC

### Phase 3: Testing (Days 11-14)

**Test files to create:**
- `gfm_tasks/tests/test_heterogeneous_gnn.py`
  - Unit tests for GNN layers
  - Integration tests for fit/transform
  - Edge case handling
- `gfm_data/tests/test_network_loaders.py`
  - Network loading tests
  - Merge functionality tests
- `gfm_eval/tests/test_gnn_interpretation.py`
  - Interpretation extraction tests

### Phase 4: Documentation & Examples (Days 15-17)

**Deliverables:**
- [ ] Config templates: `configs/tasks/hetgnn_default.yaml`
- [ ] Example notebook: `examples/04-tasks/hetgnn_rare_disease.ipynb`
- [ ] Update README with HetGNN section
- [ ] Docstrings for all public methods

### Phase 5: Review & Merge (Days 18-20)

**Review checklist:**
- [ ] Design reviewer agent: Check against ADR-012, book methodology
- [ ] Code reviewer agent: Check code quality, patterns
- [ ] Run full test suite
- [ ] Update CHANGELOG
- [ ] Create PR with comprehensive description

---

## 4. File Structure (Final State)

```
genomic-foundation-platform/
├── docs/ADR/
│   └── ADR-032-heterogeneous-gnn-aggregator.md   # NEW
├── gfm_tasks/gfm_tasks/
│   ├── base.py                                   # Existing Aggregator ABC
│   ├── aggregators.py                            # Existing 8 aggregators
│   ├── heterogeneous_gnn.py                      # NEW: HetGNN aggregator
│   └── __init__.py                               # Update exports
├── gfm_tasks/tests/
│   └── test_heterogeneous_gnn.py                 # NEW
├── gfm_data/gfm_data/
│   ├── network_loaders.py                        # NEW: Network data sources
│   └── __init__.py                               # Update exports
├── gfm_data/tests/
│   └── test_network_loaders.py                   # NEW
├── gfm_eval/gfm_eval/diagnostics/
│   ├── gnn_interpretation.py                     # NEW: Interpretability
│   └── __init__.py                               # Update exports
├── gfm_eval/tests/
│   └── test_gnn_interpretation.py                # NEW
├── configs/tasks/
│   └── hetgnn_default.yaml                       # NEW: Config template
└── examples/04-tasks/
    └── hetgnn_rare_disease.ipynb                 # NEW: Example notebook
```

---

## 5. Dependencies to Add

```toml
# pyproject.toml additions
[project.dependencies]
torch-geometric = ">=2.4.0"
networkx = ">=3.0"
```

---

## 6. Success Criteria

### Minimum Viable
- [ ] HetGNN aggregator passes all unit tests
- [ ] Can load at least one network source (STRING PPI)
- [ ] Integrates with existing Trainer
- [ ] Basic gene importance extraction works

### Full Implementation
- [ ] All 4 network sources loadable (PPI, regulatory, pathway, co-expression)
- [ ] Heterogeneous edge types supported (R-GCN)
- [ ] Comprehensive interpretability tools
- [ ] Example notebook runs end-to-end
- [ ] Documentation complete

### Publication-Ready
- [ ] Performance benchmarks vs baseline aggregators
- [ ] Fairness evaluation across ancestry groups
- [ ] Case study visualizations
- [ ] Code cleaned and documented for public release

---

## 7. Agent Prompt (for execution)

```
Execute Project 3 HetGNN platform integration:

1. SETUP
   - cd /root/genomic-foundation-platform
   - git checkout -b feature/hetgnn-aggregator

2. ADR
   - Create ADR-032-heterogeneous-gnn-aggregator.md
   - Include: context, decisions, architecture, network loaders, interpretability
   - Reference ADR-012 as primary pattern

3. IMPLEMENTATION
   - gfm_tasks/gfm_tasks/heterogeneous_gnn.py (HeterogeneousGNNAggregator)
   - gfm_data/gfm_data/network_loaders.py (GeneInteractionNetworkLoader)
   - gfm_eval/gfm_eval/diagnostics/gnn_interpretation.py (GNNInterpreter)

4. TESTING
   - Unit tests for all new modules
   - Integration test: load network → build aggregator → fit → transform

5. DOCUMENTATION
   - configs/tasks/hetgnn_default.yaml
   - examples/04-tasks/hetgnn_rare_disease.ipynb

6. REVIEW
   - Run design-reviewer agent against ADR
   - Run research-code-reviewer agent on implementation
   - Fix any issues

7. COMMIT
   - Commit with descriptive message
   - Ready for PR review

Context files:
- /root/pre-phd-genomics/docs/planning/projects/project3_implementation_plan.md
- /root/pre-phd-genomics/docs/planning/projects/project3_platform_integration_plan.md
- /root/genomic-foundation-platform/docs/ADR/ADR-012-task-layer.md
- /root/genomic-foundation-platform/gfm_tasks/gfm_tasks/base.py
```

---

## 8. Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| Week 1 | Setup + ADR + Core aggregator | Branch, ADR-032, heterogeneous_gnn.py |
| Week 2 | Network loaders + Interpretability | network_loaders.py, gnn_interpretation.py |
| Week 3 | Testing + Documentation | All tests, configs, example notebook |
| Week 4 | Review + Polish + Merge | Agent reviews, fixes, PR merged |

**Total estimated:** 4 weeks (~80-100 hours)

---

## 9. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| PyTorch Geometric compatibility | Medium | Test early; fallback to manual GNN impl |
| Network data access issues | Medium | Start with local cached networks |
| Integration with existing Trainer | Low | Follow ADR-012 exactly |
| Performance issues | Medium | Profile early; optimize message passing |

---

## 10. Next Steps

1. **Immediate:** Save this plan, wait for other process to complete
2. **Then:** Feed agent prompt (Section 7) to execution agent
3. **Monitor:** Check progress, review ADR, validate code
4. **Iterate:** Address review feedback, refine implementation