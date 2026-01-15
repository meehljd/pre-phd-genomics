"""
Patient Graph Builder for Heterogeneous GNN

Constructs per-patient heterogeneous graphs from:
- Variant genes (from VCF)
- Network databases (STRING, Reactome, ENCODE, GTEx)
- Node features (pathogenicity, expression, conservation, constraint)

Output: PyTorch Geometric HeteroData objects
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


@dataclass
class NetworkDatabases:
    """Container for pre-loaded network databases."""

    ppi: pd.DataFrame  # STRING: gene1, gene2, score
    regulatory: pd.DataFrame  # ENCODE: tf, target
    pathway: pd.DataFrame  # Reactome: gene, pathway_id
    coexpression: pd.DataFrame  # GTEx: gene1, gene2, correlation


@dataclass
class PatientData:
    """Container for patient-level data."""

    patient_id: str
    variant_genes: list[str]  # Genes with variants
    hpo_terms: list[str]  # Phenotype terms
    ancestry_pcs: Optional[np.ndarray] = None  # PC1-PC10
    diagnosis: Optional[str] = None  # Ground truth label


class PatientGraphBuilder:
    """
    Builds heterogeneous patient graphs for rare disease diagnosis.

    Each patient is represented as a graph where:
    - Nodes: genes (variant genes + network neighbors)
    - Edges: different relationship types (PPI, regulatory, pathway, co-expression)
    - Node features: pathogenicity scores, expression, conservation, etc.
    """

    def __init__(
        self,
        networks: NetworkDatabases,
        k_hops: int = 2,
        max_nodes: int = 100,
        min_nodes: int = 10,
    ):
        """
        Initialize the graph builder.

        Args:
            networks: Pre-loaded network databases
            k_hops: Number of hops to expand from variant genes
            max_nodes: Maximum genes per patient graph
            min_nodes: Minimum genes required
        """
        self.networks = networks
        self.k_hops = k_hops
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes

        # Build combined network for neighbor expansion
        self._build_combined_network()

    def _build_combined_network(self) -> None:
        """Build a NetworkX graph combining all edge types for neighbor expansion."""
        self.combined_network = nx.Graph()

        # Add PPI edges
        for _, row in self.networks.ppi.iterrows():
            self.combined_network.add_edge(row["gene1"], row["gene2"], type="ppi")

        # Add regulatory edges
        for _, row in self.networks.regulatory.iterrows():
            self.combined_network.add_edge(row["tf"], row["target"], type="regulatory")

        # Add co-expression edges
        for _, row in self.networks.coexpression.iterrows():
            self.combined_network.add_edge(row["gene1"], row["gene2"], type="coexpression")

        print(f"Combined network: {self.combined_network.number_of_nodes()} nodes, "
              f"{self.combined_network.number_of_edges()} edges")

    def _expand_neighbors(self, seed_genes: list[str]) -> set[str]:
        """
        Expand seed genes to k-hop neighbors.

        Args:
            seed_genes: Starting genes (variant genes)

        Returns:
            Set of genes including seeds and neighbors
        """
        expanded = set(seed_genes)

        for _ in range(self.k_hops):
            new_neighbors = set()
            for gene in expanded:
                if gene in self.combined_network:
                    new_neighbors.update(self.combined_network.neighbors(gene))
            expanded.update(new_neighbors)

        # Limit to max_nodes, prioritizing by centrality
        if len(expanded) > self.max_nodes:
            # Compute degree centrality for prioritization
            subgraph = self.combined_network.subgraph(expanded)
            centrality = nx.degree_centrality(subgraph)

            # Always keep seed genes, then add by centrality
            sorted_genes = sorted(
                expanded - set(seed_genes),
                key=lambda g: centrality.get(g, 0),
                reverse=True
            )
            expanded = set(seed_genes) | set(sorted_genes[: self.max_nodes - len(seed_genes)])

        return expanded

    def _extract_edges(self, genes: set[str]) -> dict[str, torch.Tensor]:
        """
        Extract edges of each type for the given gene set.

        Args:
            genes: Set of genes in the patient graph

        Returns:
            Dictionary mapping edge type to edge index tensor
        """
        gene_to_idx = {g: i for i, g in enumerate(sorted(genes))}
        edge_dict = {}

        # PPI edges
        ppi_edges = []
        for _, row in self.networks.ppi.iterrows():
            if row["gene1"] in genes and row["gene2"] in genes:
                ppi_edges.append([gene_to_idx[row["gene1"]], gene_to_idx[row["gene2"]]])
                ppi_edges.append([gene_to_idx[row["gene2"]], gene_to_idx[row["gene1"]]])  # Undirected
        if ppi_edges:
            edge_dict["ppi"] = torch.tensor(ppi_edges, dtype=torch.long).t().contiguous()

        # Regulatory edges (directed: TF -> target)
        reg_edges = []
        for _, row in self.networks.regulatory.iterrows():
            if row["tf"] in genes and row["target"] in genes:
                reg_edges.append([gene_to_idx[row["tf"]], gene_to_idx[row["target"]]])
        if reg_edges:
            edge_dict["regulatory"] = torch.tensor(reg_edges, dtype=torch.long).t().contiguous()

        # Co-expression edges
        coexpr_edges = []
        for _, row in self.networks.coexpression.iterrows():
            if row["gene1"] in genes and row["gene2"] in genes:
                coexpr_edges.append([gene_to_idx[row["gene1"]], gene_to_idx[row["gene2"]]])
                coexpr_edges.append([gene_to_idx[row["gene2"]], gene_to_idx[row["gene1"]]])
        if coexpr_edges:
            edge_dict["coexpression"] = torch.tensor(coexpr_edges, dtype=torch.long).t().contiguous()

        return edge_dict

    def build_graph(
        self,
        patient: PatientData,
        node_features: Optional[dict[str, np.ndarray]] = None,
    ) -> Optional[HeteroData]:
        """
        Build a heterogeneous graph for a single patient.

        Args:
            patient: Patient data container
            node_features: Pre-computed node features {gene: feature_vector}

        Returns:
            PyTorch Geometric HeteroData object, or None if insufficient data
        """
        # Filter to genes in our network
        seed_genes = [g for g in patient.variant_genes if g in self.combined_network]

        if len(seed_genes) < 1:
            print(f"Patient {patient.patient_id}: No variant genes in network")
            return None

        # Expand to neighbors
        all_genes = self._expand_neighbors(seed_genes)

        if len(all_genes) < self.min_nodes:
            print(f"Patient {patient.patient_id}: Too few genes ({len(all_genes)})")
            return None

        # Extract edges
        edge_dict = self._extract_edges(all_genes)

        if not edge_dict:
            print(f"Patient {patient.patient_id}: No edges found")
            return None

        # Build HeteroData
        data = HeteroData()

        # Sort genes for consistent indexing
        sorted_genes = sorted(all_genes)
        gene_to_idx = {g: i for i, g in enumerate(sorted_genes)}

        # Node features
        if node_features:
            features = []
            for gene in sorted_genes:
                if gene in node_features:
                    features.append(node_features[gene])
                else:
                    # Use zero vector for genes without features
                    features.append(np.zeros_like(list(node_features.values())[0]))
            data["gene"].x = torch.tensor(np.array(features), dtype=torch.float32)
        else:
            # Placeholder: one-hot encoding of variant status
            features = np.zeros((len(sorted_genes), 1))
            for gene in seed_genes:
                features[gene_to_idx[gene], 0] = 1.0
            data["gene"].x = torch.tensor(features, dtype=torch.float32)

        # Add edges
        for edge_type, edge_index in edge_dict.items():
            data["gene", edge_type, "gene"].edge_index = edge_index

        # Store metadata
        data.patient_id = patient.patient_id
        data.gene_names = sorted_genes
        data.variant_genes = seed_genes
        data.num_nodes = len(sorted_genes)

        # Label (if available)
        if patient.diagnosis:
            data.y = patient.diagnosis

        # Ancestry (if available)
        if patient.ancestry_pcs is not None:
            data.ancestry_pcs = torch.tensor(patient.ancestry_pcs, dtype=torch.float32)

        return data

    def build_dataset(
        self,
        patients: list[PatientData],
        node_features: Optional[dict[str, np.ndarray]] = None,
        output_path: Optional[Path] = None,
    ) -> list[HeteroData]:
        """
        Build graphs for multiple patients.

        Args:
            patients: List of patient data containers
            node_features: Pre-computed node features
            output_path: Optional path to save the dataset

        Returns:
            List of HeteroData objects
        """
        graphs = []
        for i, patient in enumerate(patients):
            if i % 50 == 0:
                print(f"Processing patient {i + 1}/{len(patients)}")

            graph = self.build_graph(patient, node_features)
            if graph is not None:
                graphs.append(graph)

        print(f"Built {len(graphs)} graphs from {len(patients)} patients")

        if output_path:
            torch.save(graphs, output_path)
            print(f"Saved to {output_path}")

        return graphs


def main():
    """Example usage and testing."""
    # This would be called with actual data
    print("PatientGraphBuilder ready for use")
    print("See notebooks/02_graph_construction.ipynb for usage examples")


if __name__ == "__main__":
    main()
