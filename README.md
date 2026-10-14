# DynaPhArM

DynaPhArM is a research codebase for protein-ligand complex modeling. It builds
graph representations for protein backbones, side chains, and ligands, combines
them with physics/geometry labels, and trains a neural model to predict complex
coordinates with physically informed losses.

The current version supports an atom-level protein backbone representation
(`pipeline_tasks.backbone_graph_task.node_level: "atom"`), so predicted test-set
structures can be exported as PDB files containing backbone atoms, side-chain
atoms, and ligand atoms.

## Features

- Protein and ligand preprocessing for docking-oriented datasets.
- Backbone, side-chain, and ligand graph construction.
- Lennard-Jones matrix generation for backbone-sidechain, backbone-ligand, and
  sidechain-ligand interactions.
- Physics/geometry label generation, including bonds, angles, dihedrals,
  van der Waals terms, electrostatics, hydrogen bonds, and pi-pi interactions.
- A PyTorch/PyTorch Geometric model with separate encoders for backbone,
  sidechain, and ligand nodes.
- Physics-constrained cross-attention and an equivariant coordinate decoder.
- Checkpoint evaluation and test-set PDB export utilities.

## Repository Layout

```text
.
├── config.yaml                  # Main project configuration
├── data_processing/             # Preprocessing and graph/label generation
│   ├── split.py                 # Split complexes into backbone/sidechain/ligand
│   ├── LJ_generate.py           # Generate LJ interaction matrices
│   ├── phy_geo.py               # Generate physics/geometry labels
│   ├── backbone_graph.py        # Generate backbone graphs
│   ├── sidechain_graph.py       # Generate side-chain graphs
│   ├── drug_graph.py            # Generate ligand graphs
│   └── combine.py               # Merge graphs and labels into .pt payloads
├── models/                      # Model, encoder, interaction, decoder, losses
├── scripts/
│   ├── data.py                  # Dataset and PyG dataloader
│   ├── train.py                 # Training entry point
│   ├── evaluate_checkpoint.py   # Evaluate a checkpoint on train/val/test
│   └── predict_testset_pdb.py   # Export predicted complexes as PDB files
└── case/                        # Case-study, plotting, and analysis scripts
```

## Requirements

This project is intended for a Linux GPU environment. The original development
environment used a Conda environment named `FlexPose`.

Core Python dependencies include:

- Python 3.8+
- PyTorch
- PyTorch Geometric
- torch-scatter / torch-cluster
- RDKit
- BioPython
- NumPy, pandas, PyYAML, tqdm

External tools may be required for the full preprocessing pipeline:

- OpenBabel
- AutoDock Vina
- MGLTools
- Schrodinger tools, for workflows that use `.maegz` conversion or related
  structure preparation

Install the Python packages according to your CUDA and PyTorch version. For
example, create your environment first, then install the PyTorch Geometric
packages from the matching official wheel index.

## Configuration

Most paths and hyperparameters live in `config.yaml`.

Important fields:

```yaml
project_base_dir: "/home/zdy/Project2"

pipeline_tasks:
  backbone_graph_task:
    node_level: "atom"
    overwrite_existing: true

model_params:
  in_scalar_dim_bb: 45

data_loading:
  combined_dir: "data/processed_data/combined_pt"
  physical_combined_dir: "/workspace/guest/zdy/Project2/data/processed_data/combined_pt"
  max_total_nodes: 10000
