# CoupleDiff

CoupleDiff is a research codebase for protein-ligand complex modeling. It builds
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
├── bin/
│   └── vina_1.2.5_linux_x86_64  # AutoDock Vina executable
├── cdhit/                       # CD-HIT related files or output
├── config.yaml                  # Main project configuration
├── requirements.txt             # Python dependency list, if maintained
├── mmcif_pdbx.dic               # mmCIF dictionary used by structure tools
├── data/
│   ├── INDEX_general_PL_data.2020
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   ├── posebusters.csv
│   ├── astex.csv
│   └── structures.sdf
├── data_processing/             # Preprocessing and graph/label generation
│   ├── split.py                 # Split complexes into backbone/sidechain/ligand
│   ├── LJ_generate.py           # Generate LJ interaction matrices
│   ├── phy_geo.py               # Generate physics/geometry labels
│   ├── backbone_graph.py        # Generate backbone graphs
│   ├── sidechain_graph.py       # Generate side-chain graphs
│   ├── drug_graph.py            # Generate ligand graphs
│   └── combine.py               # Merge graphs and labels into .pt payloads
├── mgltools_x86_64Linux2_1.5.7/ # MGLTools installation used for PDBQT prep
├── models/                      # Model, encoder, interaction, decoder, losses
├── scripts/
│   ├── data.py                  # Dataset and PyG dataloader
│   ├── train.py                 # Training entry point
│   ├── test.py                  # Test/evaluation entry point
│   ├── smoke_data.py            # Quick data-loading smoke test
│   ├── calculate_flops.py       # FLOPs estimation
│   └── calculate_params.py      # Parameter counting
└── README.md
```

## Requirements

This project is intended for a Linux GPU environment. The original development
environment used a Conda environment.

Core Python dependencies include:

- Python 3.8+
- PyTorch
- PyTorch Geometric
- torch-scatter / torch-cluster
- RDKit
- BioPython
- NumPy, pandas, PyYAML, tqdm

If `requirements.txt` is up to date for your environment, install it with:

```bash
pip install -r requirements.txt
```

External tools may be required for the full preprocessing pipeline:

- OpenBabel
- AutoDock Vina, such as `bin/vina_1.2.5_linux_x86_64`
- MGLTools, such as `mgltools_x86_64Linux2_1.5.7/`
- CD-HIT, for dataset clustering/splitting workflows
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

training:
  device: "cuda:2"
  batch_size: 1
  epochs: 3
```

Before running the pipeline on a new machine, update `project_base_dir`,
`physical_combined_dir`, GPU IDs, and external tool paths.

## Data Preparation

The repository expects large structure datasets to live outside Git, usually
under:

```text
data/PDBbind_v2020/
data/CASF_v2016/
data/astex_diverse_set/
data/PoseBusters/posebusters_benchmark_set/
```

The typical processed-data layout is:

```text
data/processed_data/
├── split/
├── LJ/
├── label/
├── graph/
└── combined_pt/
```

CSV split files are configured in `config.yaml`:

```text
data/train.csv
data/valid.csv
data/test.csv
data/posebusters.csv
```

## Preprocessing Pipeline

Run commands from the project root:

```bash
cd /home/zdy/Project2
```

Generate graph features and labels:

```bash
python data_processing/backbone_graph.py
python data_processing/sidechain_graph.py
python data_processing/drug_graph.py
python data_processing/LJ_generate.py
python data_processing/phy_geo.py
```

Merge graph and label files into combined training payloads:

```bash
python data_processing/combine.py
```

For long jobs, run in the background:

```bash
nohup python data_processing/combine.py \
  > logs/combine_$(date +%Y%m%d_%H%M%S).out 2>&1 &
```

## Training

Compile-check the main scripts:

```bash
python -m py_compile scripts/train.py scripts/data.py models/model.py models/loss.py
```

Start training:

```bash
python scripts/train.py
```

Run training in the background:

```bash
nohup python scripts/train.py \
  > logs/training/train_$(date +%Y%m%d_%H%M%S).out 2>&1 &
```

Monitor logs:

```bash
tail -f logs/training/training_*.log
```

Checkpoints are written to the directory configured by
`training.checkpoint_dir`.

## Resume Training

Set the checkpoint path in `config.yaml`:

```yaml
training:
  resume_checkpoint: "checkpoints/<checkpoint_name>.pt"
  resume_optimizer: true
  resume_scheduler: true
  resume_scaler: true
```

Then run:

```bash
python scripts/train.py
```

If the graph representation or model dimensions changed, old checkpoints are
not compatible and training should restart from scratch.

## Evaluate a Checkpoint

The repository includes `scripts/test.py` for test-time evaluation:

```bash
python scripts/test.py
```

If `scripts/evaluate_checkpoint.py` is included in your branch, you can also
evaluate a checkpoint with a detailed loss breakdown. For validation data:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/<checkpoint_name>.pt \
  --split val \
  --device cuda:0
```

Evaluate a configured test set:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/<checkpoint_name>.pt \
  --split PDBbind_Core_Set \
  --device cuda:0
```

Useful options:

```bash
--max-batches 100
--batch-size 1
--num-workers 4
--use-amp
```

## Export Test-Set PDB Structures

If `scripts/predict_testset_pdb.py` is included in your branch, generate one
predicted complex PDB per test-set sample with:

```bash
python scripts/predict_testset_pdb.py \
  --checkpoint checkpoints/<checkpoint_name>.pt \
  --split PDBbind_Core_Set \
  --device cuda:0 \
  --output-dir /home/zdy/Project2/results/testset \
  --overwrite
```

Outputs:

```text
/home/zdy/Project2/results/testset/<pdb_id>.pdb
/home/zdy/Project2/results/testset/prediction_summary.csv
```

The exported structure contains the atoms represented by the current graph
payload. With `node_level: "atom"`, backbone atoms are exported from the
backbone graph/template. If hydrogens are not present in the input structures
or `include_hydrogen: false`, hydrogens will not be generated automatically.

## Notes on Large Files

Datasets, generated `.pt` files, checkpoints, logs, and exported structures can
be very large. They should usually be excluded from Git and stored on local
workspace storage or an external artifact store.

Recommended ignored paths include:

```text
data/PDBbind_v2020/
data/CASF_v2016/
data/astex_diverse_set/
data/PoseBusters/
data/processed_data/
cdhit/
checkpoints/
logs/
results/
*.pt
*.pth
*.ckpt
```

## Development Tips

Quick dataset smoke check:

```bash
python scripts/smoke_data.py
```

Count trainable parameters:

```bash
python scripts/calculate_params.py
```

Estimate model FLOPs:

```bash
python scripts/calculate_flops.py
```

If `scripts/evaluate_checkpoint.py` is available, inspect a checkpoint loss
breakdown:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/<checkpoint_name>.pt \
  --split val \
  --max-batches 20
```

When switching between residue-level backbone graphs and atom-level backbone
graphs, regenerate backbone graph files and rerun `data_processing/combine.py`.
The combined `.pt` files must match the current graph schema and model input
dimensions.

## Project Status

This repository is a research prototype. Some scripts are dataset- and
environment-specific, especially preprocessing scripts that depend on external
chemistry software or hard-coded server paths. Review `config.yaml` carefully
before running the full pipeline on another machine.

## License

No license file is included yet. Add a license before distributing or publishing
the project.
This repository is currently released for academic review and communication
only. The manuscript associated with this project is under review, and no
open-source license has been granted yet.

Unless a LICENSE file is added later, all rights are reserved by the authors.
Please do not redistribute, modify, or reuse the code or data for public or
commercial purposes without permission.
