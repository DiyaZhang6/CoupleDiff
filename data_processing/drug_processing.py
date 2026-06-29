# drug_processing.py
# Read each drug's SMILES from a CSV file and convert them into 3D structures using RDKit

import os
import csv
from rdkit import Chem
from rdkit.Chem import AllChem

# current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# generate relative directory
INPUT_CSV = os.path.join(BASE_DIR, '..', 'data', 'drug', 'SMILES', 'smiles.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'drug', 'SDF')

# make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open and read the CSV file
with open(INPUT_CSV, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        drug_id = row['drug_id']
        smiles = row['smiles']

        # Create a molecule object from the SMILES
        mol = Chem.MolFromSmiles(smiles)
        # Add hydrogens
        mol = Chem.AddHs(mol)
        # Embed 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        # Optimize the geometry
        AllChem.UFFOptimizeMolecule(mol)

        # Define output path for the SDF file
        output_path = os.path.join(OUTPUT_DIR, f"{drug_id}.sdf")
        # Write the 3D molecule to an SDF file
        writer = Chem.SDWriter(output_path)
        writer.write(mol)
        writer.close()

        print(f"Processed {drug_id}: saved 3D structure to {output_path}")
