import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import os
import glob

# --- Configuration ---
DRUGBANK_CSV_PATH = '/home/zdy/Project2/data/drugbank/drugbank.csv'
PDBBIND_BASE_DIR = '/home/zdy/Project2/data/PDBbind_v2020/'
OUTPUT_CSV_FILENAME = '/home/zdy/Project2/data/data.csv'

# Suppress RDKit console logs for cleaner output
RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smiles_string):
    """
    Converts a SMILES string into its canonical form.
    Returns None if the SMILES is invalid.
    """
    if not smiles_string or pd.isna(smiles_string):
        return None
    mol = Chem.MolFromSmiles(str(smiles_string).strip())
    if mol:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return None


def read_smiles_from_smi_file(smi_file_path):
    """
    Reads the first line of a .smi file and extracts the SMILES string.
    Assumes format: <SMILES> <ID>
    """
    try:
        with open(smi_file_path, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                if parts:
                    return parts[0]
    except Exception as e:
        print(f"  Error: Reading or parsing SMILES file '{smi_file_path}' failed: {e}")
    return None


def process_pdbbind_and_drugbank():
    """
    Main function to find complexes from PDBbind that have a matching drug in DrugBank.
    """
    # 1. Load and process the DrugBank CSV file
    print(f"Reading DrugBank file: {DRUGBANK_CSV_PATH} ...")
    try:
        drugbank_df = pd.read_csv(DRUGBANK_CSV_PATH)
        if 'smiles' not in drugbank_df.columns or 'name' not in drugbank_df.columns:
            print(f"Error: DrugBank CSV file '{DRUGBANK_CSV_PATH}' is missing 'smiles' or 'name' columns.")
            return

        drugbank_smiles_map = {}  # {canonical_smiles: drug_name}
        invalid_drugbank_smiles_count = 0

        for idx, row in drugbank_df.iterrows():
            original_smi = row['smiles']
            drug_name = str(row['name']).strip() if pd.notna(row['name']) else "UnknownDrug"

            if pd.notna(original_smi) and str(original_smi).strip():
                cano_smi = canonicalize_smiles(original_smi)
                if cano_smi:
                    if cano_smi not in drugbank_smiles_map:
                        drugbank_smiles_map[cano_smi] = drug_name
                else:
                    print(f"  Warning: Unable to canonicalize SMILES from DrugBank CSV (row {idx + 2}): '{original_smi}'")
                    invalid_drugbank_smiles_count += 1

        if not drugbank_smiles_map:
            print("Error: Failed to load or canonicalize any valid SMILES from DrugBank CSV.")
            return
        print(f"Loaded {len(drugbank_smiles_map)} canonical SMILES and drug names from DrugBank CSV.")
        if invalid_drugbank_smiles_count > 0:
            print(f"  Note: {invalid_drugbank_smiles_count} SMILES strings were skipped due to canonicalization failure.")

    except FileNotFoundError:
        print(f"Error: DrugBank CSV file '{DRUGBANK_CSV_PATH}' not found.")
        return
    except Exception as e:
        print(f"Error: Failed to read or process DrugBank CSV file '{DRUGBANK_CSV_PATH}': {e}")
        return

    # 2. Iterate through the PDBbind directory, find ligand SMILES, and compare
    print(f"\nScanning PDBbind directory: {PDBBIND_BASE_DIR} ...")
    matched_results = []
    pdb_dirs_scanned = 0
    smi_files_processed = 0

    if not os.path.isdir(PDBBIND_BASE_DIR):
        print(f"Error: PDBbind base directory '{PDBBIND_BASE_DIR}' does not exist or is not a directory.")
        return

    # Iterate through the first level of subdirectories
    for pdb_id_folder_name in os.listdir(PDBBIND_BASE_DIR):
        pdb_id_folder_path = os.path.join(PDBBIND_BASE_DIR, pdb_id_folder_name)
        if os.path.isdir(pdb_id_folder_path) and len(pdb_id_folder_name) == 4:
            pdb_dirs_scanned += 1
            if pdb_dirs_scanned % 500 == 0:
                print(f"  Scanned {pdb_dirs_scanned} PDB ID directories...")

            # Find subfolders ending with "_prot"
            for item_in_pdb_id_folder in os.listdir(pdb_id_folder_path):
                if item_in_pdb_id_folder.endswith("_prot"):
                    prot_folder_path = os.path.join(pdb_id_folder_path, item_in_pdb_id_folder)
                    if os.path.isdir(prot_folder_path):
                        # Find all .smi files within the "*_prot" folder
                        smi_file_paths = glob.glob(os.path.join(prot_folder_path, "*.smi"))
                        for smi_file_path in smi_file_paths:
                            smi_files_processed += 1
                            raw_pdbbind_smi = read_smiles_from_smi_file(smi_file_path)
                            if raw_pdbbind_smi:
                                pdbbind_cano_smi = canonicalize_smiles(raw_pdbbind_smi)
                                if pdbbind_cano_smi and pdbbind_cano_smi in drugbank_smiles_map:
                                    drug_name_match = drugbank_smiles_map[pdbbind_cano_smi]
                                    print(
                                        f"  Match found! PDB ID: {pdb_id_folder_name.upper()}, SMILES: {pdbbind_cano_smi}, DrugBank Name: {drug_name_match}")
                                    matched_results.append({
                                        'pdb_id': pdb_id_folder_name.upper(),
                                        'smiles': pdbbind_cano_smi,
                                        'drug_name': drug_name_match
                                    })

    print(f"\nScanned a total of {pdb_dirs_scanned} PDB ID directories, processed {smi_files_processed} .smi files.")
    print(f"Found {len(matched_results)} matching entries.")

    # 3. Output the results to a CSV file
    if matched_results:
        output_df = pd.DataFrame(matched_results)
        output_df.drop_duplicates(subset=['pdb_id', 'smiles', 'drug_name'], inplace=True)
        output_df.sort_values(by=['pdb_id', 'drug_name'], inplace=True)
        output_df = output_df[['pdb_id', 'smiles', 'drug_name']]

        # Define the output path relative to the input CSV directory
        output_dir = os.path.dirname(os.path.abspath(DRUGBANK_CSV_PATH))
        output_csv_path = os.path.join(output_dir, os.path.basename(OUTPUT_CSV_FILENAME)) # Use basename for filename
        try:
            output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"\nMatched results successfully written to: {output_csv_path}")
        except Exception as e:
            print(f"\nError: Could not write to the output CSV file '{output_csv_path}': {e}")
    else:
        print("\nNo matching complexes were found. No output CSV file was generated.")


if __name__ == '__main__':
    process_pdbbind_and_drugbank()