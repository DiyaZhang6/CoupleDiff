import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import os

# --- Configuration Section ---
SDF_FILE_PATH = '/home/zdy/Project2/data/drugbank/structures.sdf'
OUTPUT_CSV_FILENAME = 'drugbank.csv'  # Output filename

# Suppress RDKit's verbose warnings, keeping only errors
RDLogger.DisableLog('rdApp.*')


def parse_drugbank_sdf_to_csv():
    """
    Parses the DrugBank SDF file to extract the generic name and SMILES
    for all unique drug entries.
    """
    print(f"Starting to parse SDF file: {SDF_FILE_PATH} ...")

    if not os.path.exists(SDF_FILE_PATH):
        print(f"Error: SDF file not found at path '{SDF_FILE_PATH}'")
        return

    suppl = Chem.SDMolSupplier(SDF_FILE_PATH, removeHs=False, sanitize=True)

    all_drugs_data = []
    processed_drug_signatures = set()
    molecules_processed = 0
    drugs_extracted = 0

    for mol_idx, mol in enumerate(suppl):
        molecules_processed += 1
        if mol is None:
            print(f"  Warning: Could not parse molecule at index {mol_idx + 1} in the SDF file. Skipping.")
            continue

        try:
            generic_name = "N/A"
            smiles_string = "N/A"
            canonical_smiles_for_check = None  # To be used for uniqueness check

            # Extract the generic name
            if mol.HasProp('GENERIC_NAME'):
                generic_name_prop = mol.GetProp('GENERIC_NAME')
                if generic_name_prop and str(generic_name_prop).strip():
                    generic_name = str(generic_name_prop).strip()
            elif mol.HasProp('_Name'):  # Fallback to the SDF title line if GENERIC_NAME is absent
                mol_name_prop = mol.GetProp('_Name')
                if mol_name_prop and str(mol_name_prop).strip():
                    generic_name = str(mol_name_prop).strip()

            # Extract or generate SMILES, and attempt to canonicalize it
            if mol.HasProp('SMILES'):
                smiles_prop = mol.GetProp('SMILES')
                if smiles_prop and str(smiles_prop).strip():
                    smiles_string = str(smiles_prop).strip()
                    try:
                        # Attempt to canonicalize the SMILES for the uniqueness check
                        mol_from_smiles = Chem.MolFromSmiles(smiles_string)
                        if mol_from_smiles:
                            canonical_smiles_for_check = Chem.MolToSmiles(mol_from_smiles, isomericSmiles=True,
                                                                          canonical=True)
                        else:
                            print(
                                f"  Warning: The SMILES string '{smiles_string}' for molecule {mol_idx + 1} (Name: {generic_name}) could not be parsed by RDKit.")
                            smiles_string = "INVALID_SMILES_FIELD"
                    except Exception as e_cano:
                        print(
                            f"  Warning: Error canonicalizing SMILES '{smiles_string}' for molecule {mol_idx + 1} (Name: {generic_name}): {e_cano}")
                        smiles_string = "CANONICALIZATION_ERROR"
            else:  # If the SMILES field does not exist, try to generate it from the structure
                try:
                    generated_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)  # Already canonical
                    smiles_string = generated_smiles
                    canonical_smiles_for_check = smiles_string  # Use the generated canonical SMILES directly
                except Exception as e_smiles:
                    print(
                        f"    Error: Could not generate SMILES for molecule {mol_idx + 1} (Name: {generic_name}): {e_smiles}")
                    smiles_string = "GENERATION_ERROR"

            # Build a unique signature (lowercase name + canonical SMILES)
            current_drug_signature = None
            if canonical_smiles_for_check:  # Prioritize canonical SMILES for uniqueness
                current_drug_signature = (generic_name.lower().strip(), canonical_smiles_for_check)
            elif smiles_string not in ["N/A", "GENERATION_ERROR", "INVALID_SMILES_FIELD", "CANONICALIZATION_ERROR"]:
                # If no canonical SMILES, use the original (but seemingly valid) SMILES
                current_drug_signature = (generic_name.lower().strip(), smiles_string)

            # Add the drug data if its signature has not been processed yet
            if current_drug_signature and current_drug_signature not in processed_drug_signatures:
                all_drugs_data.append(
                    {'name': generic_name, 'smiles': smiles_string})  # Store the original or generated SMILES
                processed_drug_signatures.add(current_drug_signature)
                drugs_extracted += 1

        except Exception as e:
            print(f"  Error: An unexpected exception occurred while processing molecule {mol_idx + 1}: {e}")

        if molecules_processed % 500 == 0:
            print(f"  ...processed {molecules_processed} molecules.")

    print(f"\nSDF file parsing complete. Total molecules processed: {molecules_processed}.")
    print(f"Extracted {drugs_extracted} unique drug name and SMILES pairs.")

    # Save the results to a CSV file
    if all_drugs_data:
        df_output = pd.DataFrame(all_drugs_data)

        output_dir = os.path.dirname(os.path.abspath(SDF_FILE_PATH))
        output_csv_path = os.path.join(output_dir, OUTPUT_CSV_FILENAME)

        try:
            df_output.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"\nResults successfully written to: {output_csv_path}")
        except Exception as e:
            print(f"\nError: Could not write to the output CSV file '{output_csv_path}': {e}")
    else:
        print("\nNo drug data was extracted. No CSV file was generated.")


if __name__ == '__main__':
    parse_drugbank_sdf_to_csv()