# analyze_learnable_set.py
# A script to perform a detailed statistical analysis of a combined
# training and validation set. It reads PDB IDs from both train.csv and valid.csv,
# combines them, and uses the RCSB PDB GraphQL API to fetch metadata for analysis.

import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Set

# Third-party libraries - required
try:
    import requests
    from tqdm import tqdm
    import pandas as pd
except ImportError:
    print("Error: 'requests', 'tqdm', and 'pandas' libraries are required.")
    print("Please install them using: pip install requests tqdm pandas")
    exit(1)

TRAINING_SET_CSV = Path("/home/zdy/Project2/data/train.csv")
VALIDATION_SET_CSV = Path("/home/zdy/Project2/data/valid.csv")

# Number of PDB IDs to query in a single API call
BATCH_SIZE = 150
# Official RCSB PDB GraphQL API endpoint.
PDB_API_URL = "https://data.rcsb.org/graphql"


def load_pdb_ids_from_csvs(filepaths: List[Path]) -> Set[str]:
    """
    Loads PDB IDs from the 'pdb_id' column of multiple CSV files and returns a unique set.
    """
    all_ids = set()
    for filepath in filepaths:
        if not filepath.exists():
            print(f"Warning: CSV file not found at '{filepath}'. Skipping this file.")
            continue

        try:
            df = pd.read_csv(filepath)
            if 'pdb_id' not in df.columns:
                print(f"Warning: The CSV file '{filepath}' must contain a column named 'pdb_id'. Skipping.")
                continue

            # Read IDs, strip whitespace, and convert to uppercase for consistency.
            ids_from_file = set(df['pdb_id'].astype(str).str.strip().str.upper())
            print(f"Loaded {len(ids_from_file)} PDB IDs from '{filepath.name}'.")
            all_ids.update(ids_from_file)
        except Exception as e:
            print(f"Error reading or processing CSV file '{filepath}': {e}")

    return all_ids


def fetch_pdb_data_in_batches(pdb_ids: Set[str]) -> List[Dict[str, Any]]:
    """
    Fetches required metadata for a set of PDB IDs from the RCSB GraphQL API.
    The function queries in batches to avoid overwhelming the API.
    """

    # fetch all necessary information in one go
    graphql_query = """
    query getPdbData($entry_ids: [String!]!) {
      entries(entry_ids: $entry_ids) {
        rcsb_id
        struct_keywords {
          pdbx_keywords
        }
        polymer_entities {
          rcsb_polymer_entity_container_identifiers {
            uniprot_ids
          }
          entity_poly {
            rcsb_entity_polymer_type
          }
        }
        nonpolymer_entities {
          nonpolymer_comp {
            rcsb_id
          }
        }
      }
    }
    """

    all_results = []
    unique_ids = sorted(list(pdb_ids))
    print(f"\nQuerying metadata for {len(unique_ids)} unique PDB IDs from RCSB PDB...")

    # Loop through the unique IDs in chunks of BATCH_SIZE.
    for i in tqdm(range(0, len(unique_ids), BATCH_SIZE), desc="Fetching PDB Data"):
        batch_ids = unique_ids[i:i + BATCH_SIZE]
        variables = {"entry_ids": batch_ids}

        try:
            response = requests.post(PDB_API_URL, json={"query": graphql_query, "variables": variables})
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()
            if "data" in data and data["data"].get("entries"):
                valid_entries = [entry for entry in data["data"]["entries"] if entry is not None]
                all_results.extend(valid_entries)
            else:
                print(f"Warning: Unexpected API response for batch starting with {batch_ids[0]}: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Error during API call for batch starting with {batch_ids[0]}: {e}")

        time.sleep(0.1)

    return all_results


def analyze_dataset(all_pdb_data: List[Dict[str, Any]], all_pdb_ids_in_set: Set[str]):
    """

    Performs the three requested analyses on the fetched data and prints a summary.
    """

    protein_to_ligands = defaultdict(set)
    protein_to_structures = defaultdict(int)
    family_keywords = []

    # --- Step 1: Pre-process fetched data to map PDB IDs to a canonical protein target ID (UniProt) ---
    pdb_to_uniprot = {}
    for entry in all_pdb_data:
        if not entry or not entry.get("polymer_entities"):
            continue

        uniprot_id = None
        for entity in entry["polymer_entities"]:
            if entity.get("entity_poly", {}).get("rcsb_entity_polymer_type") == "Protein":
                uniprot_ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids")
                if uniprot_ids:
                    uniprot_id = uniprot_ids[0]
                    break

        if uniprot_id:
            pdb_to_uniprot[entry["rcsb_id"]] = uniprot_id
            if entry.get("struct_keywords") and entry["struct_keywords"].get("pdbx_keywords"):
                family_keywords.append(entry["struct_keywords"]["pdbx_keywords"])

    # --- Step 2: Perform analysis by iterating through the combined list of learnable set complexes ---
    for pdb_id in all_pdb_ids_in_set:
        uniprot_id = pdb_to_uniprot.get(pdb_id)
        if not uniprot_id:
            continue

        protein_to_structures[uniprot_id] += 1

        entry_data = next((item for item in all_pdb_data if item and item["rcsb_id"] == pdb_id), None)

        if entry_data and entry_data.get("nonpolymer_entities"):
            for entity in entry_data["nonpolymer_entities"]:
                ligand_id = entity.get("nonpolymer_comp", {}).get("rcsb_id")
                if ligand_id and ligand_id not in ["HOH"]:
                    protein_to_ligands[uniprot_id].add(ligand_id)

    # --- Step 3: Print the formatted results ---
    print("\n" + "=" * 80)
    print(" " * 20 + "COMBINED TRAINING & VALIDATION DATASET STATISTICS")
    print("=" * 80)

    print(f"Total Unique Complexes in Learnable Set (Train + Valid): {len(all_pdb_ids_in_set)}")

    print("\n--- 1. Distribution of Unique Ligands per Protein Target ---")
    ligand_counts = [len(ligands) for ligands in protein_to_ligands.values()]
    if ligand_counts:
        print(f"Total protein targets with ligand data: {len(ligand_counts)}")
        print(f"Mean number of unique ligands per protein: {sum(ligand_counts) / len(ligand_counts):.2f}")

        ligand_counts.sort()
        median_index = len(ligand_counts) // 2
        median_val = (ligand_counts[median_index] + ligand_counts[~median_index]) / 2 if len(ligand_counts) > 0 else 0

        print(f"Median number of unique ligands per protein: {median_val}")
        print(f"Proteins with >5 unique ligands: {sum(1 for c in ligand_counts if c > 5)}")
        print(f"Proteins with >10 unique ligands: {sum(1 for c in ligand_counts if c > 10)}")
        print(f"Proteins with >20 unique ligands: {sum(1 for c in ligand_counts if c > 20)}")
    else:
        print("No ligand data could be analyzed.")

    print("\n--- 2. Distribution of Co-crystal Structures per Protein Target ---")
    structure_counts = list(protein_to_structures.values())
    if structure_counts:
        print(f"Total unique protein targets in dataset: {len(structure_counts)}")
        print(f"Proteins with >5 co-crystal structures: {sum(1 for c in structure_counts if c > 5)}")
        print(f"Proteins with >10 co-crystal structures: {sum(1 for c in structure_counts if c > 10)}")
        print(f"Proteins with >20 co-crystal structures: {sum(1 for c in structure_counts if c > 20)}")
    else:
        print("No structure count data could be analyzed.")

    print("\n--- 3. Breakdown of Protein Families Represented ---")
    if family_keywords:
        keyword_counts = Counter(k.upper() for k in family_keywords if k)
        print("Top 10 most common PDB keywords (as a proxy for families):")
        total_keywords = len(family_keywords)
        for keyword, count in keyword_counts.most_common(10):
            percentage = (count / total_keywords) * 100
            print(f"- {keyword:<20}: {count} occurrences ({percentage:.1f}%)")
    else:
        print("No family keyword data could be analyzed.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Step 1: Load PDB IDs from BOTH the training and validation CSV files.
    learnable_pdb_ids = load_pdb_ids_from_csvs([TRAINING_SET_CSV, VALIDATION_SET_CSV])

    if learnable_pdb_ids:
        # Step 2: Fetch data from the API for the combined set of IDs.
        fetched_data = fetch_pdb_data_in_batches(learnable_pdb_ids)
        if fetched_data:
            # Step 3: Run the analysis and print the results.
            analyze_dataset(fetched_data, learnable_pdb_ids)
        else:
            print("Failed to fetch any data from the PDB API. Aborting analysis.")
    else:
        print("No PDB IDs were loaded from the CSV files. Aborting analysis.")