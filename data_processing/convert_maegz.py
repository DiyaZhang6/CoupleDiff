# /home/zdy/Project2/data_processing/convert_maegz.py

import os
import argparse
from pathlib import Path
from tqdm import tqdm

from schrodinger.structure import StructureReader
from schrodinger.structutils import analyze, structconvert


def convert_maegz_to_pdb_sdf(maegz_file: Path, output_dir: Path):
    """
    Reads a .maegz complex file, splits it into protein and ligand,
    and saves them as PDB and SDF files respectively.
    """
    try:
        st = next(StructureReader(str(maegz_file)))

        pdb_id = maegz_file.stem.split('_')[0]

        pdb_output_dir = output_dir / pdb_id
        pdb_output_dir.mkdir(parents=True, exist_ok=True)

        # 分离蛋白质和配体
        # analyze.find_ligands() 返回一个列表，我们取第一个配体
        ligands = analyze.find_ligands(st)
        if not ligands:
            print(f"WARNING: No ligand found in {maegz_file.name}. Skipping.")
            return

        ligand = ligands[0]  # 假设只有一个配体
        protein_asl = f"not (res.pt AST '{ligand.pdbres.strip()}_{ligand.chain.strip()}_{ligand.resnum}')"
        protein = st.extract(analyze.evaluate_asl(st, protein_asl))
        ligand_st = st.extract(ligand.atom_indices)

        # 定义输出文件名
        protein_out_path = pdb_output_dir / f"{pdb_id}_protein.pdb"
        ligand_out_path = pdb_output_dir / f"{pdb_id}_ligand.sdf"

        # 使用 structconvert 进行转换和保存
        structconvert.write_structure(protein, str(protein_out_path), "pdb")
        structconvert.write_structure(ligand_st, str(ligand_out_path), "sdf")

        # print(f"Successfully converted {maegz_file.name} for PDB ID {pdb_id}")
        return True

    except Exception as e:
        print(f"ERROR: Failed to process {maegz_file.name}. Reason: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert Schrodinger .maegz complex files to PDB/SDF pairs.")
    parser.add_argument("-i", "--input_dir", required=True, type=str,
                        help="Directory containing the .maegz files (can be nested).")
    parser.add_argument("-o", "--output_dir", required=True, type=str,
                        help="Directory where the output PDB/SDF pairs will be saved.")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Searching for .maegz files in: {input_path}")
    maegz_files = list(input_path.glob("**/*_complex.maegz"))
    print(f"Found {len(maegz_files)} .maegz files to process.")

    for maegz_file in tqdm(maegz_files, desc="Converting complexes"):
        convert_maegz_to_pdb_sdf(maegz_file, output_path)

    print("\nConversion process complete.")


if __name__ == "__main__":
    main()