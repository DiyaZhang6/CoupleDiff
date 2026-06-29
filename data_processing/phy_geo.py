# /home/zdy/Project2/data_processing/phy_geo.py
import argparse
import logging
import yaml
import datetime
import numpy as np
import torch
import os
import sys
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial

# --- Third-party imports ---
try:
    from rdkit import Chem
    from rdkit.rdBase import BlockLogs
    from rdkit.Chem import AllChem, rdBase

    rdBase.DisableLog('rdApp.*')
except ImportError:
    print("FATAL: RDKit is required. Please install it in your conda environment.")
    exit(1)

# --- Hardcoded Defaults ---
DEFAULT_CONFIG_PATH = "/home/zdy/Project2/config.yaml"
PROJECT_ROOT = Path("/home/zdy/Project2")

# --- Global Constants ---
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}
_STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL"
}

_ELEMENT_CHARGE_PRIORS = {
    "H": 0.10,
    "C": 0.00,
    "N": -0.30,
    "O": -0.50,
    "S": -0.20,
    "P": 0.30,
    "F": -0.40,
    "CL": -0.20,
    "BR": -0.15,
    "I": -0.10,
    "MG": 0.50,
    "ZN": 0.70,
    "CA": 0.50,
    "NA": 0.50,
    "K": 0.50,
}

_PROTEIN_AROMATIC_RINGS = {
    "PHE": [["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]],
    "TYR": [["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]],
    "HIS": [["CG", "ND1", "CE1", "NE2", "CD2"]],
    "HID": [["CG", "ND1", "CE1", "NE2", "CD2"]],
    "HIE": [["CG", "ND1", "CE1", "NE2", "CD2"]],
    "HIP": [["CG", "ND1", "CE1", "NE2", "CD2"]],
    "TRP": [
        ["CG", "CD1", "NE1", "CE2", "CD2"],
        ["CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"],
    ],
}


def setup_logging_from_config(config: dict, root: Path):
    """Configures logging and returns the log file path."""
    log_cfg = config.get('logging', {}).get('phy_geo_log', {})
    log_dir = root / log_cfg.get('log_dir', 'logs/phy_geo')
    log_dir.mkdir(parents=True, exist_ok=True)

    log_base_name = log_cfg.get('log_base_name', 'phy_geo')
    use_timestamp = log_cfg.get('use_timestamp_in_log_name', True)

    if use_timestamp:
        log_file = log_dir / f"{log_base_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_file = log_dir / f"{log_base_name}.log"

    # Simple console + file config for the Main Process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return log_file


def get_atom_id(atom: Chem.Atom) -> str:
    info = atom.GetPDBResidueInfo()
    if info:
        return f"{info.GetChainId().strip()}_{info.GetResidueNumber()}_{info.GetResidueName().strip()}_{info.GetName().strip()}"
    return f"UNK_{atom.GetIdx()}"


def get_atom_ids_fast(mol):
    return [get_atom_id(a) for a in mol.GetAtoms()]


def _compute_partial_charges(mol: Chem.Mol) -> torch.Tensor:
    num_atoms = mol.GetNumAtoms()
    charges = torch.full((num_atoms,), float("nan"), dtype=torch.float32)

    try:
        mol_for_charge = Chem.Mol(mol)
        Chem.SanitizeMol(mol_for_charge, catchErrors=True)
        AllChem.ComputeGasteigerCharges(mol_for_charge, nIter=12)
        values = []
        for atom in mol_for_charge.GetAtoms():
            try:
                values.append(float(atom.GetProp('_GasteigerCharge')))
            except Exception:
                values.append(float("nan"))
        charges = torch.tensor(values, dtype=torch.float32)
    except Exception:
        pass

    fallback = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol().upper()
        prior = _ELEMENT_CHARGE_PRIORS.get(symbol, 0.0)
        fallback.append(float(atom.GetFormalCharge()) + prior)
    fallback = torch.tensor(fallback, dtype=torch.float32)

    finite = torch.isfinite(charges)
    charges = torch.where(finite, charges, fallback)
    if torch.nan_to_num(charges, nan=0.0).abs().sum() == 0:
        charges = fallback
    return torch.nan_to_num(charges.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _compute_atomic_dipoles(pos: torch.Tensor, charges: torch.Tensor, adj) -> torch.Tensor:
    dipoles = torch.zeros((pos.size(0), 3), dtype=torch.float32)
    for i, neighbors in enumerate(adj):
        if not neighbors:
            continue
        accum = torch.zeros(3, dtype=torch.float32)
        for j in neighbors:
            direction = pos[j] - pos[i]
            norm = torch.linalg.norm(direction).clamp_min(1e-6)
            accum += (charges[j] - charges[i]) * (direction / norm)
        dipoles[i] = accum
    return torch.nan_to_num(dipoles, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_aromatic_rings(mol: Chem.Mol):
    rings = []
    seen = set()

    try:
        for ring in Chem.GetSymmSSSR(mol):
            ring = [int(idx) for idx in ring]
            if len(ring) >= 5 and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                key = tuple(sorted(ring))
                if key not in seen:
                    rings.append(ring)
                    seen.add(key)
    except Exception:
        pass

    residues = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if not info:
            continue
        resname = info.GetResidueName().strip().upper()
        atom_name = info.GetName().strip().upper()
        key = (info.GetChainId().strip(), info.GetResidueNumber(), resname)
        residues.setdefault(key, {})[atom_name] = atom.GetIdx()

    for (_, _, resname), atoms_by_name in residues.items():
        for atom_names in _PROTEIN_AROMATIC_RINGS.get(resname, []):
            if all(name in atoms_by_name for name in atom_names):
                ring = [atoms_by_name[name] for name in atom_names]
                key = tuple(sorted(ring))
                if key not in seen:
                    rings.append(ring)
                    seen.add(key)

    return rings


def extract_physics_labels(mol: Chem.Mol, config: dict):
    mol.UpdatePropertyCache(strict=False)
    num_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer()
    pos = torch.from_numpy(conf.GetPositions().astype(np.float32))

    # Adjacency for exclusion
    adj = [set() for _ in range(num_atoms)]
    bonds, bond_lens = [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[u].add(v);
        adj[v].add(u)
        bonds.append([u, v])
        bond_lens.append(torch.norm(pos[u] - pos[v]))

    exclude_mask = torch.zeros((num_atoms, num_atoms), dtype=torch.bool)
    for i in range(num_atoms):
        exclude_mask[i, i] = True
        for n1 in adj[i]:
            exclude_mask[i, n1] = True
            for n2 in adj[n1]:
                exclude_mask[i, n2] = True
                for n3 in adj[n2]:
                    exclude_mask[i, n3] = True

    # Angles
    angles, ang_vals = [], []
    for i in range(num_atoms):
        nbs = list(adj[i])
        for idx_a in range(len(nbs)):
            for idx_b in range(idx_a + 1, len(nbs)):
                u, v = nbs[idx_a], nbs[idx_b]
                angles.append([u, i, v])
                v1, v2 = pos[u] - pos[i], pos[v] - pos[i]
                cos_t = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
                ang_vals.append(torch.acos(torch.clamp(cos_t, -1.0, 1.0)))

    # Dihedrals
    dihedrals, phi_vals = [], []
    for b in bonds:
        u, v = b[0], b[1]
        nu, nv = [n for n in adj[u] if n != v], [n for n in adj[v] if n != u]
        if nu and nv:
            dihedrals.append([nu[0], u, v, nv[0]])
            try:
                phi_vals.append(float(AllChem.GetDihedralRad(conf, nu[0], u, v, nv[0])))
            except:
                phi_vals.append(0.0)

    # Non-bonded
    dist_mat = torch.cdist(pos, pos, p=2)
    nb_mask = (dist_mat < 12.0) & (~exclude_mask)
    nb_indices = torch.triu(nb_mask, diagonal=1).nonzero(as_tuple=False)

    # HBonds
    hb_triplets = []
    if num_atoms < 12000:
        d_sm = Chem.MolFromSmarts(config.get('hbond_donors_smarts', '[N,O,S;!H0]'))
        a_sm = Chem.MolFromSmarts(config.get('hbond_acceptors_smarts', '[N,O,S;H0]'))
        donors = sum(mol.GetSubstructMatches(d_sm), ())
        acceptors = sum(mol.GetSubstructMatches(a_sm), ())
        for d in donors:
            for h_idx in adj[d]:
                if mol.GetAtomWithIdx(h_idx).GetAtomicNum() == 1:
                    for a in acceptors:
                        if d != a and dist_mat[d, a] < 3.5:
                            hb_triplets.append([d, h_idx, a])

    # Ring Pairs (Pi-Pi)
    rings = _extract_aromatic_rings(mol)
    pi_pairs = []
    if len(rings) >= 2:
        cents = torch.stack([pos[r].mean(dim=0) for r in rings])
        r_dist = torch.cdist(cents, cents)
        cutoff = float(config.get('pi_pi_distance_cutoff', 6.0))
        pi_idx = torch.triu(r_dist < cutoff, diagonal=1).nonzero()
        pi_pairs = [[torch.tensor(rings[p[0]]), torch.tensor(rings[p[1]])] for p in pi_idx]

    # Props
    charges = _compute_partial_charges(mol)
    dipoles = _compute_atomic_dipoles(pos, charges, adj)
    vdw_map = config.get('vdw_radii', {})
    radii = torch.tensor([vdw_map.get(a.GetSymbol(), 1.7) for a in mol.GetAtoms()])

    return {
        'bond_indices': torch.tensor(bonds, dtype=torch.long) if bonds else torch.empty((0, 2), dtype=torch.long),
        'ref_bond_lengths': torch.tensor(bond_lens, dtype=torch.float32) if bond_lens else torch.empty(0),
        'angle_indices': torch.tensor(angles, dtype=torch.long) if angles else torch.empty((0, 3), dtype=torch.long),
        'ref_angles': torch.tensor(ang_vals, dtype=torch.float32) if ang_vals else torch.empty(0),
        'dihedral_indices': torch.tensor(dihedrals, dtype=torch.long) if dihedrals else torch.empty((0, 4),
                                                                                                    dtype=torch.long),
        'true_dihedrals': torch.tensor(phi_vals, dtype=torch.float32) if phi_vals else torch.empty(0),
        'vdw_indices': nb_indices, 'electro_indices': nb_indices,
        'vdw_radii': radii, 'partial_charges': charges, 'dipole_vectors': dipoles,
        'hbond_indices': torch.tensor(hb_triplets, dtype=torch.long) if hb_triplets else torch.empty((0, 3),
                                                                                                     dtype=torch.long),
        'pi_pi_ring_pair_indices': pi_pairs
    }


def process_item(item_path: Path, task_config: dict):
    """Sub-process worker: DOES NOT use logging, only returns (id, status)."""
    pdb_id = item_path.name.lower()
    out_dir = PROJECT_ROOT / task_config.get('output_dir', 'data/processed_data/label') / pdb_id
    out_file = out_dir / f"{pdb_id}_labels.pt"

    if str(task_config.get('overwrite_existing', 'false')).lower() == 'false' and out_file.exists():
        return pdb_id, "skipped"

    try:
        prot = Chem.MolFromPDBFile(str(item_path / f"{pdb_id}_protein.pdb"), removeHs=False, sanitize=False)
        lig = next(Chem.SDMolSupplier(str(item_path / f"{pdb_id}_ligand.sdf"), removeHs=False, sanitize=False), None)
        if not prot or not lig: return pdb_id, "error: files missing or unreadable"

        complex_mol = Chem.CombineMols(prot, lig)
        num_atoms = complex_mol.GetNumAtoms()

        # Reordering [B, S, D]
        g_ids = []
        for a in complex_mol.GetAtoms():
            info = a.GetPDBResidueInfo()
            if info and info.GetResidueName().strip() in _STANDARD_AMINO_ACIDS:
                g_ids.append(0 if info.GetName().strip() in _BACKBONE_ATOMS else 1)
            else:
                g_ids.append(2)
        g_ids = torch.tensor(g_ids)

        new_order = torch.cat([(g_ids == 0).nonzero(as_tuple=True)[0],
                               (g_ids == 1).nonzero(as_tuple=True)[0],
                               (g_ids == 2).nonzero(as_tuple=True)[0]])
        old_to_new = torch.zeros(num_atoms, dtype=torch.long)
        old_to_new[new_order] = torch.arange(num_atoms)

        raw = extract_physics_labels(complex_mol, task_config)

        def remap(idx):
            return old_to_new[idx] if idx.numel() > 0 else idx

        labels = {
            'r_true': torch.from_numpy(complex_mol.GetConformer().GetPositions().astype(np.float32))[new_order],
            'atomic_nums': torch.tensor([a.GetAtomicNum() for a in complex_mol.GetAtoms()], dtype=torch.long)[
                new_order],
            'atom_group_ids': g_ids[new_order],
            'partial_charges': raw['partial_charges'][new_order],
            'dipole_vectors': raw['dipole_vectors'][new_order],
            'vdw_radii': raw['vdw_radii'][new_order],
            'bond_indices': remap(raw['bond_indices']),
            'ref_bond_lengths': raw['ref_bond_lengths'],
            'angle_indices': remap(raw['angle_indices']),
            'ref_angles': raw['ref_angles'],
            'dihedral_indices': remap(raw['dihedral_indices']),
            'true_dihedrals': raw['true_dihedrals'],
            'vdw_indices': remap(raw['vdw_indices']),
            'electro_indices': remap(raw['electro_indices']),
            'hbond_indices': remap(raw['hbond_indices']),
            'pi_pi_ring_pair_indices': [[remap(p[0]), remap(p[1])] for p in raw['pi_pi_ring_pair_indices']]
        }

        # r_init
        r_init = None
        init_dir = PROJECT_ROOT / task_config['initial_structure_dir'] / pdb_id
        tmpl = task_config['split_file_templates']
        try:
            m_b = Chem.MolFromPDBFile(str(init_dir / f"{pdb_id}{tmpl['backbone_suffix']}"), removeHs=False,
                                      sanitize=False)
            m_s = Chem.MolFromPDBFile(str(init_dir / f"{pdb_id}{tmpl['sidechain_suffix']}"), removeHs=False,
                                      sanitize=False)
            m_l = next(
                Chem.SDMolSupplier(str(init_dir / f"{pdb_id}{tmpl['ligand_suffix']}"), removeHs=False, sanitize=False),
                None)
            if m_b and m_s and m_l:
                init_mol = Chem.CombineMols(Chem.CombineMols(m_b, m_s), m_l)
                init_map = {gid: idx for idx, gid in enumerate(get_atom_ids_fast(init_mol))}
                comp_ids = get_atom_ids_fast(complex_mol)
                aligned = [init_mol.GetConformer().GetPositions()[init_map[comp_ids[old_idx]]]
                           if comp_ids[old_idx] in init_map
                           else labels['r_true'][idx_in_new].numpy() + np.random.randn(3) * 2.0
                           for idx_in_new, old_idx in enumerate(new_order.tolist())]
                r_init = torch.tensor(np.array(aligned), dtype=torch.float32)
        except:
            pass
        labels['r_init'] = r_init if r_init is not None else labels['r_true'] + torch.randn_like(labels['r_true']) * 2.0

        # Atomic Save
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = out_file.with_suffix('.tmp')
        torch.save(labels, tmp_file)
        os.rename(tmp_file, out_file)

        return pdb_id, "success"
    except Exception as e:
        return pdb_id, f"failed error: {str(e)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Setup Master Process Logging
    log_file_path = setup_logging_from_config(config, PROJECT_ROOT)
    task_config = config['pipeline_tasks']['phy_geo_task']

    all_dirs = []
    for src in task_config['data_sources']:
        src_path = PROJECT_ROOT / src['input_dir']
        if src_path.exists():
            all_dirs.extend([d for d in src_path.iterdir() if d.is_dir()])

    num_procs = max(1, cpu_count() // 2)
    logging.info(f"Master: Processing {len(all_dirs)} folders using {num_procs} workers...")

    worker_func = partial(process_item, task_config=task_config)

    results = []
    with Pool(processes=num_procs) as pool:
        pbar = tqdm(pool.imap_unordered(worker_func, all_dirs), total=len(all_dirs), desc="Preprocessing")
        for res in pbar:
            results.append(res)
            # If failed, print immediate notification to console
            if "failed" in res[1]:
                pbar.set_postfix({"Last Error": res[0]})

    # 2. Categorize and Audit Results
    successes = [r for r in results if r[1] == "success"]
    skipped = [r for r in results if r[1] == "skipped"]
    failures = [r for r in results if "failed" in r[1]]

    # 3. Force Write Report to Log File (using raw file handle for reliability)
    with open(log_file_path, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"FINAL PREPROCESSING REPORT - {datetime.datetime.now()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Folders Scanned: {len(all_dirs)}\n")
        f.write(f"Successfully Created: {len(successes)}\n")
        f.write(f"Skipped (Existed):    {len(skipped)}\n")
        f.write(f"Failed (Total):       {len(failures)}\n")
        f.write("-" * 60 + "\n")

        if failures:
            f.write("DETAILED FAILURE LIST:\n")
            for pid, reason in failures:
                f.write(f"PDB_ID: {pid:8} | REASON: {reason}\n")
        else:
            f.write("NO FAILURES ENCOUNTERED.\n")
        f.write("=" * 60 + "\n")

    # 4. Generate a separate failed_list for easy reference
    if failures:
        fail_txt = log_file_path.parent / "failed_pdbs.txt"
        with open(fail_txt, 'w') as f:
            for pid, reason in failures:
                f.write(f"{pid}\t{reason}\n")
        logging.info(f"Quick-access failed list saved to {fail_txt}")

    logging.info(f"Audit complete. Audit report appended to {log_file_path}")


if __name__ == "__main__":
    main()
