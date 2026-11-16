# data generated here is going to be used in benchmark_moleculenet
# i just didn't want to bloat the process_moleculenet file

import numpy as np
import torch
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')
import re
import os

def parse_atomref(file):
    f = open(file, "r") # atomref.txt
    text = f.read()
    features = {
        "zpve": torch.zeros(10, dtype=torch.float64),
        "u0": torch.zeros(10, dtype=torch.float64),
        "u298": torch.zeros(10, dtype=torch.float64),
        "h298": torch.zeros(10, dtype=torch.float64),
        "g298": torch.zeros(10, dtype=torch.float64),
        "cv": torch.zeros(10, dtype=torch.float64),
    }
    atomic_numbers = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
    }
    columns = list(features.keys())
    for line in text.splitlines():
        match = re.match(
            r"^\s*([HCNOF])\s+"
            r"([-+]?\d*\.\d+)\s+"
            r"([-+]?\d*\.\d+)\s+"
            r"([-+]?\d*\.\d+)\s+"
            r"([-+]?\d*\.\d+)\s+"
            r"([-+]?\d*\.\d+)\s+"
            r"([-+]?\d*\.\d+)\s*$",
            line,
        ) # thanks gpt
        if not match: continue
        element = match.group(1)
        values = [float(x) for x in match.groups()[1:]]
        z = atomic_numbers[element]
        for feature, value in zip(columns, values):
            features[feature][z] = value
    f.close()
    return features

allowed_numbers = set([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53])

def process_qm_dataset(sdf_path, csv_path, save_dir, target_cols, out_csv_name):
    """
    target_cols: list of column names to use as regression targets (multitask).
    Saves a wide CSV with "smiles" + one column per target, and .pt files with
    "target" shaped [N, len(target_cols)].
    """
    props_df = pd.read_csv(csv_path)
    features = parse_atomref(os.path.join(os.path.dirname(os.path.abspath(__file__)), "atomref.txt"))
    for col in target_cols:
        assert col in props_df.columns, f"{col} not found; available: {props_df.columns.tolist()}"
    targets_all = props_df[target_cols].values.astype(np.float64)  # [N_csv, T]
    # target column index -> per-element reference values (float64, indexed by atomic number)
    col_to_ref = {target_cols.index(k): ref.double().numpy() for k, ref in features.items() if k in target_cols}

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    assert len(supplier) == len(props_df)

    molecules = {}
    smiles_out, energy_out = [], []
    skipped, kept = 0, 0

    for index, mol in enumerate(tqdm(supplier, desc=f"Processing {out_csv_name}", total=len(supplier))):
        smiles = ""
        if mol is not None:
            try:
                numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                if set(numbers) - allowed_numbers == set():
                    conf = mol.GetConformer()
                    coord = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
                    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                smiles = ""

        if smiles == "":
            skipped += 1
            continue

        num_atoms = len(numbers)
        if num_atoms not in molecules:
            molecules[num_atoms] = {"coord": [], "numbers": [], "target": [], "charge": [], "id": []}
        molecules[num_atoms]["coord"].append(torch.tensor(coord))
        molecules[num_atoms]["numbers"].append(torch.tensor(numbers))
        molecules[num_atoms]["charge"].append(Chem.GetFormalCharge(mol)) # total molecular charge; dataset.py passes it to AIMNet2
        z = np.array(numbers)
        target = targets_all[index].copy()  # [T]-length row, float64
        for col, ref in col_to_ref.items():
            target[col] -= ref[z].sum()  # subtract atomic reference values (atomization quantity)
        molecules[num_atoms]["target"].append(target)
        # id = row index in the output csv (not the source csv), so load_id_to_smiles lines up after skips
        molecules[num_atoms]["id"].append(kept)

        smiles_out.append(smiles)
        energy_out.append(target)  # keep the csv consistent with the .pt targets
        kept += 1

    print(f"[{out_csv_name}] skipped {skipped}, kept {kept}")

    csv_data = {"smiles": smiles_out}
    energy_arr = np.array(energy_out)  # [kept, T]
    for t_idx, col_name in enumerate(target_cols):
        csv_data[col_name] = energy_arr[:, t_idx]
    pd.DataFrame(csv_data).to_csv(f"{save_dir}/{out_csv_name}", index=False)

    for num_atoms in molecules:
        ids = torch.tensor(molecules[num_atoms]["id"], dtype=torch.int64)
        target_stack = torch.tensor(np.stack(molecules[num_atoms]["target"]), dtype=torch.float32)  # [n, T]
        entry = {
            "coord": torch.stack(molecules[num_atoms]["coord"]),
            "numbers": torch.stack(molecules[num_atoms]["numbers"]),
            "target": target_stack,
            "charge": torch.tensor(molecules[num_atoms]["charge"], dtype=torch.int8),  # [n]
            "id": ids,
        }
        entry["coord"] = entry["coord"].reshape((-1, 1, num_atoms, 3))
        assert len(entry["coord"].shape) == 4
        assert len(entry["target"].shape) <= 2
        torch.save(entry, f"{save_dir}/{str(num_atoms).zfill(3)}.pt")


# # QM7 -- single task, only one column exists
save_dir_qm7 = "SAVE_DIR_QM7"
process_qm_dataset(
    sdf_path=f"{save_dir_qm7}/gdb7_v2.sdf",
    csv_path=f"{save_dir_qm7}/gdb7_v2.sdf.csv",
    save_dir=save_dir_qm7,
    target_cols=["u0_atom"],
    out_csv_name="qm7.csv",
)

# # QM8 -- 12 tasks: CC2, PBE0, CAM levels of theory (4 each); PBE0.1 duplicate columns excluded
save_dir_qm8 = "SAVE_DIR_QM8"
process_qm_dataset(
    sdf_path=f"{save_dir_qm8}/qm8.sdf",
    csv_path=f"{save_dir_qm8}/qm8.sdf.csv",
    save_dir=save_dir_qm8,
    target_cols=["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2",
                 "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0",
                 "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"],
    out_csv_name="qm8.csv",
)

# QM9 -- 12 standard tasks (excludes rotational constants A/B/C and *_atom variants)
save_dir_qm9 = "SAVE_DIR_QM9"
process_qm_dataset(
    sdf_path=f"{save_dir_qm9}/qm9.sdf",
    csv_path=f"{save_dir_qm9}/qm9.sdf.csv",
    save_dir=save_dir_qm9,
    target_cols=["mu", "alpha", "homo", "lumo", "gap", "r2",
                 "zpve", "cv", 
                 "u0", "u298", "h298", "g298"],
    out_csv_name="qm9.csv",
)