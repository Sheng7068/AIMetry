import numpy as np
import torch
import os
import rdkit
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
from multiprocessing import Pool
RDLogger.DisableLog('rdApp.*')

allowed_numbers = set([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53])
def _process_one(args):
    index, smile = args
    try:
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        if set(numbers) - allowed_numbers != set(): return (index, None, smile, None, None)
        if len(numbers) > 150: return (index, None, smile, None, None) # taking too long to generate a molecule, so only <= 150 molecules allowed
        try: AllChem.MMFFOptimizeMoleculeConfs(mol)
        except: return (index, None, smile, None, None)
        conf = mol.GetConformer()
        coord = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        charge = Chem.GetFormalCharge(mol) # total molecular charge; dataset.py passes it to AIMNet2 (defaults to 0 if missing)
        return (index, len(numbers), coord, numbers, charge)
    except:
        return (index, None, smile, None, None)

def save_smiles_targets(smiles, targets, save_dir, n_workers=10, target_type=torch.int8):
    assert smiles.shape[0] == targets.shape[0] and len(targets.shape) <= 2
    molecules = {}
    skipped = 0
    kept = 0
    with Pool(n_workers) as pool:
        for index, num_atoms, coord, numbers, charge in tqdm(pool.imap_unordered(_process_one, enumerate(smiles)), total=len(smiles)):
            if num_atoms is None:
                skipped += 1
                # print(f"smiles: {coord} didnt parse properly, removing from dataset")
                continue
            if num_atoms not in molecules: molecules[num_atoms] = {"coord": [], "numbers": [], "target": [], "charge": [], "id": []}
            molecules[num_atoms]["coord"].append(torch.tensor(coord))
            molecules[num_atoms]["numbers"].append(torch.tensor(numbers))
            molecules[num_atoms]["target"].append(targets[index])
            molecules[num_atoms]["charge"].append(torch.tensor(charge, dtype=torch.int8)) # stacked to [num_molecules]
            molecules[num_atoms]["id"].append(index)  # original CSV row index, carried through
            kept += 1
    print(f"[{save_dir}] skipped a total of {skipped} points, kept {kept} points")
    for num_atoms in molecules:
        ids = torch.tensor(molecules[num_atoms]["id"], dtype=torch.int64)  # pop before the generic stack loop below
        molecules[num_atoms] = {k: torch.stack(v) for k, v in molecules[num_atoms].items() if k != "id"}
        molecules[num_atoms]["id"] = ids
        molecules[num_atoms]["coord"] = molecules[num_atoms]["coord"].reshape((-1, 1, num_atoms, 3))
        assert len(molecules[num_atoms]["coord"].shape) == 4 # [num_molecules, num_conformers, N, 3]
        molecules[num_atoms]["target"] = molecules[num_atoms]["target"].to(target_type)
        assert len(molecules[num_atoms]["target"].shape) <= 2 
        torch.save(molecules[num_atoms], f"{save_dir}/{str(num_atoms).zfill(3)}.pt")

# BBBP:
data_dir = "DATA_DIR_BBBP"
save_dir = "SAVE_DIR_BBBP"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 2]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 3]
save_smiles_targets(smiles, targets, save_dir)
    
# tox21:
data_dir = "DATA_DIR_TOX21"
save_dir = "SAVE_DIR_TOX21"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 0:12]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 13]
save_smiles_targets(smiles, targets, save_dir)

# toxcast
data_dir = "DATA_DIR_TOXCAST"
save_dir = "SAVE_DIR_TOXCAST"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 1:]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 0]
save_smiles_targets(smiles, targets, save_dir)

# sider
data_dir = "DATA_DIR_SIDER"
save_dir = "SAVE_DIR_SIDER"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 1:]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 0]
save_smiles_targets(smiles, targets, save_dir)

# clintox
data_dir = "DATA_DIR_CLINTOX"
save_dir = "SAVE_DIR_CLINTOX"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 1:]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 0]
save_smiles_targets(smiles, targets, save_dir)

# muv
data_dir = "DATA_DIR_MUV"
save_dir = "SAVE_DIR_MUV"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, :-2]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, -1]
filter_empty = smiles != ""
smiles = smiles[filter_empty]
targets = targets[filter_empty]
save_smiles_targets(smiles, targets, save_dir)

# bace
data_dir = "DATA_DIR_BACE"
save_dir = "SAVE_DIR_BACE"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 2]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 0]
save_smiles_targets(smiles, targets, save_dir)

# HIV
data_dir = "DATA_DIR_HIV"
save_dir = "SAVE_DIR_HIV"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 2]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 0]
filter_empty = smiles != ""
smiles = smiles[filter_empty]
targets = targets[filter_empty]
save_smiles_targets(smiles, targets, save_dir)

# esol
data_dir = "DATA_DIR_ESOL"
save_dir = "SAVE_DIR_ESOL"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 8]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 9]
filter_empty = smiles != ""
smiles = smiles[filter_empty]
targets = targets[filter_empty]
save_smiles_targets(smiles, targets, save_dir, target_type=torch.float)

# lipo
data_dir = "DATA_DIR_LIPO"
save_dir = "SAVE_DIR_LIPO"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 1]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 2]
filter_empty = smiles != ""
smiles = smiles[filter_empty]
targets = targets[filter_empty]
save_smiles_targets(smiles, targets, save_dir, target_type=torch.float)

# freesol
data_dir = "DATA_DIR_FREESOL"
save_dir = "SAVE_DIR_FREESOL"
df = pd.read_csv(data_dir, sep=',', header=None).values
targets = df[1:, 2]
targets = torch.tensor(targets.astype(float))
smiles = df[1:, 1]
filter_empty = smiles != ""
smiles = smiles[filter_empty]
targets = targets[filter_empty]
save_smiles_targets(smiles, targets, save_dir, target_type=torch.float)