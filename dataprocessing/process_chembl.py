### data used to train the model
### source: ChEMBL36
### data: download online (https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/)
### specifically,chembl_36.fps.gz and chembl_36_chemreps.txt.gz

import os, sys, time, pickle, glob
from tqdm import tqdm
from argparse import ArgumentParser
from multiprocessing import Pool
from functools import partial

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))) # pretty sure this isn't needed?

############# PREPROCESSING FUNCTIONS
def get_ids_smiles(chemreps_path):
    with open(chemreps_path, "r") as f:
        chemreps = f.read()

        chembl_ids = []
        smiles = []
        seen = set()
        for i, data in enumerate(chemreps.splitlines()):
            if i == 0: 
                print(f"information parsed: {' '.join(data.split())}")
                continue
            chembl_id, smile = data.split()[0], data.split()[1]
            if smile in seen: continue # some SMILES keys in chembl_36 are equivalent even if the molecule is in a E/Z alkene config, this filters it out
            chembl_ids.append(chembl_id)
            smiles.append(smile)
            seen.add(smile)
    print("done, sample of list:")
    print(len(chembl_ids), chembl_ids[0])
    print(len(smiles), smiles[0])
    return chembl_ids, smiles

def filter_ids(chembl_ids, smiles, size_filter):
    # N = set([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 34, 35, 53, 80, 83]) # supported by LoQI
    N =   set([1, 5, 6, 7, 8, 9,     14, 15, 16, 17, 33, 34, 35, 53]) # supported by AIMNet2, lapse!
    def filter_large_molecules(info, size):
        chembl_id, smiles = info
        mol = Chem.MolFromSmiles(smiles)
        if mol == None: return False # implicitly gets rid of bad smiles
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        atomic_nums = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
        charge = Chem.GetFormalCharge(mol)
        return 1 < num_atoms <= size and atomic_nums - N == set() and all(atom.GetFormalCharge() in range(-2, 4) for atom in mol.GetAtoms())

    f = partial(filter_large_molecules, size=size_filter)
    molecules = list(zip(chembl_ids, smiles))
    print("original length:", len(molecules))
    print(f"filtering out molecules with more than {size_filter} number of atoms as well as those with elements outside our defined set")
    molecules_subset = filter(f, molecules)
    chembl_ids_subset, smiles_subset = zip(*molecules_subset)
    print("done, sample of list:")
    print(len(chembl_ids_subset), chembl_ids[0])
    print(len(smiles_subset), smiles[0])
    assert(len(smiles_subset) == len(set(smiles_subset))), "There are duplicate smiles"
    print("no duplicates :)")
    return chembl_ids_subset, smiles_subset

def get_fingerprints(chembl_ids_subset, fingerprint_file):
    print("mapping the chembl_id fingerprint from the chembl_36.fps file")
    d = {}
    with open(fingerprint_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i < 6: # prints metadata
                print(line)
                continue
            hex_fp, chembl_id = line.split("\t")
            if chembl_id in d: assert(hex_fp == d[chembl_id])
            d[chembl_id] = hex_fp

    def fps_to_numbers(s): 
        # chembl36 stores fingerprints as 256 byte bitvector, which is a 512 length string
        # a substring of 16 = 8 bytes = 64 bits = one 64 uint
        return [int(s[i:i+16], 16) for i in range(0, 512, 16)]

    fingerprints_subset = []
    for chembl_id in chembl_ids_subset:
        fingerprints_subset.append(fps_to_numbers(d[chembl_id]))
    return fingerprints_subset

def save_id_smiles_fingerprint(chembl_ids_subset, smiles_subset,fingerprints_subset, save_dir):
    numatoms_to_info = {}
    for i in range(len(smiles_subset)):
        chembl_id = chembl_ids_subset[i]
        smiles = smiles_subset[i]
        fingerprint = fingerprints_subset[i]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        if num_atoms in numatoms_to_info: numatoms_to_info[num_atoms].append((chembl_id, smiles, fingerprint, i))
        else: numatoms_to_info[num_atoms] = [(chembl_id, smiles, fingerprint, i)]

    # save_dir = "DATA_ROOT/chembl/chembl_le128"
    os.makedirs(save_dir, exist_ok=True)
    dtype = np.dtype([
        ("chembl_id", "S32"), # ascii string up to 32 chars
        ("smiles", "S512"), # ascii string up to 512 chars
        ("fingerprint", np.uint64, (32,)),
        ("id", np.uint32)
    ])
    for k in numatoms_to_info:
        data = np.array(numatoms_to_info[k], dtype=dtype)
        np.save(f"{save_dir}/{str(k).zfill(3)}_id_smiles_fingerprint.npy", data)

    print("info on the number of molecules split by the number of atoms")
    number_of_mols = np.zeros(max(numatoms_to_info.keys()) + 1, dtype=np.uint32)
    for k in numatoms_to_info:
        number_of_mols[k] = len(numatoms_to_info[k])
    print(number_of_mols)

def preprocess(args):
    # chemreps_path = "DATA_ROOT/chembl/chembl_36_chemreps.txt"
    chembl_ids, smiles = get_ids_smiles(args.chembl_path)
    chembl_ids_subset, smiles_subset = filter_ids(chembl_ids, smiles, args.size_filter)
    # path = "DATA_ROOT/chembl/chembl_36.fps"
    fingerprints_subset = get_fingerprints(chembl_ids_subset, args.fingerprint_file)
    save_id_smiles_fingerprint(chembl_ids_subset, smiles_subset, fingerprints_subset, args.id_smiles_fps_dir)



############# PROCESSING FUNCTIONS 
full_atom_encoder = {
    "H": 0, 
    "B": 1, "C": 2, "N": 3, "O": 4, "F": 5, 
    "Al": 6, "Si": 7, "P": 8, "S": 9, "Cl": 10, 
    "As": 11, "Br": 12,
    "I": 13,
    "Hg": 14, "Bi": 15, "Se": 16}
idx_to_number = torch.tensor([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83, 34], dtype=torch.int) # note that it should be out of order by above
# the reason we have full_atom_encoder and idx_to_number separately is because legacy code from LoQI
def add_stereo_bonds(mol, chi_bonds, ez_bonds, edge_index=None, edge_attr=None, from_3D=True):
    result = []
    if from_3D and mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if bond.GetBondType() == Chem.BondType.DOUBLE and stereo in ez_bonds:
            idx_3, idx_4 = bond.GetStereoAtoms()
            atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
            idx_1, idx_2 = atom_1.GetIdx(), atom_2.GetIdx()

            idx_5 = [nbr.GetIdx() for nbr in atom_1.GetNeighbors() if nbr.GetIdx() not in {idx_2, idx_3}]
            idx_6 = [nbr.GetIdx() for nbr in atom_2.GetNeighbors() if nbr.GetIdx() not in {idx_1, idx_4}]

            inv_stereo = Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ
            result.extend([(idx_3, idx_4, ez_bonds[stereo]), (idx_4, idx_3, ez_bonds[stereo])])

            if idx_5:
                result.extend([(idx_5[0], idx_4, ez_bonds[inv_stereo]), (idx_4, idx_5[0], ez_bonds[inv_stereo])])
            if idx_6:
                result.extend([(idx_3, idx_6[0], ez_bonds[inv_stereo]), (idx_6[0], idx_3, ez_bonds[inv_stereo])])
            if idx_5 and idx_6:
                result.extend([(idx_5[0], idx_6[0], ez_bonds[stereo]), (idx_6[0], idx_5[0], ez_bonds[stereo])])

        if bond.GetBeginAtom().HasProp('_CIPCode'):
            idx = bond.GetBeginAtom().GetIdx()
            chirality = bond.GetBeginAtom().GetProp('_CIPCode')
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_neighbors = sorted(neighbors, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
                sorted_neighbors = [a.GetIdx() for a in sorted_neighbors]
                a, b, c = sorted_neighbors[:3] if chirality == "R" else sorted_neighbors[:3][::-1]
                d = sorted_neighbors[-1]
                result.extend([
                    (a, d, chi_bonds[0]), (b, d, chi_bonds[0]), (c, d, chi_bonds[0]),
                    (d, a, chi_bonds[0]), (d, b, chi_bonds[0]), (d, c, chi_bonds[0]),
                    (b, a, chi_bonds[1]), (c, b, chi_bonds[1]), (a, c, chi_bonds[1])
                ])

    if not result:
        return edge_index, edge_attr
    new_edge_index = torch.tensor([[i, j] for i, j, _ in result], dtype=torch.long).T
    new_edge_attr = torch.tensor([b for _, _, b in result], dtype=torch.uint8)

    if edge_index is None:
        return new_edge_index, new_edge_attr
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    edge_attr = torch.cat([edge_attr, new_edge_attr])
    return edge_index, edge_attr

def mol_to_torch_geometric(mol, conf_id=0):
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.to(torch.uint8)
    pos = torch.tensor(mol.GetConformer(conf_id).GetPositions()).float()
    atom_types = torch.tensor([full_atom_encoder[atom.GetSymbol()] for atom in mol.GetAtoms()], dtype=torch.uint8)
    all_charges = torch.tensor([atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.int8)
    chi_bonds = [7, 8]
    ez_bonds = {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}
    edge_index, edge_attr = add_stereo_bonds(mol, chi_bonds, ez_bonds, edge_index, edge_attr, from_3D=True)
    return Data(x=atom_types, edge_index=edge_index, edge_attr=edge_attr.to(torch.uint8), pos=pos, charges=all_charges)

def embed_single(args_tuple):
    smi, n_confs, smiles_id, fingerprint = args_tuple
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=42)
    if len(ids) != n_confs: return None
    try: AllChem.MMFFOptimizeMoleculeConfs(mol)
    except: return None
    return mol, smiles_id, fingerprint

def load_and_embed(path, n_confs, num_workers=8):
    data = np.load(path, allow_pickle=True)
    smiles_list = [s.decode() for s in data["smiles"]]
    smiles_id_list = data["id"]
    fingerprints_list = data["fingerprint"]
    task_args = [(smi, n_confs, smiles_id_list[i], fingerprints_list[i]) 
                 for i, smi in enumerate(smiles_list)]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(embed_single, task_args), 
                           total=len(task_args), desc="embedding"))

    mols, return_smiles_id, return_fingerprints = [], [], []
    skipped = 0
    for r in results:
        if r is None: skipped += 1
        else:
            mol, sid, fp = r
            mols.append(mol)
            return_smiles_id.append(sid)
            return_fingerprints.append(fp)
    print(f"for {path}, skipped {skipped} datapoints")
    return mols, return_smiles_id, return_fingerprints

def process(args):
    files = sorted(glob.glob(f"{args.id_smiles_fps_dir}/[0-9][0-9][0-9]_id_smiles_fingerprint.npy"))

    for file in files:
        prefix = os.path.basename(file)[:3]
        out_path = f"{args.output}/{prefix}_processed.pt"
        original64_path = f"DATA_ROOT/chembl/chembl_le64_rdkit/{prefix}_processed.pt"
        id_smiles_fingerprint_path64 = f"DATA_ROOT/chembl/chembl_le64/{prefix}_id_smiles_fingerprint.npy"
        if os.path.isfile(original64_path):
            if os.path.isfile(out_path):
                print(f"already remapped {file} in le128, skipping")
                continue    
            print(f"already processed {file} in le64, remapping")
            id_smiles_fingerprint_data64 = np.load(id_smiles_fingerprint_path64, allow_pickle=True)
            id_smiles_fingerprint_data128 = np.load(file, allow_pickle=True)
            original64_data = torch.load(original64_path)
            for i, original64_id in tqdm(enumerate(original64_data["id"])):
                smiles = id_smiles_fingerprint_data64["smiles"][id_smiles_fingerprint_data64["id"] == original64_data["id"][i]]
                new_id = id_smiles_fingerprint_data128["id"][id_smiles_fingerprint_data128["smiles"] == smiles]
                original64_data["id"][i] = torch.tensor(new_id, dtype=torch.int)
            torch.save(original64_data, out_path)
            continue
        ### delete above when done
        if os.path.isfile(out_path):
            print(f"already processed {file}, skipping")
            continue

        print(f"processing {file}")
        start = time.time()

        original_data = np.load(file, allow_pickle=True)
        mols, smiles_id, fingerprint = load_and_embed(file, n_confs=args.n_confs)

        # filter None and track original indices
        valid = [(i, mol) for i, mol in enumerate(mols)]
        num_molecules = len(valid)
        num_atoms = int(prefix)

        coord       = torch.zeros((num_molecules, args.n_confs, num_atoms, 3), dtype=torch.float)
        numbers     = torch.zeros((num_molecules, num_atoms), dtype=torch.uint8)
        charges     = torch.zeros((num_molecules, num_atoms), dtype=torch.int8)
        smiles_id   = torch.tensor(smiles_id, dtype=torch.int)
        fingerprint = torch.tensor(np.array(fingerprint).astype(np.int64), dtype=torch.int64)

        ptr         = torch.zeros(num_molecules + 1, dtype=torch.int)
        edge_attr_list  = []
        edge_index_list = []

        for mol_idx, mol in tqdm(valid, desc="converting"):
            data_graph = mol_to_torch_geometric(mol, conf_id=0)
            numbers[mol_idx]     = idx_to_number[data_graph.x.to(torch.int)]
            charges[mol_idx]     = data_graph.charges
            edge_index_list.append(data_graph.edge_index)
            edge_attr_list.append(data_graph.edge_attr)
            ptr[mol_idx + 1]     = ptr[mol_idx] + data_graph.edge_index.shape[-1]

            for conf_idx in range(args.n_confs):
                coord[mol_idx, conf_idx] = torch.tensor(mol.GetConformer(conf_idx).GetPositions()).float()

        output = {
            "coord":       coord,
            "numbers":     numbers,
            "charges":     charges,
            "id":          smiles_id,
            "fingerprint": fingerprint,
            "ptr":         ptr,
            "edge_index":  torch.cat(edge_index_list, dim=1),
            "edge_attr":   torch.cat(edge_attr_list),
        }
        for k in output:
            print(k, output[k].shape)

        torch.save(output, out_path)
        print(f"Done! Took {((time.time() - start) / 60):.3f} minutes")

############# POSTPROCESSING FUNCTIONS

def id_smiles_fingerprint_path_to_smiles_dict(path):
    res = {}
    for file in os.listdir(path):
        data = np.load(f"{path}/{file}", allow_pickle=True)
        ids = data["id"].tolist()
        smiles = np.char.decode(data["smiles"], "ascii")
        res.update(zip(ids, smiles))
    return res

def count_substructures_single(args_tuple):
    smiles, smarts_keys = args_tuple
    res = []
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    res.append(Descriptors.MolWt(mol))
    res.append(rdMolDescriptors.CalcNumRings(mol))
    res.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    for key in smarts_keys:
        substructure = Chem.MolFromSmarts(key)
        res.append(len(mol.GetSubstructMatches(substructure)))
    return res

def postprocess(args):
    smarts_keys = [
        '[CX4][OX2H,OX1-]', '[c][OX2H,OX1-]', '[CX3;H1,$(C-[#6])](=[OX1])[OX2H,OX1-]', '[CX3;H1,$(C-[#6])](=[OX1])[OX2H0][#6;!$(C=[O,S,N])]',
        '[NX3][CX3;H1,$(C-[#6])](=[OX1])', '[NX3H2+0,NX4H3+;!$(N-[!#6;!#1]);!$(N-C=[N,O,S]);!$(N-C#N)]', '[NX3H1+0,NX4H2+;!$(N-[!#6;!#1]);!$(N-C=[N,O,S]);!$(N-C#N)]', '[NX3H0+0,NX4H1+;!$(N-[!#6;!#1]);!$(N-C=[N,O,S]);!$(N-C#N)]',
        '[$([NX3+0]-[c]),$([NX4+;H1,H2,H3]-[c]);!$(N-[!#6;!#1]);!$(N-C=[N,O,S]);!$(N-C#N)]', '[NX1]#[CX2]', '[F]',
        '[Cl]', '[Br]', '[I]', '[#6]-[SX4](=,-[OX1;+0,-1])(=,-[OX1;+0,-1])-[NX3,NX2-]',
        '[OX2]([#6;!$(C=[O,S,N])])[#6;!$(C=[O,S,N])]', '[#6][CX3](=O)[#6]', '[CX3;H2,$([CH1]-[#6])]=[OX1]', '[CX3]=[OX1]',
        '[#6][SX2H,SX1-]', '[#6]-[SX2H0+0]-[#6]', '[NX3,NX4+][CX4][CX3](=[OX1])[OX2H,OX1-,NX3]',
        '[NX2]=[OX1]', '[$([NX2+](=[NX1-])=[NX1-,NX2]),$([NX2+](#[NX1])-[NX2-])]', '[CX3]=[NX2+0,NX3+]',
        '[NX3][CX3;H1,$(C-[#6])]=[NX2+0,NX3+]', '[NX3][CX3](=[NX2+0,NX3+])[NX3]', '[NX3][CX3](=[OX1])[NX3]',
        '[NX3][CX3](=[OX1])[OX2,OX1-]', '[NX3][CX3](=[SX1])[NX3]',
        '[#6]-[SX4](=,-[OX1;+0,-1])(=,-[OX1;+0,-1])-[OX2H,OX1-]', '[#6]-[SX4](=,-[OX1;+0,-1])(=,-[OX1;+0,-1])-[OX2H0]-[#6]',
        '[#6]-[SX3](=,-[OX1;+0,-1])-[#6]', '[#6]-[SX4](=,-[OX1;+0,-1])(=,-[OX1;+0,-1])-[#6]',
        '[PX4](=,-[OX1;+0,-1])([OX2,OX1-])([OX2,OX1-])[OX2,OX1-]', '[PX4](=,-[OX1;+0,-1])([#6])([OX2,OX1-])[OX2,OX1-]',
        '[CX3]=[CX3]', '[CX2]#[CX2]', '[cX3]1[cX3][cX3][cX3][cX3][cX3]1',
        '[OX2]1[CX4][CX4]1', '[CX3](=[OX1])[F,Cl,Br,I]', '[CX3](=[OX1])[OX2][CX3](=[OX1])',
        '[NX2]=[CX2]=[OX1]', '[NX2]=[CX2]=[SX1]', '[#6]-[NX2+0]=[NX2+0]-[#6]', '[NX3][NX3]',
        '[CX3]=[NX2][OX2H,OX1-]', '[#7+;!$([#7+](=[OX1])[OX1-])]-[OX1-]', '[NX4+]([#6])([#6])([#6])[#6]'
    ]

    smarts_names = [
        'alcohol', 'phenol', 'carboxylic_acid', 'ester',
        'amide', 'amine_primary', 'amine_secondary', 'amine_tertiary',
        'aniline', 'nitrile', 'fluorine',
        'chlorine', 'bromine', 'iodine', 'sulfonamide',
        'ether', 'ketone', 'aldehyde', 'carbonyl',
        'thiol', 'sulfide', 'peptide',
        'nitroso', 'azide', 'imine',
        'amidine', 'guanidine', 'urea',
        'carbamate', 'thiourea',
        'sulfonic_acid', 'sulfonate_ester',
        'sulfoxide', 'sulfone',
        'phosphate', 'phosphonate',
        'alkene', 'alkyne', 'benzene_ring',
        'epoxide', 'acyl_halide', 'anhydride',
        'isocyanate', 'isothiocyanate', 'azo', 'hydrazine',
        'oxime', 'n_oxide', 'quaternary_ammonium'
    ]
    smiles_dict = id_smiles_fingerprint_path_to_smiles_dict(args.id_smiles_fps_dir)
    torch_files = list(sorted(glob.glob(f"{args.output}/*.pt")))
    for file in torch_files:
        data = torch.load(file)
        task_args = [(smiles_dict[int(smiles_id)], smarts_keys) for smiles_id in data["id"]]
        with Pool(processes=8) as pool:
            results = list(tqdm(pool.imap(count_substructures_single, task_args), total=len(task_args), desc="counting"))
    
        data["weight"] = torch.tensor([results[i][0] for i in range(len(results))], dtype=torch.float)
        data["rings_total"] = torch.tensor([results[i][1] for i in range(len(results))], dtype=torch.float)
        data["rings_aromatic"] = torch.tensor([results[i][2] for i in range(len(results))], dtype=torch.float)
        for j, name in enumerate(smarts_names):
            data[name] = torch.tensor([results[i][j+3] for i in range(len(results))], dtype=torch.int)
        torch.save(data, file)
        print(f"metadata on {file}:")
        for name in smarts_names:
            print(f"{(name+':'):<18}", end=" ")
            print(f"mean: {float(torch.mean(data[name].to(torch.float))):.3f}",end=", ")
            print(f"median: {torch.median(data[name]):}",end=", ")
            print(f"stdev: {float(torch.std(data[name].to(torch.float))):.3f}",end=", ")
            print(f"min: {torch.min(data[name])}",end=", ")
            print(f"max: {torch.max(data[name])}")


def main():
    parser = ArgumentParser()

    parser.add_argument("--chembl_path", type=str, required=True) 
    parser.add_argument("--size_filter", type=int, default=128) # 128
    parser.add_argument("--fingerprint_file", type=str, required=True) 
    parser.add_argument("--id_smiles_fps_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_confs", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.id_smiles_fps_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    
    preprocess(args) 
    process(args) # requires input, output, n_confs
    postprocess(args) # requires input, output

main()