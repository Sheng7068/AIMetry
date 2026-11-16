import os, glob, bisect, random
from pathlib import Path
from itertools import chain

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
mmap = {'mmap': True} if torch.__version__ >= '2' else {}

def ConformerEdge_collate_fn(data_list_list):
    res = Batch.from_data_list(list(chain(*data_list_list)))
    del res.x
    return res

class ConformerEdgeDataset(Dataset):
    '''Dataset class used to load molecule data for training or visualization
    use_edges: bool on whether to include edge indices from dataset. Exists for chembl
    use_molecule_properties: bool on whether to include molecular level properties like substructure
            counting. Do not use in training, wastes GPU space
    num_conformers: int representing number of conformers per unique molecule
    sample_per_batch: int representing how many conformers to take per unique molecule
    data_dirs: directory with torch files of following attributes:
    (N = #unique molecules, M=#conformers per molecule, X=#atoms per molecule, F=fingerprint size
     E = #edges for N molecules in file, Y = number of attributes per edge,)
    
    atom level properties:
        coord (REQUIRED): [N, M, X, 3]
        numbers (REQUIRED): [N, X]
        charges (mutually required with charge, see blow): [N, X]
    edge level properties:
        edge_index: [E, 2]
        edge_attr: [E, Y]
        ptr: [E]
    molecule level properties:
        charge (unnecessary if charges exist, necessary otherwise): [N]
        id (REQUIRED for referencing which molecule): [N]
        fingerprint (REQUIRED for training): [N, F]
        other properties: [N,]
    '''
    def __init__(self, data_dirs, num_conformers=16, sample_per_batch=16, 
                 use_edges=False, use_molecule_properties=False, randomize=True, preload=True):
        self.num_conformers = num_conformers
        self.sample_per_batch = sample_per_batch
        self.use_edges = use_edges
        self.use_molecule_properties = use_molecule_properties
        self.randomize = randomize
        if isinstance(data_dirs, str): self.paths = list(sorted(glob.glob(os.path.join(data_dirs, '*.pt'))))
        else: self.paths = list(chain.from_iterable([sorted(glob.glob(os.path.join(data_dir, '*.pt'))) for data_dir in data_dirs]))

        self.aggregate_lengths = [0]
        self.section_id = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(self.paths)}
        self.data = []
        for path in self.paths:
            data = torch.load(path, weights_only=False, **({} if preload else mmap))
            self.data.append(data)
            num_molecules = data["coord"].shape[0]
            self.aggregate_lengths.append(num_molecules + self.aggregate_lengths[-1])
        
    def __len__(self):
        return self.aggregate_lengths[-1]

    def __getitem__(self, index):
        file_idx = bisect.bisect_right(self.aggregate_lengths[1:], index)
        mol_idx = (index - self.aggregate_lengths[file_idx])
        if self.randomize: conf_idxs = torch.randperm(self.num_conformers)[:self.sample_per_batch]
        else: conf_idxs = torch.arange(self.num_conformers)[:self.sample_per_batch]
        
        data = self.data[file_idx]
        res = []
        excluded_keys = {"numbers", "coord", "numbers", "charges", "edge_index", "edge_attr", "charge", "id", "fingerprint"}
        for conf_idx in conf_idxs:
            # molecule level labels. Tries to add id and fingerprint if they exist.
            # Adds all molecular properties gathered in postprocessing if those exist and are toggled
            # Tries to add charges, then charge, or 0 if those exist
            molecule_labels = {}
            if "id" in data: molecule_labels["smiles"] = data["id"][mol_idx]
            if "fingerprint" in data: molecule_labels["fingerprint"] = data["fingerprint"][mol_idx].unsqueeze(0)
            if self.use_molecule_properties: 
                for k, v in data.items():
                    if k in excluded_keys: continue
                    if len(v[mol_idx].shape) != 0: molecule_labels[k] = v[mol_idx].unsqueeze(0) # multilabel molecular property
                    else: molecule_labels[k] = v[mol_idx]
                # molecule_labels |= {k:v[mol_idx] for k, v in data.items() if k not in excluded_keys}
            if "charges" in data: 
                molecule_labels["charges"] = data["charges"][mol_idx]
                molecule_labels["charge"] = torch.sum(molecule_labels["charges"])
            elif "charge" in data: molecule_labels["charge"] = data["charge"][mol_idx]
            else: molecule_labels["charge"] = torch.tensor(0)

            edges = {}
            if self.use_edges: 
                assert "edge_index" in data, "dataset does not have edge_index to use as edges"
                edges["edge_index"] = data["edge_index"][:,data["ptr"][mol_idx]:data["ptr"][mol_idx+1]].to(torch.int64)
                if "edge_attr" in data: edges["edge_attr"] = data["edge_attr"][data["ptr"][mol_idx]:data["ptr"][mol_idx+1]].to(torch.int)

            res.append(Data(
                x           = data["numbers"][mol_idx],
                coord       = data["coord"][mol_idx, conf_idx],
                numbers     = data["numbers"][mol_idx],
                **molecule_labels,
                **edges))
        return res
    
    def toggle_use_edges(self):
        self.use_edges = not self.use_edges
        print(f"use_edges: {self.use_edges}")
    
    def toggle_use_molecule_properties(self):
        self.use_molecule_properties = not self.use_molecule_properties
        print(f"use_molecule_properties: {self.use_molecule_properties}")