### data used in visualization.ipynb
### source: https://arxiv.org/pdf/2006.13655
### data: https://www.repository.cam.ac.uk/items/ac362ef1-fa35-4bd4-8b56-2fde6d7b1d2c

import numpy as np
import torch

from ase.io import iread
from tqdm import tqdm

all_positions, all_numbers, all_charges, all_names, all_formulas = [], [], [], [], []
file_path = "FILE_PATH"
out_dir = "OUT_DIR"

for atoms in tqdm(iread(file_path, index=':')):
    all_positions.append(atoms.get_positions())
    all_numbers.append(atoms.get_atomic_numbers())
    all_charges.append(0)

    # "name" -- whatever label is in the comment line
    all_names.append(atoms.info.get('config_type'))

    # chemical formula derived from atomic composition, e.g. "C64"
    all_formulas.append(atoms.get_chemical_formula())

result_dict = {}
for i in range(len(all_positions)):
    name, formula = all_names[i], all_formulas[i]
    if (name, formula) not in result_dict: result_dict[(name, formula)] = {"coord": [], "numbers": [], "charge": []}
    result_dict[name, formula]["coord"].append(all_positions[i])
    result_dict[name, formula]["numbers"].append(all_numbers[i])
    result_dict[name, formula]["charge"].append(all_charges[i])

for key in result_dict:
    name, formula = key
    if name == "C": name = "C1"
    coord = torch.tensor(np.array(result_dict[key]["coord"])).unsqueeze(1)
    numbers = torch.tensor(np.array(result_dict[key]["numbers"]))
    charge = torch.tensor(np.array(result_dict[key]["charge"]))
    torch.save({"coord": coord, "numbers": numbers, "charge":charge}, f"{out_dir}/{name}_{formula}.pt")