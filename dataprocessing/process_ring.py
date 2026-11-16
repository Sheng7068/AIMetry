### data used in benchmark/benchmark_ring.py
### source: https://doi.org/10.1039/d5sc04079e
### data from: data for aimnet (below) is from lab directly, https://github.com/pkzheng94/ring is usable

import sys
import h5py
import torch
from tqdm import tqdm
 
 
def process_ring_h5(h5_path, save_dir):
 
    f = h5py.File(h5_path, "r")
    group_names = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))

    for gname in tqdm(group_names):
        g = f[gname]

        coord = torch.as_tensor(g["coord"][()], dtype=torch.float32) # [N, X, 3]
        numbers = torch.as_tensor(g["numbers"][()], dtype=torch.int) # [N, X]
        charge = torch.as_tensor(g["charge"][()]) # [N]
        nring = torch.as_tensor(g["nring"][()]) # [N]
        properties = torch.as_tensor(g["property"][()], dtype=torch.float32) # [N, 5]
    
        data = {
            "coord": coord.unsqueeze(1), # unsqueeze here to be compatible with ConformerEdgeDataset class
            "numbers": numbers,
            "charge": charge,
            "nring": nring,
            "gap": properties[:, 0],
            "ip": properties[:, 1],
            "ea": properties[:, 2],
            "eox": properties[:, 3],
            "ered": properties[:, 4],
        }

        torch.save(data, f"{save_dir}/{gname}.pt")

process_ring_h5("RING_TRAIN_DIR/train.h5", "RING_TRAIN_DIR")
process_ring_h5("RING_VAL_DIR/valid.h5", "RING_VAL_DIR")
process_ring_h5("RING_TEST_DIR/test.h5", "RING_TEST_DIR")