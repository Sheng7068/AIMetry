import os
from train import parse_args, main
from model import *

# You can still add command line arguments
if __name__ == '__main__':
    args = parse_args({
        'wandb': False,
        'preload': True,
        'device': 'cuda:1',
        "num_workers": 4,
        'molecular_embedding_size': 256,
        "batch_confs_per_mol": 4,
        "batch_size": 32, # 16 conformer per = 256 conformers per batch
        "num_epochs": 100,
        "data_directory": "DATA_DIR"
        })
    if args["model_version"] == 0:
        args["save_best"] = "saves/AIMetry-ANT.pt"
        model = AIMNet2ECGPSMMLPANTXEncoder
    elif args["model_version"] == 1:
        args["batch_confs_per_mol"] = 2
        args["save_best"] = "saves/AIMetry-NT.pt"
        model = AIMNet2ECGPSMMLPNTXEncoder
    else:
        raise Exception("invalid version number")
    args["load_file"] = args["save_best"] if os.path.isfile(args["save_best"]) else None
    main(args, model)