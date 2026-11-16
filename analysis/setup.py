from pathlib import Path
import glob
import itertools
import math
import os
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from matplotlib import font_manager
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from tqdm import tqdm

SRC_DIR = Path("SRC_DIR")
DATA_ROOT = Path("DATA_ROOT")
DEVICE = "cuda"
FONT_PATH = Path("FONT_PATH")

CHEMBL_DIR = DATA_ROOT / "chembl/chembl_le128_rdkit"
SMILES_DIR = DATA_ROOT / "chembl/chembl_le128"
EMBEDDING_DIR = CHEMBL_DIR / "embeddings"
FINGERPRINT_DIR = CHEMBL_DIR / "fingerprints"
HAMMETT_PATH = DATA_ROOT / "hammett/hammett.csv"
DFT_CACHE = DATA_ROOT / "torsion/dft_energies.csv"
OUTPUT_DIR = SRC_DIR / "visualizations"

CARBON_DIRS = {
    "unit_cell": DATA_ROOT / "carbon",
    "periodic_5A": DATA_ROOT / "carbon_periodic_5A",
    "periodic_8A": DATA_ROOT / "carbon_periodic_8A",
}

CHECKPOINTS = {
    "ANT": SRC_DIR / "ANT_SAVE_PATH",
    "NT": SRC_DIR / "NT_SAVE_PATH",
    "nopre": SRC_DIR / "NOPRE_SAVE_PATH",
}

MODEL_TITLES = {
    "ANT": "Chemically Aware NT-Xent",
    "NT": "Normal NT-Xent",
    "nopre": "No Pretraining",
}

NOPRE_ARGS = {
    "atom_embedding_size": 256,
    "bond_features": 1,
    "molecular_embedding_size": 256,
    "num_transformer_layers": 4,
    "num_attention_heads": 4,
    "dropout_rate": 0.1,
    "batch_norm": False,
    "batch_size": 32,
    "batch_confs_per_mol": 4,
    "num_workers": 4,
    "device": DEVICE,
    "lr": 0.001,
    "num_epochs": 100,
    "early_stop_lr": 1e-8,
    "save_best": None,
    "save_last": None,
    "save_every": -1,
    "save_every_out": None,
    "wandb": False,
    "run_name": None,
    "project": "Contrastive-Encoder-Fullsize",
    "preload": True,
    "model_version": 0,
}

sys.path.insert(0, str(SRC_DIR))
os.chdir(SRC_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from dataset import *
from model import *
from utils import *

cm = 1 / 2.54

font_manager.fontManager.addfont(str(FONT_PATH))
matplotlib.rcParams.update({
    "font.family": font_manager.FontProperties(fname=str(FONT_PATH)).get_name(),
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.dpi": 600,
})
