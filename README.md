# AIMetry

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate aimetry
```

This also installs `aimnet` from PyPI and `aimnet2calc` from the `aimnet2calc` branch of the AIMNet2 repository. To install `aimnet2calc` into an existing environment:

```bash
pip install "git+https://github.com/isayevlab/AIMNet2.git@aimnet2calc"
```

The pretrained AIMNet2 model `aimnet2_b973c_0.jpt` must be in the repo root, where `model.py` loads it.
