from model import EncoderModel
from dataset import *
import os, re
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn import Linear, Sequential, ReLU, GELU
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, random_split, Subset
from torch_optimizer import Lamb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
import wandb
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score as prc_auc_score
from rdkit.Chem.Scaffolds import MurckoScaffold
from functools import partial
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

class Normalizer(object):
    """Normalize a Tensor and restore it later.
    code modified from: https://www.nature.com/articles/s42256-022-00447-x
     """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)
        if self.mean.ndim == 0: 
            self.mean = torch.tensor([self.mean])
            self.std = torch.tensor([self.std])
        print(self.mean, self.std)
        assert self.mean.shape[0] == self.std.shape[0]
        assert self.mean.ndim == self.std.ndim

    def norm(self, tensor, index=None):
        if index==None:
            assert tensor.shape[1] == self.mean.shape[0], (tensor.shape[1], self.mean.shape[0])
            return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)
        else:
            return (tensor - self.mean[index].to(tensor.device)) / self.std[index].to(tensor.device)

    def denorm(self, normed_tensor, index=None):
        if index==None:
            assert normed_tensor.shape[1] == self.mean.shape[0], (normed_tensor.shape[1], self.mean.shape[0])
            return normed_tensor * self.std.to(normed_tensor.device) + self.mean.to(normed_tensor.device)
        else:
            return normed_tensor * self.std[index].to(normed_tensor.device) + self.mean[index].to(normed_tensor.device)

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def calculate_weighted_BCE(train_loader, device=None, target_idx=None):
    """Calculates pos_weight across a train DataLoader and returns an nn.BCEWithLogitsLoss instance."""
    all_targets = []
    
    for batch in train_loader:
        # Handles both object batches (batch.target) and tuple batches (embeddings, targets)
        targets = batch.target if hasattr(batch, 'target') else batch[1]
        all_targets.append(targets.detach().cpu())
        
    targets = torch.cat(all_targets, dim=0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    if target_idx is not None and targets.shape[1] > 1:
        targets = targets[:, target_idx:target_idx + 1]
    # Ignore NaNs automatically by explicitly matching 1s and 0s
    num_pos = torch.sum(targets == 1, dim=0).float()
    num_neg = torch.sum(targets == 0, dim=0).float()
    pos_weight = num_neg / torch.clamp(num_pos, min=1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    return loss_fn.to(device)

def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))

def load_id_to_smiles(csv_path, smiles_col, sep=','):
    df = pd.read_csv(csv_path, sep=sep, header=None).values
    smiles_col_data = df[1:, smiles_col]
    return {i: smi for i, smi in enumerate(smiles_col_data)}

def get_all_smiles(dataset, id_to_smiles):
    if isinstance(dataset, ConformerEdgeDataset):
        import bisect
        smiles_list = []
        for idx in range(len(dataset)):
            file_idx = bisect.bisect_right(dataset.aggregate_lengths[1:], idx)
            mol_idx = idx - dataset.aggregate_lengths[file_idx]
            mol_id = dataset.data[file_idx]["id"][mol_idx].item()
            smiles_list.append(id_to_smiles[mol_id])
        return smiles_list
    else:
        smiles_list = []
        for idx in range(len(dataset)):
            mol_id = dataset.ids[idx].item()
            smiles_list.append(id_to_smiles[mol_id])
        return smiles_list

    
def generate_scaffold(smiles, include_chirality=True):
    # FH-GNN uses includeChirality=True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(dataset, id_to_smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=1):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    smiles_list = get_all_smiles(dataset, id_to_smiles)

    scaffolds = defaultdict(list)  # insertion-ordered, same as the original
    for idx, smi in enumerate(smiles_list):
        scaffolds[generate_scaffold(smi, include_chirality=True)].append(idx)
    scaffold_sets = list(scaffolds.values())

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(scaffold_sets))

    n_total = len(dataset)
    n_total_valid = int(np.floor(frac_valid * n_total))
    n_total_test = int(np.floor(frac_test * n_total))

    train_idx, valid_idx, test_idx = [], [], []
    for i in order:
        group = scaffold_sets[i]
        if len(valid_idx) + len(group) <= n_total_valid:
            valid_idx.extend(group)
        elif len(test_idx) + len(group) <= n_total_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    if len(valid_idx) == 0 or len(test_idx) == 0:
        warnings.warn(f"Scaffold split produced an empty valid ({len(valid_idx)}) or test "
                      f"({len(test_idx)}) set; the dataset may be too small or dominated by one scaffold.")

    print(f"scaffold_split (seed={seed}): train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")
    return Subset(dataset, train_idx), Subset(dataset, valid_idx), Subset(dataset, test_idx)

def move_batch_to_device(batch, device):
    if isinstance(batch, (tuple, list)):
        return type(batch)(
            x.to(device) if hasattr(x, "to") else x
            for x in batch
        )
    return batch.to(device)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")

def best_model_path(mode, data_path, model_name, seed, target_name=None):
    """One file per (mode, task, model, seed, target) so runs don't overwrite each other."""
    task = os.path.basename(os.path.normpath(data_path))
    name = f"best_{mode}_{task}_{model_name}_seed{seed}" + (f"_{target_name}" if target_name is not None else "")
    return os.path.join(SAVE_DIR, re.sub(r"[^\w.+-]", "_", name) + ".pt")

def save_best_model(path, head, epoch, val_loss, hyperparams, target_names, normalizer=None, encoder_model=None):
    """Checkpoint at the lowest validation loss so far.
    normalizer is stored because regression heads predict normalized targets.
    encoder_model (finetune only) is stored in train.py's checkpoint format, so EncoderModel.load_model(path)
    loads the finetuned encoder directly; frozen runs don't change the encoder, so it isn't stored."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = dict(head_state_dict=head.state_dict(), epoch=epoch, val_loss=val_loss, hyperparams=hyperparams,
                target_names=target_names, normalizer=normalizer.state_dict() if normalizer is not None else None)
    if encoder_model is not None:
        ckpt |= dict(args=encoder_model.args,
                     initializer=encoder_model.initializer.__class__.__name__,
                     encoder=encoder_model.encoder.__class__.__name__,
                     aggregator=encoder_model.aggregator.__class__.__name__,
                     evaluator=encoder_model.evaluator.__class__.__name__,
                     model_state_dict=encoder_model.state_dict())
    torch.save(ckpt, path)

class FrozenBenchmarkDataset(Dataset):
    def __init__(self, model, data_dir, device='cuda:0'):
        dataset = ConformerEdgeDataset(data_dir, num_conformers=1, sample_per_batch=1, use_molecule_properties=True)
        loader = DataLoader(dataset, 64, False, num_workers=4, collate_fn=ConformerEdge_collate_fn, pin_memory=True)
        embeds = []
        self.target = []
        self.ids = []  # NEW: record ids in the same order embeddings/targets are collected
        for batch in tqdm(loader, desc='Generating Embeddings'):
            embeds.append(model.encode(batch.to(device)).to('cpu'))
            self.target.append(batch.target.clone())
            self.ids.append(batch.smiles.clone().to('cpu'))  # "smiles" field actually holds the molecule id, see ConformerEdgeDataset
        self.embeddings = torch.concat(embeds).detach().to('cpu')
        self.target = torch.concat(self.target, dim=0).detach().to("cpu")
        self.ids = torch.concat(self.ids, dim=0).detach().to("cpu")
        if len(self.target.shape) < 2: self.target = self.target.unsqueeze(1)
        self.target_size = self.target.shape[1]
        self.molecular_embedding_size = self.embeddings.shape[1]

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, index):
        return (self.embeddings[index], self.target[index])

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, index):
        return (self.embeddings[index], self.target[index])

def smiles2fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=256)
    bv = mfpgen.GetFingerprint(mol)
    # bv = MACCSkeys.GenMACCSKeys(mol)
    arr = torch.tensor(bv.ToList(), device="cpu")
    return arr

class ECFPDataset(Dataset):
    def __init__(self, data_dir, id_to_smiles, device=None):
        dataset = ConformerEdgeDataset(data_dir, num_conformers=1, sample_per_batch=1, use_molecule_properties=True)
        loader = DataLoader(dataset, 64, False, num_workers=0, collate_fn=ConformerEdge_collate_fn, pin_memory=False)
        embeds = []
        self.target = []
        self.ids = []  # NEW: record ids in the same order embeddings/targets are collected
        for batch in tqdm(loader, desc='Generating Embeddings'):
            self.target.append(batch.target)
            self.ids.append(batch.smiles.to('cpu'))  # "smiles" field actually holds the molecule id, see ConformerEdgeDataset
            id_list = batch.smiles.to('cpu').tolist()
            smiles_list = [id_to_smiles[i] for i in id_list]
            embeds.append(torch.stack([smiles2fingerprint(smiles) for smiles in smiles_list]))
        self.embeddings = torch.concat(embeds).detach().to('cpu').to(torch.float)
        self.target = torch.concat(self.target, dim=0).detach().to("cpu")
        self.ids = torch.concat(self.ids, dim=0).detach().to("cpu")
        if len(self.target.shape) < 2: self.target = self.target.unsqueeze(1)
        self.target_size = self.target.shape[1]
        self.molecular_embedding_size = self.embeddings.shape[1]

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, index):
        return (self.embeddings[index], self.target[index])

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, index):
        return (self.embeddings[index], self.target[index])


class CombinedModel(nn.Module):
    """Wraps encoder + prediction head so a single forward(batch) call, given
    a raw ConformerEdgeDataset batch, produces predictions end-to-end -- this
    is what lets gradients flow back into the encoder for finetuning."""
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch):
        embeddings, _, _ = self.encoder._encode_batch(batch)
        return self.head(embeddings)


# ---------- pred/target extraction, one per data source ----------
# both return (pred, target), NOT loss/score directly, so train_one_epoch/
# evaluate_one_epoch/test can stay identical across the frozen and finetune paths

def frozen_pred_target(model, batch):
    embeddings, targets = batch
    targets = targets.float()
    pred = model(embeddings)
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    assert pred.shape == targets.shape, (pred.shape, targets.shape)
    return pred, targets

def finetune_pred_target(model, batch, target_idx=None):
    pred = model(batch)
    targets = batch.target.float()
    if target_idx is not None:
        targets = targets[:, target_idx:target_idx+1]
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    assert pred.shape == targets.shape, (pred.shape, targets.shape)
    return pred, targets

def model_batch_loss(model, batch, loss_function, pred_target_fn, normalizer=None, index=None):
    pred, targets = pred_target_fn(model, batch)
    if normalizer: 
        targets = normalizer.norm(targets, index=index)
    
    valid_mask = ~torch.isnan(targets)
    targets = targets.masked_fill(~valid_mask, 0.0)
    loss_tensor = loss_function(pred, targets)  # Shape: (batch_size, num_tasks)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
    return loss_tensor[valid_mask].mean()

def frozen_collate_fn(batch):
    embeddings = torch.stack([x[0] for x in batch])
    targets = torch.stack([x[1] for x in batch])
    return embeddings, targets

def frozen_benchmark(encoder_model, data_path, loss_function, test_function, hyperparams, model_name, target_names, normalize, task_batch_size,
                     id_to_smiles=None, use_scaffold_split=False, target_idx=None, smiles_col=None):
    dataset = FrozenBenchmarkDataset(encoder_model, data_path, device=encoder_model.device)
    # dataset = ECFPDataset(data_path, id_to_smiles, )
    if target_idx is not None:
        dataset.target = dataset.target[:, target_idx:target_idx+1]
        dataset.target_size = 1 
        target_names = [target_names[target_idx]]
        target_idx = 0 # this has to be done otherwise the normalizer will go out of bounds
    molecular_embedding_size = dataset.molecular_embedding_size
    target_size = dataset.target_size
    # model = Sequential(
    #             Linear(molecular_embedding_size, molecular_embedding_size // 2),
    #             nn.LayerNorm(molecular_embedding_size // 2),
    #             GELU(),
    #             nn.Dropout(0.2),
    #             Linear(molecular_embedding_size // 2, target_size)
    #         ).to(encoder_model.device)
    model = Sequential( # for probe model
                Linear(molecular_embedding_size, target_size)
            ).to(encoder_model.device)
    if use_scaffold_split:
        assert id_to_smiles is not None, "id_to_smiles is required for scaffold_split"
        train_dataset, val_dataset, test_dataset = scaffold_split(dataset, id_to_smiles, seed=hyperparams["seed"])
    else:
        generator = torch.Generator().manual_seed(hyperparams["seed"])
        train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=task_batch_size, collate_fn=frozen_collate_fn,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=task_batch_size, collate_fn=frozen_collate_fn,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=task_batch_size, collate_fn=frozen_collate_fn,
                              shuffle=True, num_workers=4, pin_memory=True)
    # an extra edge case for using weighted BCE: for the train dataset, set a positive weight BCE:
    if loss_function == "weighted_BCE":
        loss_function = calculate_weighted_BCE(train_loader, device=encoder_model.device, target_idx=target_idx)

    normalizer = None
    if normalize:
        labels = []
        for batch in train_loader:
            _, targets = batch
            labels.append(targets.to("cpu"))
        normalizer = Normalizer(torch.concat(labels))


    optimizer = AdamW([{'params': model.parameters(), 'lr': hyperparams["model_lr"], "weight_decay":1e-4}])
    mode="triangular"
    # if isinstance(loss_function, nn.BCEWithLogitsLoss):
    #     mode="triangular2"
    # else:
    #     mode="triangular"
    # print(mode)
    scheduler = CyclicLR(optimizer, base_lr=hyperparams["model_lr"], max_lr=hyperparams["model_lr"] * 8,
                          step_size_up=len(train_loader) * 8, mode=mode, cycle_momentum=False)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=15)

    loss_fn = partial(model_batch_loss, loss_function=loss_function, pred_target_fn=frozen_pred_target)
    device = encoder_model.device
    wandb.init(
        project="PROJECT_NAME",
        name=f"frozen_{data_path.split('/')[-1]}_{model_name}" + (f"_{target_names[0]}" if target_idx is not None else "")
    )
    wandb.log({"seed": hyperparams["seed"]})
    best_loss = 1e10
    best_test = None
    for epoch in range(hyperparams["num_epochs"]):
        print(f"Epoch {epoch + 1} of {hyperparams['num_epochs']}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, normalizer=normalizer, index=target_idx)
        val_loss = evaluate_one_epoch(model, val_loader, loss_fn, device, normalizer=normalizer, index=target_idx)
        if val_loss < best_loss:
            best_loss = val_loss
            save_best_model(best_model_path("frozen", data_path, model_name, hyperparams["seed"], target_names[0] if target_idx is not None else None),
                            model, epoch + 1, val_loss, hyperparams, target_names, normalizer)
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Current LR: {optimizer.param_groups[0]['lr']}")
        test_scores, num_valid_elems = test(model, test_loader, test_function, device, frozen_pred_target, normalizer=normalizer, index=target_idx)
        assert len(test_scores) == len(target_names) == len(num_valid_elems)
        nan_filter = ~np.isnan(test_scores)
        test_score = np.average(test_scores[nan_filter])
        # test_score = np.average(test_scores[nan_filter], weights=num_valid_elems[nan_filter])
        named_scores = {}
        for i, score in enumerate(test_scores): named_scores[target_names[i]] = score
        print(f"test score after {epoch + 1} epochs: {test_score}")
        if best_test is None or (test_score > best_test and test_function == roc_auc_score) or (test_score < best_test and test_function in [mean_absolute_error, rmse]):
            best_test = test_score
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "average_test_score": test_score,
            "best_val_loss": best_loss,
            "best_test": best_test,
            "head_lr": optimizer.param_groups[0]["lr"],
        } | named_scores)
        # scheduler.step(val_loss)
    wandb.finish()
    return best_test


def finetune_benchmark(encoder_path, data_path, loss_function, test_function, hyperparams, model_name, target_names, normalize, device, task_batch_size,
                        id_to_smiles=None, use_scaffold_split=False, target_idx=None):
    """
    Unlike frozen_benchmark (which pre-computes embeddings once and freezes the
    encoder), this rebuilds embeddings on every forward pass so encoder
    gradients flow -- more expensive per step, but lets the encoder adapt to
    the downstream task rather than treating its representations as fixed.
    """
    dataset = ConformerEdgeDataset(data_path, num_conformers=1, sample_per_batch=1, use_molecule_properties=True)

    if use_scaffold_split:
        assert id_to_smiles is not None, "id_to_smiles is required for scaffold_split"
        train_dataset, val_dataset, test_dataset = scaffold_split(dataset, id_to_smiles, seed=hyperparams["seed"])
    else:
        generator = torch.Generator().manual_seed(hyperparams["seed"])
        train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=task_batch_size, collate_fn=ConformerEdge_collate_fn,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=task_batch_size, collate_fn=ConformerEdge_collate_fn,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=task_batch_size, collate_fn=ConformerEdge_collate_fn,
                              shuffle=True, num_workers=4, pin_memory=True)
    # an extra edge case for using weighted BCE: for the train dataset, set a positive weight BCE:
    if loss_function == "weighted_BCE":
        loss_function = calculate_weighted_BCE(train_loader, device=device, target_idx=target_idx)
    
    normalizer = None
    if normalize:
        labels = []
        for batch in train_loader:
            targets = batch.target
            labels.append(targets.to("cpu"))
        normalizer = Normalizer(torch.concat(labels))

    encoder_model = EncoderModel.load_model(encoder_path, device=device).eval()

    wandb.init(
        project="PROJECT_NAME",
        name=f"finetune_{data_path.split('/')[-1]}_{model_name}" + (f"_{target_names[target_idx]}" if target_idx is not None else "")
    )
    wandb.log({"seed": hyperparams["seed"]})
    # infer embedding/target size from one real batch, rather than assuming
    # attributes exist on encoder_model
    probe_loader = DataLoader(dataset, batch_size=4, collate_fn=ConformerEdge_collate_fn)
    probe_batch = next(iter(probe_loader)).to(device)
    with torch.no_grad():
        probe_embed, _, _ = encoder_model._encode_batch(probe_batch)
    molecular_embedding_size = probe_embed.shape[1]
    target_size = probe_batch.target.shape[1] if probe_batch.target.ndim > 1 else 1
    if target_idx is not None:
        target_size = 1
        target_names = [target_names[target_idx]]
        # target_idx = 0 # this has to be done otherwise the normalizer will go out of bounds xd

    for param in encoder_model.parameters():
        param.requires_grad = True  # unfreeze for finetuning

    head = Sequential(
                Linear(molecular_embedding_size, molecular_embedding_size // 2),
                nn.LayerNorm(molecular_embedding_size // 2),
                GELU(),
                nn.Dropout(0.2),
                Linear(molecular_embedding_size // 2, target_size)
            ).to(encoder_model.device)
    # warm start from the frozen run's best head:
    # head.load_state_dict(torch.load(best_model_path("frozen", data_path, model_name, hyperparams["seed"], target_names[0] if target_idx is not None else None))["head_state_dict"])
    model = CombinedModel(encoder_model, head).to(device)

    encoder_lr = hyperparams.get("encoder_lr", hyperparams["model_lr"] * 0.01)
    optimizer = AdamW([
        {'params': encoder_model.parameters(), 'lr': encoder_lr, "weight_decay":1e-4},
        {'params': head.parameters(), 'lr': hyperparams["model_lr"], "weight_decay":1e-4},
    ])
    mode="triangular"
    # if isinstance(loss_function, nn.BCEWithLogitsLoss):
    #     mode="triangular2"
    # else:
    #     mode="triangular"
    # print(mode)
    scheduler = CyclicLR(optimizer,
                          base_lr=[encoder_lr, hyperparams["model_lr"]],
                          max_lr=[encoder_lr * 10, hyperparams["model_lr"] * 8],
                          step_size_up=len(train_loader) * 8, mode=mode, cycle_momentum=False)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=15)

    loss_fn = partial(model_batch_loss, loss_function=loss_function,
                       pred_target_fn=partial(finetune_pred_target, target_idx=target_idx))

    best_loss = 1e10
    best_test = None
    for epoch in range(hyperparams["num_epochs"]):
        print(f"Epoch {epoch + 1} of {hyperparams['num_epochs']}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, normalizer=normalizer, index=target_idx)
        val_loss = evaluate_one_epoch(model, val_loader, loss_fn, device, normalizer=normalizer, index=target_idx)
        if val_loss < best_loss:
            best_loss = val_loss
            save_best_model(best_model_path("finetune", data_path, model_name, hyperparams["seed"], target_names[0] if target_idx is not None else None),
                            head, epoch + 1, val_loss, hyperparams, target_names, normalizer, encoder_model=encoder_model)
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Encoder LR: {optimizer.param_groups[0]['lr']}, Head LR: {optimizer.param_groups[1]['lr']}")
        test_scores, num_valid_elems = test(model, test_loader, test_function, device,
                                         partial(finetune_pred_target, target_idx=target_idx),
                                         normalizer=normalizer, index=target_idx)
        assert len(test_scores) == len(target_names) == len(num_valid_elems)
        nan_filter = ~np.isnan(test_scores)
        test_score = np.average(test_scores[nan_filter])
        # test_score = np.average(test_scores[nan_filter], weights=num_valid_elems[nan_filter])
        named_scores = {}
        for i, score in enumerate(test_scores): named_scores[target_names[i]] = score
        print(f"test score after {epoch + 1} epochs: {test_score}")
        if best_test is None or (test_score > best_test and test_function == roc_auc_score) or (test_score < best_test and test_function in [mean_absolute_error, rmse]):
            best_test = test_score
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "average_test_score": test_score,
            "best_val_loss": best_loss,
            "best_test": best_test,
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]['lr']
        } | named_scores)
        # scheduler.step(val_loss)
    wandb.finish()
    return best_test


def train_one_epoch(model, train_loader, model_batch_loss, optimizer, scheduler, device, normalizer=None, index=None):
    model.train()
    running_loss = 0.0
    epoch = tqdm(train_loader)
    for i, batch in enumerate(epoch):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        loss = model_batch_loss(model, batch, normalizer=normalizer, index=index)
        running_loss += loss.item()
        epoch.set_postfix(
            avg_loss=running_loss / (i + 1),
            loss=loss.item()
        )
        loss.backward()
        optimizer.step()
        scheduler.step() # for cooldown, i commented this out

    return running_loss / len(train_loader)

@torch.inference_mode()
def evaluate_one_epoch(model, val_loader, model_batch_loss, device, normalizer=None, index=None):
    model.eval()
    running_loss = 0.0
    epoch = tqdm(val_loader)
    for i, batch in enumerate(epoch):
        batch = move_batch_to_device(batch, device)
        loss = model_batch_loss(model, batch, normalizer=normalizer, index=index)
        running_loss += loss.item()
        epoch.set_postfix(dict(avg_loss=running_loss / (i + 1), loss=loss.item()))
    return running_loss / len(val_loader)

@torch.inference_mode()
def test(model, test_loader, test_function, device, pred_target_fn, normalizer=None, index=None):
    model.eval()
    predictions = []
    targets = []
    epoch = tqdm(test_loader, desc="Testing")
    for batch in epoch:
        batch = move_batch_to_device(batch, device)
        pred, batch_targets = pred_target_fn(model, batch)
        if normalizer: pred = normalizer.denorm(pred, index=index)
        # valid = ~torch.isnan(batch_targets)
        if pred.ndim == 1: pred = pred.unsqueeze(1)
        if batch_targets.ndim == 1: batch_targets = batch_targets.unsqueeze(1)
        predictions.append(pred.detach().cpu())
        targets.append(batch_targets.detach().cpu())
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    scores = []
    num_valid_targets = []
    for i in range(predictions.shape[1]):
        valid_i = ~torch.isnan(targets[:, i])
        true_targets = targets[valid_i, i]
        num_valid_targets.append(true_targets.shape[0])
        if test_function == roc_auc_score and true_targets.sum().item() == 0 and len(true_targets.unique()) < 2: scores.append(float("nan"))
        else: scores.append(test_function(targets[valid_i, i].numpy(), predictions[valid_i, i].numpy()))

    return np.array(scores), np.array(num_valid_targets)

# single_target_tasks = {"qm9"} # qm7 is already a single task
single_target_tasks = {}
def frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, hyperparams, model_name, batch_sizes):
    assert len(tasks) == len(loss_functions) == len(test_functions)
    for i in range(len(tasks)):
        dataset_csv_path = f"{moleculenet_dir}/{tasks[i]}/{tasks[i]}.csv"
        df = pd.read_csv(dataset_csv_path)
        target_names = df.columns.tolist()[task_to_target_names_cols[tasks[i]]]
        if type(target_names) == str: target_names = [target_names]
        id_to_smiles = load_id_to_smiles(dataset_csv_path, task_to_smilescol[tasks[i]])
        regression_tasks = {"esol", "lipo", "freesol", "qm7", "qm8", "qm9"}
        # regression_tasks = {}
        normalize = tasks[i] in regression_tasks
        task_batch_size = batch_sizes[tasks[i]]
        if tasks[i] in single_target_tasks:
            for target_idx in range(len(target_names)):
                frozen_benchmark(encoder_model, f"{moleculenet_dir}/{tasks[i]}", loss_functions[i], test_functions[i],
                                 hyperparams, model_name, target_names, normalize, task_batch_size,
                                 id_to_smiles=id_to_smiles, use_scaffold_split=task_to_scaffoldsplit[tasks[i]],
                                 target_idx=target_idx)
        else:
            frozen_benchmark(encoder_model, f"{moleculenet_dir}/{tasks[i]}", loss_functions[i], test_functions[i],
                             hyperparams, model_name, target_names, normalize, task_batch_size,
                            id_to_smiles=id_to_smiles, use_scaffold_split=task_to_scaffoldsplit[tasks[i]])

def finetune_ablation(encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, hyperparams, model_name, device, batch_sizes):
    assert len(tasks) == len(loss_functions) == len(test_functions)
    for i in range(len(tasks)):
        dataset_csv_path = f"{moleculenet_dir}/{tasks[i]}/{tasks[i]}.csv"
        df = pd.read_csv(dataset_csv_path)
        target_names = df.columns.tolist()[task_to_target_names_cols[tasks[i]]]
        if type(target_names) == str: target_names = [target_names]
        id_to_smiles = load_id_to_smiles(dataset_csv_path, task_to_smilescol[tasks[i]])
        regression_tasks = {"esol", "lipo", "freesol", "qm7", "qm8", "qm9"}
        # regression_tasks = {}
        task_batch_size = batch_sizes[tasks[i]]
        normalize = tasks[i] in regression_tasks
        if tasks[i] in single_target_tasks:
            for target_idx in range(len(target_names)):
                finetune_benchmark(encoder_path, f"{moleculenet_dir}/{tasks[i]}", loss_functions[i], test_functions[i],
                                 hyperparams, model_name, target_names, normalize, device, task_batch_size,
                                 id_to_smiles=id_to_smiles, use_scaffold_split=task_to_scaffoldsplit[tasks[i]],
                                 target_idx=target_idx)
        else:
            finetune_benchmark(encoder_path, f"{moleculenet_dir}/{tasks[i]}", loss_functions[i], test_functions[i],
                             hyperparams, model_name, target_names, normalize, device, task_batch_size,
                            id_to_smiles=id_to_smiles, use_scaffold_split=task_to_scaffoldsplit[tasks[i]])

if __name__ == '__main__':
    task_to_smilescol = {
        "bbbp":     3, "tox21":   13, "toxcast":  0, "sider":    0,
        "clintox":  0, "muv":     -1, "bace":     0, "hiv":      0,
        "esol":     9, "lipo":     2, "freesol":  1, "qm7":      0,
        "qm8":      0, "qm9":      0,
    }
    task_to_scaffoldsplit = {
        "bbbp":    True, "tox21":   True, "toxcast": True, "sider":   True,
        "clintox": True, "muv":     True, "bace":    True, "hiv":     True,
        "esol":    True, "lipo":    True, "freesol": True, "qm7":     False,
        "qm8":    False, "qm9":    False,
    }
    task_to_target_names_cols = {
        "bbbp": 2, "tox21": slice(0, 12), "toxcast": slice(1, None), # 617
        "sider": slice(1, None), # 27
        "clintox": slice(1, None), "muv": slice(0, -2), "bace": 2,
        "hiv": 2, "esol": 8, "lipo": 1, "freesol": 2, "qm7": 1,
        "qm8": slice(1, None), "qm9": slice(1, None)
    }
    frozen_task_to_batch_size = { 
        "bbbp": 32, "tox21": 128, "toxcast": 128, "sider": 32, 
        "clintox": 32, "muv": 256, "bace": 32, "hiv": 256,
        "esol": 32, "lipo": 32, "freesol": 16, "qm7": 128,
        "qm8": 256,"qm9": 256}
    finetune_task_to_batch_size = { 
        "bbbp": 32, "tox21": 128, "toxcast": 128, "sider": 32, 
        "clintox": 32, "muv": 128, "bace": 32, "hiv": 128,
        "esol": 32, "lipo": 32, "freesol": 16, "qm7": 128,
        "qm8": 128,"qm9": 128}
    
    moleculenet_dir = "MOLECULENET_DIR"
    loss_function_classification = nn.BCEWithLogitsLoss(reduction="none")
    loss_function_regression = nn.MSELoss(reduction="none")
    test_regression = mean_absolute_error

    classification_tasks = ["bbbp", "bace", "sider", "tox21", "hiv", "clintox", "muv", "toxcast"]
    classification_tasks = ["muv"]
    classification_loss_functions = [loss_function_classification] 
    classification_test_functions = [roc_auc_score]
    
    regression1_tasks = ["lipo"]
    regression1_loss_functions = [loss_function_regression]
    regression1_test_functions = [rmse]

    regression2_tasks = ["qm7", "qm8", "qm9"]
    regression2_loss_functions = [nn.L1Loss(reduction="none")] * 3
    regression2_test_functions = [test_regression] * 3

    tasks = classification_tasks
    loss_functions = classification_loss_functions
    test_functions = classification_test_functions
    # tasks = classification_tasks + regression1_tasks
    # loss_functions = classification_loss_functions + regression1_loss_functions
    # test_functions = classification_test_functions + regression1_test_functions

    device = "cuda:1"
    ANT_encoder_path = "ANT_SAVE_PATH"
    NT_encoder_path = "NT_SAVE_PATH"
    nopre_encoder_path = "NOPRE_SAVE_PATH"
    finetuning_epochs = 250
    seed = 1

    # encoder_model = Linear(1, 1) # filler model for ecfp
    # encoder_model.device = device
    # frozen benchmark (encoder weights fixed, embeddings pre-computed once)

    # encoder_model = EncoderModel.load_model(ANT_encoder_path, device=device).eval() # asymmetry here because encoder_model needs to be loaded every time for finetune
    # frozen_hyperparams = {"model_lr": 1e-4, "num_epochs": finetuning_epochs, "seed": 0}
    # frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, frozen_hyperparams, "ANT_probe", frozen_task_to_batch_size)
    # encoder_model = EncoderModel.load_model(ANT_encoder_path, device=device).eval() # asymmetry here because encoder_model needs to be loaded every time for finetune
    # frozen_hyperparams = {"model_lr": 1e-4, "num_epochs": finetuning_epochs, "seed": 1}
    # frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, frozen_hyperparams, "ANT_probe", frozen_task_to_batch_size)
    # encoder_model = EncoderModel.load_model(ANT_encoder_path, device=device).eval() # asymmetry here because encoder_model needs to be loaded every time for finetune
    # frozen_hyperparams = {"model_lr": 1e-4, "num_epochs": finetuning_epochs, "seed": 2}
    # frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, frozen_hyperparams, "ANT_probe", frozen_task_to_batch_size)

    # finetune benchmark (encoder weights updated jointly with the head)
    # finetune_hyperparams = {"model_lr": 1e-6, "encoder_lr": 1e-6, "num_epochs": finetuning_epochs, "seed": 0}
    # finetune_ablation(ANT_encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, finetune_hyperparams, "ANT", device, finetune_task_to_batch_size)
    # finetune_hyperparams = {"model_lr": 1e-6, "encoder_lr": 1e-6, "num_epochs": finetuning_epochs, "seed": 1}
    # finetune_ablation(ANT_encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, finetune_hyperparams, "ANT", device, finetune_task_to_batch_size)
    # finetune_hyperparams = {"model_lr": 1e-6, "encoder_lr": 1e-6, "num_epochs": finetuning_epochs, "seed": 2}
    # finetune_ablation(ANT_encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, finetune_hyperparams, "ANT", device, finetune_task_to_batch_size)


    # frozen benchmark (encoder weights fixed, embeddings pre-computed once)
    # encoder_model = EncoderModel.load_model(NT_encoder_path, device=device).eval()
    # frozen_hyperparams = {"model_lr": 1e-4, "num_epochs": finetuning_epochs, "seed": 0}
    # frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, frozen_hyperparams, "NT_probe", frozen_task_to_batch_size)
    # encoder_model = EncoderModel.load_model(NT_encoder_path, device=device).eval()
    # frozen_hyperparams = {"model_lr": 1e-4, "num_epochs": finetuning_epochs, "seed": 1}
    # frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, frozen_hyperparams, "NT_probe", frozen_task_to_batch_size)
    # encoder_model = EncoderModel.load_model(NT_encoder_path, device=device).eval()
    # frozen_hyperparams = {"model_lr": 1e-4, "num_epochs": finetuning_epochs, "seed": 2}
    # frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, frozen_hyperparams, "NT_probe", frozen_task_to_batch_size)

    # finetune benchmark (encoder weights updated jointly with the head)
    # finetune_hyperparams = {"model_lr": 1e-6, "encoder_lr": 1e-6, "num_epochs": finetuning_epochs, "seed": 0}
    # finetune_ablation(NT_encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, finetune_hyperparams, "NT", device, finetune_task_to_batch_size)
    # finetune_hyperparams = {"model_lr": 1e-6, "encoder_lr": 1e-6, "num_epochs": finetuning_epochs, "seed": 1}
    # finetune_ablation(NT_encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, finetune_hyperparams, "NT", device, finetune_task_to_batch_size)
    # finetune_hyperparams = {"model_lr": 1e-6, "encoder_lr": 1e-6, "num_epochs": finetuning_epochs, "seed": 2}
    # finetune_ablation(NT_encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, finetune_hyperparams, "NT", device, finetune_task_to_batch_size)


    # frozen benchmark (encoder weights fixed, embeddings pre-computed once)
    # encoder_model = EncoderModel.load_model(nopre_encoder_path, device=device).eval()
    # frozen_hyperparams = {"model_lr": 1e-4, "num_epochs": finetuning_epochs, "seed": 0}
    # frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, frozen_hyperparams, "nopre_probe", frozen_task_to_batch_size)
    # encoder_model = EncoderModel.load_model(nopre_encoder_path, device=device).eval()
    # frozen_hyperparams = {"model_lr": 1e-4, "num_epochs": finetuning_epochs, "seed": 1}
    # frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, frozen_hyperparams, "nopre_probe", frozen_task_to_batch_size)
    # encoder_model = EncoderModel.load_model(nopre_encoder_path, device=device).eval()
    # frozen_hyperparams = {"model_lr": 1e-4, "num_epochs": finetuning_epochs, "seed": 2}
    # frozen_ablation(encoder_model, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, frozen_hyperparams, "nopre_probe", frozen_task_to_batch_size)

    # finetune benchmark (encoder weights updated jointly with the head)
    # finetune_hyperparams = {"model_lr": 1e-6, "encoder_lr": 1e-6, "num_epochs": finetuning_epochs, "seed": 0}
    # finetune_ablation(nopre_encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, finetune_hyperparams, "nopre", device, finetune_task_to_batch_size)
    # finetune_hyperparams = {"model_lr": 1e-6, "encoder_lr": 1e-6, "num_epochs": finetuning_epochs, "seed": 1}
    # finetune_ablation(nopre_encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, finetune_hyperparams, "nopre", device, finetune_task_to_batch_size)
    finetune_hyperparams = {"model_lr": 1e-6, "encoder_lr": 1e-6, "num_epochs": finetuning_epochs, "seed": 2}
    finetune_ablation(nopre_encoder_path, moleculenet_dir, tasks, loss_functions, test_functions, task_to_smilescol, task_to_scaffoldsplit, task_to_target_names_cols, finetune_hyperparams, "nopre", device, finetune_task_to_batch_size)