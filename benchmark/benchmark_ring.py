import torch, wandb
from tqdm import tqdm

from torch.nn import Module, Linear, Sequential, ReLU
from torch.utils.data import DataLoader, random_split
from torch_optimizer import Lamb
from torch.optim import AdamW
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from model import EncoderModel
from sklearn.metrics import roc_auc_score
from dataset import *

# from model import MultiLayerPoolAIMNet2StructuralEncoder
# # zero shot model
# ENCODER = MultiLayerPoolAIMNet2StructuralEncoder(zero_shot_args).to(device).eval()
#    0, 1,2, 3, 4,       7,  8,  9,      11,    13,       16
# N = [1, 5, 6, 7, 8, 0, 0, 14, 15, 16, 0, 33, 0, 34, 0, 0, 53]
def train_one_epoch(encoder, model, labels, train_loader, optimizer, device, finetune=False, normalize=False, scheduler=None):
    running_loss = 0.0
    epoch = tqdm(train_loader, leave=False)
    if finetune: encoder.train()
    else: encoder.eval()
    
    for i, batch in enumerate(epoch):
        batch.to(device)
        optimizer.zero_grad()
        if finetune: embed, _, _ = encoder._encode_batch(batch, normalize=normalize)
        else: 
            with torch.no_grad(): 
                embed, _, _ = encoder._encode_batch(batch, normalize=normalize)
                embed = embed.detach()
        pred = model(embed).squeeze() # does nothing if multilabel
        y = [getattr(batch, label).float() for label in labels]
        y = torch.stack(y, dim=1).squeeze()
        assert pred.shape == y.shape, (pred.shape, y.shape)
        loss = mse_loss(pred, y) 
        running_loss += loss.item()
        epoch.set_postfix(dict(avg_loss=running_loss / (i + 1), loss=loss.item()))
        loss.backward()
        optimizer.step()
        scheduler.step()
    return running_loss / len(train_loader)

@torch.inference_mode()
def evaluate_one_epoch(encoder, model, labels, valid_loader, device, finetune=False, normalize=False):
    model.eval()
    if finetune: encoder.eval()
    running_loss = 0.0
    epoch = tqdm(valid_loader, leave=False)
    for i, batch in enumerate(epoch):
        batch.to(device)
        embed = encoder.encode(batch, normalize=normalize).clone()
        pred = model(embed).squeeze() # does nothing if multilabel
        y = [getattr(batch, label).float() for label in labels]
        y = torch.stack(y, dim=1).squeeze()
        assert pred.shape == y.shape, (pred.shape, y.shape)
        loss = mse_loss(pred, y)
        running_loss += loss.item()
        epoch.set_postfix(dict(avg_loss=running_loss / (i + 1), loss=loss.item()))
    if finetune: encoder.train()
    model.train()
    return running_loss / len(valid_loader)

def compute_metrics(pred, y):
    assert pred.shape == y.shape, pred.shape
    assert pred.shape[-1] <= 5, pred.shape # soft check that the dimensions are [N, <=5]
    if pred.dim() == 1:
        pred = pred.unsqueeze(1)
        y = y.unsqueeze(1)

    mae  = (pred - y).abs().mean(dim=0)
    rmse = (pred - y).pow(2).mean(dim=0).sqrt()

    ss_res = (pred - y).pow(2).sum(dim=0)
    ss_tot = (y - y.mean(dim=0)).pow(2).sum(dim=0)
    r2 = 1 - ss_res / ss_tot
    metrics = {"mae": mae.squeeze(), "rmse": rmse.squeeze(), "r2": r2.squeeze()}
    return metrics

@torch.inference_mode()
def test(encoder, model, labels, test_loader, device, finetune=False, normalize=False):
    model.eval()
    if finetune: encoder.eval()
    epoch = tqdm(test_loader, leave=False)
    pred, y = [], []
    for batch in epoch:
        batch.to(device)
        embed = encoder.encode(batch, normalize=normalize).clone()
        pred.append(model(embed).squeeze().detach().cpu())
        y.append(torch.stack([getattr(batch, label).float().detach().cpu() for label in labels], dim=1).squeeze())
    pred = torch.concat(pred)
    if finetune: encoder.train()
    y = torch.concat(y)

    return compute_metrics(pred, y)

def train(run_name, encoder_path, save_path, labels, device="cpu", finetune=False, schedule=False, normalize=False):
    assert(isinstance(labels, list))
    num_epochs = 500
    batch_size = 128
    lr = 0.001
    train_path = 'TRAIN_PATH'
    valid_path = 'VALID_PATH'
    test_path  = 'TEST_PATH'
    train_dataset = ConformerEdgeDataset(train_path, num_conformers=1, sample_per_batch=1, use_molecule_properties=True)
    valid_dataset = ConformerEdgeDataset(valid_path, num_conformers=1, sample_per_batch=1, use_molecule_properties=True)
    test_dataset  = ConformerEdgeDataset(test_path, num_conformers=1, sample_per_batch=1, use_molecule_properties=True)

    encoder = EncoderModel.load_model(encoder_path, device=device)
    # def reinitialize_weights(model):
    #     for module in model.modules():
    #         if hasattr(module, 'reset_parameters'):
    #             module.reset_parameters()
    # reinitialize_weights(encoder) # zero shot
    if finetune:
        # Freeze all parameters
        for param in encoder.parameters(): param.requires_grad = True
    else:
        encoder.eval()

    molecular_embedding_size = encoder.args['molecular_embedding_size']
    num_labels = len(labels)
    model = Sequential(
        Linear(molecular_embedding_size, 128),
        ReLU(),
        Linear(128, 128),
        ReLU(),
        Linear(128, num_labels),
    ).to(device)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=ConformerEdge_collate_fn,
                              shuffle=True,
                              num_workers=2)
    valid_loader = DataLoader(valid_dataset, 
                            batch_size=batch_size,
                            collate_fn=ConformerEdge_collate_fn,
                            shuffle=True,
                            num_workers=2)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=ConformerEdge_collate_fn,
                             shuffle=True,
                             num_workers=2)

    # optimizer = Lamb(model.parameters(), lr=lr)
    if finetune:
        params = [
            {'params': filter(lambda p: p.requires_grad, encoder.parameters()), 'lr': 0.0001},
            {'params': model.parameters(), 'lr': 0.001},
        ]
        base_lrs = [1e-5, 1e-4]   # e.g. proportionally scaled down from max
        max_lrs  = [1e-4, 8e-4]
    else:
        params = list(model.parameters())
        base_lrs = 1e-4
        max_lrs = 8e-4

    optimizer = AdamW(params, lr=lr)
    scheduler = CyclicLR(optimizer, base_lr=base_lrs, max_lr=max_lrs,
                        step_size_up=240 * 8, mode="triangular", cycle_momentum=False)

    wandb.init(project='PROJECT_NAME', name=run_name)

    best_loss = 1e10
    epochs = tqdm(range(num_epochs), desc='Epoch')
    for epoch in epochs:
        train_loss = train_one_epoch(encoder, model, labels, train_loader, optimizer, device, finetune=finetune, normalize=normalize, scheduler=scheduler)
        val_loss = evaluate_one_epoch(encoder, model, labels, valid_loader, device, finetune=finetune, normalize=normalize)
        # if schedule: scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            if finetune: torch.save({'model': model.state_dict(), 'encoder': encoder.state_dict()}, save_path)
            else: torch.save(model.state_dict(), save_path)
            wandb.save(save_path)
        epochs.set_postfix_str(
            f"Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"Current LR: {optimizer.param_groups[0]['lr']}"
        )

        test_results = dict()
        if (epoch + 1) % 50 == 0: 
            metrics = test(encoder, model, labels, test_loader, device, finetune=finetune, normalize=normalize)
            print(f"Test Score After {epoch + 1} Epochs: {', '.join(f'{k}: {v}' for k, v in metrics.items())}")
        
            for i, label in enumerate(labels):
                for k in metrics:
                    val = metrics[k] if len(labels) == 1 else metrics[k][i]
                    test_results[f"{label}_{k}"] = val.item()

        wandb.log(dict(
            train_loss=train_loss, 
            val_loss=val_loss,
            best_val_loss=best_loss,
            lr=optimizer.param_groups[0]['lr']
        ) | test_results)

    wandb.finish()


if __name__ == '__main__':
    device = 'cuda:0'
    # device = "cpu"
    finetune = True
    schedule = True
    normalize = False
    labels = ["gap", "ip", "ea", "eox", "ered"]
    # labels: 'num_rings', 'gap', 'ip', 'ea', 'eox', 'ered'

    train('AIMetry_ANT', 'ANT_SAVE_PATH', 'CHECKPOINT_PATH', labels, device=device, finetune=False, schedule=schedule, normalize=normalize)

    

    
    