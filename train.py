import torch, wandb, random, argparse, math, numpy as np
from model import *
from dataset import *
from tqdm import tqdm
from typing import *
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_optimizer import Lamb
from collections import defaultdict

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import warnings
warnings.filterwarnings("ignore")

from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit
import itertools
import os, sys

mmap = {'mmap': True} if torch.__version__ >= '2' else {}

def save(file_name, model, optimizer, scheduler, epoch, args):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    torch.save(dict(
        args = args,

        initializer=model.initializer.__class__.__name__,
        encoder=model.encoder.__class__.__name__,
        aggregator=model.aggregator.__class__.__name__,
        evaluator=model.evaluator.__class__.__name__,

        model_state_dict = model.state_dict(),
        optimizer_state_dict = optimizer.state_dict(),
        scheduler_state_dict = scheduler.state_dict(),
        epoch = epoch,
        rng_state = 
            {'torch': torch.get_rng_state(),
             'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             'numpy': np.random.get_state(),
             'python': random.getstate()}
    ), file_name)

def loss_values(loss, loss_dict):
    # one host sync for all logged values instead of one per .item()
    values = torch.stack([loss.detach(), *[v.detach() for v in loss_dict.values()]]).tolist()
    return values[0], dict(zip(loss_dict.keys(), values[1:]))

def train_one_epoch(model, train_loader, optimizer, scaler, device):
    running_loss, named_loss = 0.0, defaultdict(float)
    tqdm_disable = not sys.stdout.isatty() # one machine logs to files, the other logs to stdout
    epoch = tqdm(train_loader, miniters=len(train_loader) // 100, disable=tqdm_disable)
    for i, batch in enumerate(epoch):
        batch.to(device)
        optimizer.zero_grad()
        loss, loss_dict = model(batch)
        loss_val, loss_dict_vals = loss_values(loss, loss_dict)
        running_loss += loss_val
        for k, v in loss_dict_vals.items(): named_loss[k] += v
        epoch.set_postfix(dict(avg_loss=running_loss / (i + 1),
                               loss=loss_val,
                               **loss_dict_vals))

        if math.isnan(loss_val):
            raise Exception("nan loss")

        scaler.scale(loss).backward()

        if scaler.is_enabled(): scaler.step(optimizer) # skips the step on inf/NaN grads
        elif torch.stack([p.grad.isnan().any() for p in model.parameters() if p.grad is not None]).any():
            print('NaN grad detected; skipping batch')
        else: optimizer.step()
        scaler.update()
        if tqdm_disable and i % (len(train_loader) // 100) == 0:
            tqdm.write(f"avg_loss: {running_loss / (i + 1)}, avg_diff_loss: {named_loss['diff_loss']/(i + 1)}, avg_same_loss: {named_loss['same_loss']/(i + 1)}")
    return running_loss / len(train_loader), {k:v / len(train_loader) for k, v in named_loss.items()}

@torch.inference_mode()
def evaluate_one_epoch(model, val_loader, device):
    running_loss, named_loss = 0.0, defaultdict(float)
    tqdm_disable = not sys.stdout.isatty() # one machine logs to files, the other logs to stdout
    epoch = tqdm(val_loader, miniters=len(val_loader) // 100, disable=tqdm_disable)
    for i, batch in enumerate(epoch):
        batch.to(device)
        loss, loss_dict = model(batch)
        loss_val, loss_dict_vals = loss_values(loss, loss_dict)
        running_loss += loss_val
        for k, v in loss_dict_vals.items(): named_loss[k] += v
        epoch.set_postfix(dict(avg_loss=running_loss / (i + 1),
                               loss=loss_val,
                               **loss_dict_vals))
        if tqdm_disable and i % (len(val_loader) // 100) == 0:
            tqdm.write(f"avg_loss: {running_loss / (i + 1)}, avg_diff_loss: {named_loss['diff_loss']/(i + 1)}, avg_same_loss: {named_loss['same_loss']/(i + 1)}")
    return running_loss / len(val_loader), {k:v / len(val_loader) for k, v in named_loss.items()}
                                              
def init_wandb(model, args):
    initializer = model.initializer.__class__.__name__
    encoder = model.encoder.__class__.__name__
    aggregator = model.aggregator.__class__.__name__
    evaluator = model.evaluator.__class__.__name__

    if args['wandb']: 
        run_name = model.__class__.__name__ if args['run_name'] is None else args['run_name']
        wandb.init(project=args['project'], name=run_name)
        wandb.config.dataset = os.path.basename(args['data_directory'])
        if model.initializer: wandb.config.initializer = initializer
        if model.encoder: wandb.config.encoder = encoder
        if model.aggregator: wandb.config.aggregator = aggregator
        if model.evaluator: wandb.config.evaluator = evaluator

    print('==================================================')
    print("\033[1mMODEL\033[0m")
    model.print_parameter_counts()
    if model.initializer: print(f'Initializer: {initializer}')
    if model.encoder: print(f'Encoder: {encoder}')
    if model.aggregator: print(f'Aggregator: {aggregator}')
    if model.evaluator: print(f'Evaluator: {evaluator}')
    print()
    print("\033[1mSETTINGS\033[0m")
    for arg, value in args.items(): print(f'{arg}: {value}')
    print('==================================================')

def train(model, train_loader, val_loader, optimizer, scheduler, scaler, args, rank=None):
    if not rank: init_wandb(model if rank is None else model.module, args)
    best_loss = 1e10
    for epoch in range(args['already_trained'] + 1, args['num_epochs']):
        if not rank: print(f"Training Epoch {epoch + 1} of {args['num_epochs']}")
        if rank is not None:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        train_loss, train_loss_dict = train_one_epoch(model, train_loader, optimizer, scaler, args['device'])
        if not rank: print(f"Validating Epoch {epoch + 1} of {args['num_epochs']}")
        val_loss, val_loss_dict = evaluate_one_epoch(model, val_loader, args['device'])

        if not rank and args['save_every'] > 0 and epoch % args['save_every'] == 0:
            save(os.path.join(args['save_every_out'], f'model_{epoch}.pt'), 
                 model if rank is None else model.module, optimizer, scheduler, epoch, args)

        if not rank and val_loss < best_loss:
            best_loss = val_loss
            if args['save_best'] is not None:
                save(args['save_best'], model if rank is None else model.module, optimizer, scheduler, epoch, args)
                if args['wandb']: wandb.save(args['save_best'])

        scheduler.step(val_loss)

        if not rank:
            print(f"Train Loss: {train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Current LR: {optimizer.param_groups[0]['lr']}")
            if args['wandb']: 
                wandb.log(dict(
                    train_loss=train_loss, 
                    val_loss=val_loss,
                    best_val_loss=best_loss,
                    lr=optimizer.param_groups[0]['lr'],
                    **{f'train_{k}':v for k, v in train_loss_dict.items()},
                    **{f'val_{k}':v for k, v in val_loss_dict.items()}
                ))

        if optimizer.param_groups[0]['lr'] < args['early_stop_lr']:
            if not rank: print("Stopping early, learning rate below threshold.")
            break

    if not rank and args['save_last'] is not None:
        save(args['save_last'], model, optimizer, scheduler, epoch, args)
        if args['wandb']: wandb.save(args['save_last'])

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def parse_args(args: dict | None = None):
    '''
        Parse command-line arguments.

        Args:
            args (dict, optional): Arguments to be combined with command-line arguments.

        Returns:
            args (dict): Output arguments combining command-line, input, loaded, and default arguments.
    '''
    arg_options = dict(
        # Data Arguments
        load_file =                dict(type=str,   persistence='reset',                                          help='Path to previously saved run to load in. Stored arg_options will be used unless additional arg_options are included.'),
        data_directory =           dict(type=str,   persistence='keep',                                           help='Path to directory storing train, val, and test sets. Required unless load_file is provided.'),
        
        # Model Arguments
        atom_embedding_size =      dict(type=int,   persistence='fixed', default=256,                             help='Embedding size of each atom/node in molecules. Default: 256'),
        bond_features =            dict(type=int,   persistence='fixed', default=1,                               help='Embedding size of each bond/edge in molecules. Default: 256'),
        molecular_embedding_size = dict(type=int,   persistence='fixed', default=64,                            help='Output embedding size of a molecule. Default: 1024'),
        num_transformer_layers =   dict(type=int,   persistence='fixed', default=4,                               help='Number of transformer layers in a transformer model. Default: 4'),
        num_attention_heads =      dict(type=int,   persistence='fixed', default=4,                               help='Number of attention heads in a transformer model. Default: 4'),
        edge_hidden_size =         dict(type=int,   persistence='fixed', default=256,                             help='Hidden size of the per-edge MLP in EdgeConv layers. Default: 256'),
        edge_cutoff =              dict(type=float, persistence='fixed', default=4.0,                             help='Distance cutoff (Angstrom) for graph edges, at most the AIMNet2 cutoff of 5.0. Default: 4.0'),
        dropout_rate =             dict(type=float, persistence='keep',  default=0.1,                             help='Dropout rate used in transformers. Default: 0.1'),
        batch_norm =               dict(type=bool,  persistence='fixed', default=False,                           help='Whether to use batch normalization. Default: False'),
        amp =                      dict(type=bool,  persistence='keep',  default=True,                            help='Whether to use fp16 mixed precision for the encoder and aggregator. Default: True'),
        
        # Training Arguments
        batch_size =               dict(type=int,   persistence='keep',  default=64,                              help='Batch size to use during training. Default: 512'),
        batch_confs_per_mol =      dict(type=int,   persistence='keep',  default=4,                               help='Number of conformers per molecule in a batch. Default: 4'),
        num_workers =              dict(type=int,   persistence='keep',  default=4,                               help='Number of subprocesses to use when loading batches from the dataset. If set to 0, the main process will be used. Default: 4'),
        device =                   dict(type=str,   persistence='reset', default='cuda:0',                        help='Device to use for training. Use "multigpu" to run multi-gpu processing. Default: cuda:0'),
        lr =                       dict(type=float, persistence='keep',  default=0.001,                           help='Initial learning rate used in the optimizer. Default: 0.001'),
        num_epochs =               dict(type=int,   persistence='keep',  default=100,                             help='Number of epochs to run. Default: 100'),
        early_stop_lr =            dict(type=float, persistence='keep',  default=1e-8,                            help='Learning rate to stop the run at. Default: 1e-8'),
        save_best =                dict(type=str,   persistence='reset',                                          help='Path to save the run with the lowest validation loss. To use default, use "". Default: Do not save'),
        save_last =                dict(type=str,   persistence='reset',                                          help='Path to save the last run. To use default, use "". Default: Do not save'),
        save_every =               dict(type=int,   persistence='reset', default=-1,                              help='Number of epochs to run between each save. To never save, use -1. Default: -1'),
        save_every_out =           dict(type=str,   persistence='reset',                                          help='Path to output directory of saves. Required if save_every > 0.'),
        wandb =                    dict(type=bool,  persistence='reset', default=True,                            help='Whether to use wandb to track progress. Default: True'),
        run_name =                 dict(type=str,   persistence='keep',                                           help='Name to use for wandb. To use default, use "". Default: Name of model'),
        project =                  dict(type=str,   persistence='keep',  default="AIMetry",                       help='Name of the wandb project to use. Default: "Contrastive-Encoder-Fullsize"'),
        preload =                  dict(type=bool,  persistence='keep',  default=False,                           help='Whether to preload dataset data. Only use if torch.load(mmap=True) is unavailable in torch versions < 2. Default: False'),
        model_version =            dict(type=int,   persistence='keep', default=0,                                help='0 = MultiPoolConformer, 1 = AttentionPoolConformer, 2 = MultiPoolLeCun, 3 = AttentionPoolLeCun')
    )

    parser = argparse.ArgumentParser()
    for arg_name, arg_option in arg_options.items():
        if arg_option['type'] == bool:
            group = parser.add_mutually_exclusive_group()
            group.add_argument(f'--no_{arg_name}', dest=arg_name, action='store_false', default=None, help=arg_option.get('help', None))
            group.add_argument(f'--{arg_name}', dest=arg_name, action='store_true', default=None, help=arg_option.get('help', None))
        else: parser.add_argument(f'--{arg_name}', type=arg_option['type'], default=None, help=arg_option.get('help', None))
    if args is None: args = vars(parser.parse_args())
    else: args = {k:(args.get(k, None) if v is None else v) for k,v in vars(parser.parse_args()).items()}
    
    if args['load_file'] is None: old_args = None
    else: old_args = torch.load(args['load_file'], map_location='cpu', **mmap, weights_only=False)['args']

    # new_arg load_old persistence      result
    # None    False    reset        ->  default
    # None    False    keep         ->  default
    # None    False    fixed        ->  default
    # None    True     reset        ->  default
    # None    True     keep         ->  old_arg
    # None    True     fixed        ->  old_arg
    # Value   False    reset        ->  new_arg
    # Value   False    keep         ->  new_arg
    # Value   False    fixed        ->  new_arg
    # Value   True     reset        ->  new_arg
    # Value   True     keep         ->  new_arg
    # Value   True     fixed        ->  error if new_arg ≠ old_arg

    for arg_name, arg_option in arg_options.items():
        if args[arg_name] is None:
            if old_args is None or arg_name not in old_args or arg_option['persistence'] == 'reset':
                args[arg_name] = arg_option.get('default', None)
            else: args[arg_name] = old_args[arg_name]
        elif old_args is not None and arg_option['persistence'] == 'fixed' and args[arg_name] != old_args[arg_name]: 
            parser.error(f"{arg_name} is a fixed argument and cannot be changed from the load file. Stored value: {old_args[arg_name]}")
        if arg_option['type'] == str and args[arg_name] == '': args[arg_name] = None

    if args['data_directory'] is None: parser.error("Data directory is required.")

    if args['save_every'] > 0:
        if args['save_every_out'] is None: parser.error("Save output directory name required for saving every `save_every` epochs.")
        
    return args

def _main(rank, world_size, args, model_fn):
    if rank is not None:
        ddp_setup(rank, world_size)
        args['device'] = rank
    
    args['already_trained'] = -1 
    # used during continue runs, trained saves from [0, num_epochs), so -1 is used and +1 offset later
    if args['load_file'] is not None:
        checkpoint = torch.load(args['load_file'], map_location='cpu', weights_only=False) # else tensors return to the device they were saved from
        if 'rng_state' in checkpoint: # temporary for old save files
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            if torch.cuda.is_available() and checkpoint['rng_state']['cuda'] is not None:
                torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
            np.random.set_state(checkpoint['rng_state']['numpy'])
            random.setstate(checkpoint['rng_state']['python'])
        model = EncoderModel.load_model(args['load_file'], args['device'])
        args['already_trained'] = checkpoint["epoch"]
    else: model = model_fn(args).to(args['device'])
    model.amp = args['amp'] # load_model uses the checkpoint's args, which may predate amp

    dataset_path = args['data_directory']
    # dataset = ConformerEdgeDataset(dataset_path, sample_per_batch=args['batch_confs_per_mol'], use_molecule_properties=True)
    collate_fn = ConformerEdge_collate_fn
    # generator = torch.Generator().manual_seed(0)
    # train_dataset, val_dataset = random_split(dataset, [0.95, 0.05], generator=generator)
    # torch.save(val_dataset, f'{args["data_directory"]}/datasets/val_dataset.pt')
    # del train_dataset
    # del val_dataset
    # del dataset
    dataset = ConformerEdgeDataset(dataset_path, sample_per_batch=args['batch_confs_per_mol'], use_molecule_properties=False, preload=args['preload'])
    generator = torch.Generator().manual_seed(0)
    train_dataset, val_dataset = random_split(dataset, [0.95, 0.05], generator=generator)

    train_loader = DataLoader(train_dataset,
                              batch_size=args['batch_size'],
                              collate_fn=collate_fn,
                              shuffle=rank is None,
                              sampler=DistributedSampler(train_dataset) if rank is not None else None,
                              num_workers=args['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=args['batch_size'],
                            collate_fn=collate_fn,
                            shuffle=rank is None,
                            sampler=DistributedSampler(val_dataset) if rank is not None else None,
                            num_workers=args['num_workers'],
                            pin_memory=True)

    if rank is not None: model = DDP(model, device_ids=[rank])

    optimizer = Lamb(model.parameters(), lr=args['lr'])
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5) # changed to 10
    scaler = torch.amp.GradScaler('cuda', enabled=args['amp'])

    if args['load_file'] is not None:
        if rank is None: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    train(model, train_loader, val_loader, optimizer, scheduler, scaler, args, rank)
    if rank is not None: destroy_process_group()

def main(args: dict, model_fn: Type[EncoderModel] | Callable[[dict], EncoderModel] | None = None):
    '''
    Main function to set up and begin training.

    Args:
        args (dict): Model arguments. See `parse_args()`.
        model_fn (Type[EncoderModel], Callable[[dict], EncoderModel], optional): Callable object or function that defines an `EncoderModel` given arguments. Ignored if `load_file` is defined in arguments.

    '''
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if args['device'] == 'multigpu':
        world_size = torch.cuda.device_count()
        print('GPUs:', world_size)
        mp.spawn(_main, args=(world_size, args, model_fn), nprocs=world_size)
    else: _main(None, None, args, model_fn)