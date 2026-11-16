import torch, math, time
from unittest.mock import patch
from torch import device
from os import path
from torch_geometric import utils
from torch_geometric.utils import softmax
from aimnet.models.aimnet2 import AIMNet2
from aimnet2calc import AIMNet2Calculator
# from aimnet import *
from torch.nn import Sequential, ModuleList, Linear, ReLU, GELU, LayerNorm, Sigmoid, Parameter, Embedding, CosineSimilarity, Softmax, BCELoss
from torch_geometric.nn import GINEConv, GPSConv, EdgeConv
from torch_geometric.nn.aggr import MultiAggregation
from torch import Tensor, device
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch import jit
from torch_scatter import scatter_max



# ======================================== Initializer ======================================== #

class AIMNet2WithAIM(AIMNet2Calculator):
    atom_feature_keys = AIMNet2Calculator.atom_feature_keys + ['aim', "d_ij", "nbmat"]

    def __init__(self, model_path, device):
        # AIMNet2Calculator hardcodes 'cuda:0', which creates a CUDA context there; load on cpu instead
        with patch.object(torch.cuda, 'is_available', lambda: False):
            super().__init__(model_path)
        self.device = device
        self.model.to(device)

    def keep_only(self, data):
        ret = super().keep_only(data)
        if 'aim' in data: ret['aim'] = data['aim']
        if 'd_ij' in data: ret['d_ij'] = data['d_ij']
        if 'nbmat' in data: ret['nbmat'] = data['nbmat']
        return ret

class NewAIMNetInitializer(torch.nn.Module):
    def __init__(self, device: device, edge_cutoff: float | None = None, **kwargs):
        super().__init__()
        self.device = device
        self.edge_cutoff = 1000.0 if edge_cutoff is None else edge_cutoff # None keeps every AIMNet2 neighbor (5 A); old checkpoints lack this arg
        self.model = AIMNet2WithAIM('aimnet2_b973c_0.jpt', self.device)

    def forward(self, batch):
        batch = batch.clone()

        # Format 1: just pass coord, numbers, charge — no nbmat needed
        out = self.model({
            'coord':   batch.coord,                        # [total_atoms, 3]
            'numbers': batch.numbers,                      # [total_atoms]
            'charge':  batch.charge,                       # [B]
            'mol_idx': batch.batch.to(torch.int32),        # [total_atoms]
        })
        aim   = out['aim'].detach()    # [N, 256]

        batch.x = F.normalize(aim, p=2, dim=1) # prevents nan grad
        batch.x[torch.isnan(batch.x)] = 0 # for probe
        # charges are omitted because AIM vectors already bake in charge during forward pass

        if getattr(batch, "edge_index", None) is None: # might delete, this is legacy code for an old model
            assert getattr(batch, "edge_attr", None) is None
            d_ij  = out['d_ij'].detach()   # [N, k]
            nbmat = out['nbmat'].detach()  # [N, k]

            valid = d_ij < self.edge_cutoff
            src, t = valid.nonzero(as_tuple=True)
            dst = nbmat[src, t].long()
            # weights = d_ij[src, t]
            batch.edge_index = torch.stack([src, dst])
            # batch.edge_attr = weights.unsqueeze(1)
        return batch

# ======================================== Encoder ======================================== #

class FastEdgeConv(torch.nn.Module):
    """
    Drop-in replacement for `EdgeConv(nn)` (max aggregation) with identical parameters.
    The first Linear of `nn` is applied per node instead of per edge, using
    W[x_i; x_j - x_i] + b = (Wa - Wb) x_i + Wb x_j + b.
    """
    def __init__(self, nn: Sequential):
        super().__init__()
        self.nn = nn

    def forward(self, x, edge_index):
        lin, C = self.nn[0], x.shape[1]
        Wa, Wb = lin.weight[:, :C], lin.weight[:, C:]
        a = F.linear(x, Wa - Wb, lin.bias) # [num_atoms, 2 * atom_embedding_size]
        b = F.linear(x, Wb)
        src, dst = edge_index
        h = a.index_select(0, dst) + b.index_select(0, src) # [num_edges, 2 * atom_embedding_size]
        for m in self.nn[1:]: h = m(h)
        return utils.scatter(h, dst, dim=0, dim_size=x.shape[0], reduce='max')

class EdgeConvGPSLayer(torch.nn.Module):
    """
    A layer for `EdgeConvGPSEncoder`.
    
    Args:
        atom_embedding_size (int): Size of atom (node) embeddings.
        bond_embedding_size (int): Size of bond (edge) embeddings.
        num_attention_heads (int): Number of attention heads in each transformer layer.
        dropout_rate (float): Dropout rate used in transformer layers.
        device (torch.device): Device to use.
        edge_hidden_size (int): Hidden size of the per-edge MLP. Default: 2 * atom_embedding_size.
    """
    def __init__(self, 
                atom_embedding_size: int, 
                # bond_embedding_size: int,
                num_attention_heads: int,
                dropout_rate: float,
                device: device,
                edge_hidden_size: int | None = None):
        super().__init__()
        self.device = device
        self.out_atom_embedding_size = self.atom_embedding_size = atom_embedding_size
        # self.out_bond_embedding_size = self.bond_embedding_size = bond_embedding_size
        edge_hidden_size = edge_hidden_size or atom_embedding_size * 2
        self.nn = Sequential(Linear(atom_embedding_size * 2, edge_hidden_size), # * 2 because input is [x_i; x_j - x_i]
                        LayerNorm(edge_hidden_size), GELU(),
                        Linear(edge_hidden_size, atom_embedding_size))

        self.gnn = FastEdgeConv(self.nn) # edgeconv works well with point clouds with edges w/out features
        self.gps = GPSConv(atom_embedding_size, 
                           self.gnn,
                           num_attention_heads,
                           dropout_rate,
                           norm="layer_norm")
        
    def forward(self, x, edge_index, batch, **kwargs):
        y = self.gps(x, edge_index, batch=batch)
        return y

class EdgeConvGPSEncoder(torch.nn.Module):
    """
    An encoder for molecules using `torch_geometric.nn.GPSConv` layers with 
    `torch_geometric.nn.GINEConv` as the MPNN.
    
    Args:
        atom_embedding_size (int): Size of atom (node) embeddings.
        bond_embedding_size (int): Size of bond (edge) embeddings.
        num_attention_heads (int): Number of attention heads in each transformer layer.
        num_transformer_layers (int): Number of transformer layers.
        dropout_rate (float): Dropout rate used in transformer layers.
        device (torch.device): Device to use.
    """
    def __init__(self, 
                atom_embedding_size: int,
                # bond_embedding_size: int,
                num_attention_heads: int,
                num_transformer_layers: int,
                dropout_rate: float,
                device: device,
                edge_hidden_size: int | None = None,
                **kwargs):
        super().__init__()
        self.device = device
        self.atom_embedding_size = atom_embedding_size
        # self.out_bond_embedding_size = self.bond_embedding_size = bond_embedding_size
        self.out_atom_embedding_size = atom_embedding_size * num_transformer_layers
        # Initialize layers
        self.layers = ModuleList(
            [EdgeConvGPSLayer(atom_embedding_size,
                            #   bond_embedding_size,
                              num_attention_heads,
                              dropout_rate,
                              device,
                              edge_hidden_size) for _ in range(num_transformer_layers)])

    def forward(self, batch):
        x = batch.x # [num_atoms, atom_embedding_size]
        xs = []
        for layer in self.layers: 
            x = layer(x, batch.edge_index, batch.batch) # [num_atoms, atom_embedding_size]
            xs.append(x)
        return torch.stack(xs, 2)

# ======================================== Aggregator ======================================== #

class MoreMultiPoolLayerAggregator(torch.nn.Module):
    """
    Uses a max, min, and mean pool with a linear layer to aggregate. 
    
    Args:
        atom_embedding_size (int): Size of input atom (node) embeddings.
        molecular_embedding_size (int): Size of output molecular embedding.
        device (torch.device): Device to use.
    """
    def __init__(self, 
                atom_embedding_size: int, 
                molecular_embedding_size: int,
                num_transformer_layers: int,
                device: device,
                **kwargs):
        assert molecular_embedding_size % num_transformer_layers == 0
        super().__init__()
        self.device = device
        self.atom_embedding_size = atom_embedding_size
        self.molecular_embedding_size = molecular_embedding_size
        self.num_transformer_layers = num_transformer_layers
        hidden_size = atom_embedding_size
        self.pool = MultiAggregation(['max', 'min', 'mean', 'std'], mode="attn",
            mode_kwargs={
            'in_channels': atom_embedding_size,
            'out_channels': hidden_size,
            'num_heads': 4,
            })
        
        # self.mlp = ModuleList([Sequential(
        #     Linear(hidden_size, hidden_size),
        #     LayerNorm(hidden_size), GELU(),
        #     Linear(hidden_size, molecular_embedding_size // num_transformer_layers),
        # ) for _ in range(num_transformer_layers)])

        self.mlp = Sequential(
            Linear(hidden_size * num_transformer_layers, hidden_size * num_transformer_layers // 2),
            LayerNorm(hidden_size * num_transformer_layers // 2), GELU(),
            Linear(hidden_size * num_transformer_layers // 2, molecular_embedding_size),
        )
        # given a molecular embedding size of X, and N number of transformer layers, the first X//N values will represent the 0th transformer layer
        
    def forward(self, x, batch):
        # x = torch.concat([self.mlp[i](self.pool(x[:,:,i], batch.batch)) for i in range(self.num_transformer_layers)], 1) # [batch_size, molecular_embedding_size]
        x = self.mlp(torch.concat([self.pool(x[:,:,i], batch.batch) for i in range(self.num_transformer_layers)], 1)) # [batch_size, molecular_embedding_size]
        x = F.normalize(x, p=2, dim=1)
        return x

# ======================================== Evaluator ======================================== #

class AwareNTXentEvaluator(torch.nn.Module): 
    def __init__(self,
                 batch_size: int,
                 molecular_embedding_size: int,
                 batch_confs_per_mol: int,
                 device: device,
                 **kwargs):
        super().__init__()
        self.device = device
        self.batch_confs_per_mol = batch_confs_per_mol # confs per molecule in a batch
        # self.batch_size = batch_size # number of unique molecules
        # self.indices = torch.combinations(torch.arange(self.batch_size * batch_confs_per_mol, device=device), with_replacement=False)
        # self.margin = molecular_embedding_size # minimum distance that will be scaled by jaccard similarity
        # self.sigmoid = Sigmoid()
        # self.diff_weight = 0.5

        self.temperature = 0.2
        self.row_softmax = Softmax(dim=1)
        self.BCELoss = BCELoss(reduction="none")
        assert batch_confs_per_mol > 1
    
    # int64s only
    @staticmethod
    def bit_count(x: torch.Tensor):
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        return (x * 0x0101010101010101) >> 56
    
    @staticmethod
    def jaccard(x: torch.Tensor, y: torch.Tensor):
        return AwareNTXentEvaluator.bit_count(x & y).sum(-1) / AwareNTXentEvaluator.bit_count(x | y).sum(-1)

    def kabsch_torch_batched(self, P, Q, num_atoms, mask):
        """
        code from: https://hunterheidenreich.com/posts/kabsch-algorithm/
        Computes the optimal rotation and translation to align two sets of points (P -> Q),
        and their RMSD, in a batched manner.
        :param P: A BxNx3 matrix of points
        :param Q: A BxNx3 matrix of points
        :return: A tuple containing the optimal rotation matrix, the optimal
                translation vector, and the RMSD.
        """
        assert P.shape == Q.shape, "Matrix dimensions must match"
        mask = mask.unsqueeze(-1)
        # Compute centroids
        centroid_P = torch.sum(P, dim=1, keepdims=True) / num_atoms.reshape(-1, 1, 1)  # Bx1x3
        centroid_Q = torch.sum(Q, dim=1, keepdims=True) / num_atoms.reshape(-1, 1, 1)  #
        # Optimal translation
        t = centroid_Q - centroid_P  # Bx1x3
        t = t.squeeze(1)  # Bx3
        # Center the points
        p = P - centroid_P  # BxNx3
        q = Q - centroid_Q  # BxNx3
        p = p * mask
        q = q * mask
        # Compute the covariance matrix
        H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3
        # SVD
        U, S, Vt = torch.linalg.svd(H)  # Bx3x3
        # Validate right-handed coordinate system
        d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
        flip = d < 0.0
        Vt[:, -1, :] *= torch.where(flip, -1.0, 1.0).unsqueeze(-1) # no host sync
        # Optimal rotation
        R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
        # RMSD
        rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(1, 2)) - q), dim=(1, 2)) / num_atoms)
        return rmsd.detach()

    @staticmethod
    def masked_soft_cross_entropy(input, target, mask):
        input = input.masked_fill(~mask, float('-inf'))
        log_probs = F.log_softmax(input, dim=1)
        log_probs = log_probs.masked_fill(~mask, 0.0)
        target = target.masked_fill(~mask, 0.0)
        return -(target * log_probs)   # [N, N], no .sum(dim=1)
    
    def forward(self, embed, batch):
        # PRECONDITION: EMBED MUST HAVE POSITIVE PAIRS NEXT TO EACH OTHER:
        # [1, 1, ..., 1, 2, 2, ..., 2, ..., N, N, ..., N] OR IT WILL BREAK
        # shape of embed: [B, x], x = embedding size
        total_confs = len(batch)
        batch_size = total_confs // self.batch_confs_per_mol
        natoms = torch.bincount(batch.batch)

        # get cos_sim_matrix (embed is already L2-normalized by the aggregator)
        sim_matrix = embed @ embed.T
        sim_matrix /= self.temperature
        # sim_matrix.fill_diagonal_(float(-1e9))
        sim_embedding_matrix = sim_matrix
        # sim_embedding_matrix = self.row_softmax(sim_matrix)

        # set different molecules similarity score. The operation for jaccard is cheap
        # so this block of code computes jaccard of all pairs of fingerprints
        # then masks away lower triangle and block diagonal
        diff_molecule_mask = torch.ones(batch_size, self.batch_confs_per_mol, self.batch_confs_per_mol, dtype=torch.bool).to(self.device)
        diff_molecule_mask = ~torch.block_diag(*diff_molecule_mask.unbind(0))
        fingerprints = batch.fingerprint[torch.arange(batch_size) * self.batch_confs_per_mol]
        sim_anchor_matrix = self.jaccard(fingerprints.repeat_interleave(fingerprints.shape[0], dim=0), fingerprints.repeat(fingerprints.shape[0],1))
        sim_anchor_matrix = sim_anchor_matrix.reshape(batch_size, batch_size)
        sim_anchor_matrix = sim_anchor_matrix.repeat_interleave(self.batch_confs_per_mol,dim=0).repeat_interleave(self.batch_confs_per_mol,dim=1) # [B, B]
        sim_anchor_matrix *= diff_molecule_mask
        num_diff_molecules = torch.sum(diff_molecule_mask)
        # set same molecules similarity sccore. This block of code generates the index pairs
        # (upper triangle of each block of a block diagonal matrix) and then gets rmsd
        # of each pair of molecules associated with corresponding index.
        index_offset = (torch.arange(batch_size) * self.batch_confs_per_mol).view(-1, 1, 1).to(self.device)
        same_indices = torch.combinations(torch.arange(self.batch_confs_per_mol, device=self.device)).expand(batch_size, -1, -1) + index_offset
        same_indices = same_indices.flatten(0, 1) # [same_pairs, 2]
        pad_coord, pad_mask = utils.to_dense_batch(batch.coord, batch.batch) # [B, N_max, 3], [B, N_max]
        pad_coord_pairs = pad_coord[same_indices]
        natoms_pairs = natoms[same_indices][:, 0]
        pad_coord_1, pad_coord_2 = pad_coord_pairs.unbind(1) # [same_pairs, N_max, 3], [same_pairs, N_max, 3]
        pad_mask_same = pad_mask[same_indices][:,0]
        rmsd = self.kabsch_torch_batched(pad_coord_1, pad_coord_2, natoms_pairs, pad_mask_same) # [same_pairs]
        rmsd = torch.exp(-rmsd / torch.std(rmsd).clamp_min(1e-8)) # divide by stdev to normalize large rmsds
        sim_anchor_matrix[same_indices.T[0], same_indices.T[1]] = rmsd # fill it in
        sim_anchor_matrix[same_indices.T[1], same_indices.T[0]] = rmsd # fill in the symmetric
        num_same_molecules = same_indices.shape[0] * 2
        
        # compute loss. This code block takes BCE loss and calculates the average 
        # same and diff loss
        # sim_anchor_matrix.fill_diagonal_(float(-1e9))
        # sim_anchor_matrix /= self.temperature
        # sim_anchor_matrix = self.row_softmax(sim_anchor_matrix)
        loss_mask = ~torch.eye(total_confs, dtype=torch.bool, device=self.device)
        loss_matrix = self.masked_soft_cross_entropy(sim_matrix, sim_anchor_matrix, loss_mask)
        # loss_matrix = F.cross_entropy(sim_embedding_matrix, sim_anchor_matrix, reduction="none")
        diff_loss = torch.sum(diff_molecule_mask * loss_matrix) / num_diff_molecules
        # if torch.isnan(diff_loss): diff_loss = torch.tensor(0.0, requires_grad=True) # deals with nan values
        same_loss = torch.sum(~diff_molecule_mask * loss_matrix) / num_same_molecules
        loss = same_loss + diff_loss
        return loss, {"contrastive_loss": loss, "same_loss": same_loss, "diff_loss": diff_loss}

class NTXentEvaluator(AwareNTXentEvaluator): 
    def forward(self, embed, batch):
        total_confs = len(batch)
        batch_size = total_confs // self.batch_confs_per_mol
        natoms = torch.bincount(batch.batch)

        sim_matrix = embed @ embed.T # embed is already L2-normalized by the aggregator
        sim_matrix = sim_matrix / self.temperature

        target_idx = torch.arange(total_confs, device=self.device)
        target_idx[0::2] += 1
        target_idx[1::2] -= 1
        target = F.one_hot(target_idx, num_classes=total_confs).float()  # [N, N]

        loss_mask = ~torch.eye(total_confs, dtype=torch.bool, device=self.device)  # [N, N]
        loss_matrix = self.masked_soft_cross_entropy(sim_matrix, target, loss_mask)  # [N, N]

        loss = loss_matrix.sum(dim=1).mean()
        zero = torch.zeros((), device=loss.device)
        return loss, {"contrastive_loss": loss, "same_loss": zero, "diff_loss": zero}

# ======================================== Full Model ======================================== #

class EncoderModel(torch.nn.Module):    
    def __init__(self, args, initializer, encoder, aggregator, evaluator):
        super().__init__()
        self.args = args
        self.device = args["device"]
        self.amp = args.get("amp", False) # fp16 autocast for encoder + aggregator
        self.initializer = initializer(**args)
        self.encoder = encoder(**args)
        self.aggregator = aggregator(**args)
        self.evaluator = evaluator(**args)

    @torch.inference_mode()
    def encode(self, data):
        # data can either be a torch geometric data or torch geometric batch
        self.eval()
        if isinstance(data, Batch): return self._encode_batch(data)[0]
        res = self._encode_batch(Batch.from_data_list([data]))[0].squeeze(0)
        self.train()
        return res

    def _encode_batch(self, batch):
        loss, loss_dict = 0.0, {}

        batch = self.initializer(batch)
        if isinstance(batch, tuple): 
            if len(batch) > 2: loss_dict = batch[2]
            batch, loss = batch[0], batch[1]

        # initializer (AIMNet2) and evaluator (SVD in kabsch) stay in fp32
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.amp):
            x = self.encoder(batch)
            if isinstance(x, tuple):
                if len(x) > 2: loss_dict.update(x[2])
                x, loss = x[0], loss + x[1]

            x = self.aggregator(x, batch)
            if isinstance(x, tuple):
                if len(x) > 2: loss_dict.update(x[2])
                x, loss = x[0], loss + x[1]

        return x.float(), loss, loss_dict

    def forward(self, batch ) -> tuple[float, dict[str, float]]:
        x, loss, loss_dict = self._encode_batch(batch)
        x = self.evaluator(x, batch)
        if isinstance(x, tuple):
            loss = loss + x[0]
            loss_dict.update(x[1])
        else: loss = loss + x
        return loss, loss_dict
    
    def print_parameter_counts(self, verbose=False):
        def count_params(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        def print_named_params(module, label):
            print(f"\n--- {label} parameter blocks ---")
            for name, param in module.named_parameters():
                shape_str = "x".join(str(d) for d in param.shape)
                print(f"  {name:60s} {str(list(param.shape)):20s} "
                    f"({param.numel():,} params, requires_grad={param.requires_grad})")

        if verbose:
            print_named_params(self.encoder, "Encoder")
            print_named_params(self.aggregator, "Aggregator")

        enc_total, enc_trainable = count_params(self.encoder)
        agg_total, agg_trainable = count_params(self.aggregator)

        print(f"\n--- Summary ---")
        print(f"Encoder parameters:    {enc_total:,} total, {enc_trainable:,} trainable")
        print(f"Aggregator parameters: {agg_total:,} total, {agg_trainable:,} trainable")
        print(f"Combined:              {enc_total + agg_total:,} total, {enc_trainable + agg_trainable:,} trainable")

    @staticmethod
    def load_model(file_name: str, device: str | None = None) -> 'EncoderModel':
        data = torch.load(file_name, map_location='cpu', weights_only=False)
        args = data['args']
        if device is not None: args['device'] = device

        model = EncoderModel(
            args        = args, 
            initializer = globals()[data['initializer']],
            encoder     = globals()[data['encoder']],
            aggregator  = globals()[data['aggregator']],
            evaluator   = globals()[data['evaluator']],
        )

        model.to(args['device'])
        model.load_state_dict(data['model_state_dict'])
        return model

class AIMNet2ECGPSMMLPANTXEncoder(EncoderModel):
    def __init__(self, args):
        super().__init__(args=args,
                         initializer=NewAIMNetInitializer,
                         encoder=EdgeConvGPSEncoder,
                         aggregator=MoreMultiPoolLayerAggregator,
                         evaluator=AwareNTXentEvaluator)

class AIMNet2ECGPSMMLPNTXEncoder(EncoderModel):
    def __init__(self, args):
        super().__init__(args=args,
                         initializer=NewAIMNetInitializer,
                         encoder=EdgeConvGPSEncoder,
                         aggregator=MoreMultiPoolLayerAggregator,
                         evaluator=NTXentEvaluator)