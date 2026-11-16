"""Shared helpers for molecular geometries, embeddings, similarities, and analysis."""

from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter, to_dense_batch
from tqdm import tqdm


# Fingerprints and similarities


def create_fingerprint_generator(fp_type="ecfp", radius=3, n_bits=2048):
    """Construct a Morgan or RDKit fingerprint generator with explicit settings."""
    if fp_type == "ecfp":
        return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    if fp_type == "rdkit":
        return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    raise ValueError(f"Unsupported fingerprint type: {fp_type}")


def smiles_to_fingerprint(smiles, fp_type="ecfp", radius=3, n_bits=2048,
                         packed=False, sanitize_properties=True):
    """Return a dense int32 or packed int64 fingerprint for one SMILES string.

    Disabling property sanitization skips RDKit property validation while
    applying the remaining sanitization steps.
    """
    if sanitize_properties:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    generator = create_fingerprint_generator(fp_type, radius=radius, n_bits=n_bits)
    fingerprint = generator.GetFingerprint(mol)
    if packed:
        bits = np.array(fingerprint, dtype=np.uint8)
        return np.packbits(bits).view(np.int64).copy()
    return np.array(fingerprint.ToList(), dtype=np.int32)


def smiles_to_fingerprints(smiles_list, fp_type="ecfp", num_workers=8,
                          radius=3, n_bits=2048, packed=True):
    """Generate fingerprints in input order and return a CPU tensor."""
    worker = partial(smiles_to_fingerprint, fp_type=fp_type, radius=radius,
                     n_bits=n_bits, packed=packed)
    with Pool(processes=num_workers) as pool:
        fingerprints = list(tqdm(pool.imap(worker, smiles_list),
                                 total=len(smiles_list), desc="fingerprinting"))
    return torch.tensor(np.stack(fingerprints))


def match_molecule_ids(source_ids, target_ids):
    """Return source-row indices in target ID order; require unique source IDs."""
    source_ids = torch.as_tensor(source_ids).detach().cpu().reshape(-1)
    target_ids = torch.as_tensor(target_ids).detach().cpu().reshape(-1)
    sorted_ids, order = torch.sort(source_ids)
    if (sorted_ids[1:] == sorted_ids[:-1]).any():
        raise ValueError("Source molecule IDs must be unique")
    positions = torch.searchsorted(sorted_ids, target_ids)
    if (positions >= len(sorted_ids)).any() or not torch.equal(sorted_ids[positions], target_ids):
        raise ValueError("Requested molecule IDs are missing from the source")
    return order[positions]


def load_or_compute_fingerprints(directory, dataset_dir, smiles_dir, ids=None,
                                 fp_types=("ecfp", "rdkit"), radius=3,
                                 n_bits=2048, num_workers=8):
    """Load shared packed fingerprints and SMILES, optionally ordered by molecule IDs.

    Missing files are generated from the processed dataset's molecule IDs.
    Concurrent notebook runs share the same cache-generation lock.
    """
    directory = Path(directory)
    paths = {
        fp_type: directory / (
            f"fingerprints_{fp_type}"
            + (f"_r{radius}" if fp_type == "ecfp" else "")
            + f"_{n_bits}.pt")
        for fp_type in fp_types
    }
    if any(not path.is_file() for path in paths.values()):
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory / ".fingerprints.lock")):
            missing = [fp_type for fp_type, path in paths.items() if not path.is_file()]
            if missing:
                molecule_ids = torch.cat([
                    torch.load(path, map_location="cpu", weights_only=False, mmap=True)["id"]
                    .reshape(-1)
                    for path in sorted(Path(dataset_dir).glob("*.pt"))
                ])
                lookup = load_smiles_lookup(smiles_dir)
                smiles = [lookup[molecule_id] for molecule_id in molecule_ids.tolist()]
                del lookup
                for fp_type in missing:
                    print(f"Computing missing fingerprint cache: {paths[fp_type].name}")
                    fingerprints = smiles_to_fingerprints(
                        smiles, fp_type=fp_type, num_workers=num_workers,
                        radius=radius, n_bits=n_bits)
                    temporary = paths[fp_type].with_suffix(".tmp")
                    torch.save({"id": molecule_ids, "smiles": smiles,
                                "fingerprints": fingerprints}, temporary)
                    temporary.replace(paths[fp_type])
                    del fingerprints
                del molecule_ids, smiles

    caches = {fp_type: torch.load(path, map_location="cpu", weights_only=False, mmap=True)
              for fp_type, path in paths.items()}
    reference = next(iter(caches.values()))
    molecule_ids = reference["id"] if ids is None else torch.as_tensor(ids).cpu().reshape(-1)
    result = {"id": molecule_ids}
    for fp_type, cache in caches.items():
        if torch.equal(cache["id"], molecule_ids):
            result[fp_type] = cache["fingerprints"]
            if "smiles" not in result:
                result["smiles"] = cache["smiles"]
        else:
            indices = match_molecule_ids(cache["id"], molecule_ids)
            result[fp_type] = cache["fingerprints"][indices]
            if "smiles" not in result:
                result["smiles"] = [cache["smiles"][index] for index in indices.tolist()]
    return result


def jaccard_similarity(fp1, fp2):
    """Return Jaccard similarity for dense bit fingerprints; return zero for two empty sets."""
    fp1 = np.asarray(fp1, dtype=bool)
    fp2 = np.asarray(fp2, dtype=bool)
    intersection = np.logical_and(fp1, fp2).sum()
    union = np.logical_or(fp1, fp2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def bit_count(x: torch.Tensor):
    """Count set bits in each packed 64-bit integer."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def packed_jaccard_similarity(x: torch.Tensor, y: torch.Tensor):
    """Jaccard similarity for packed int64 fingerprints, with broadcasting."""
    return bit_count(x & y).sum(-1) / bit_count(x | y).sum(-1)


def pairwise_jaccard_similarity(query_fps, dataset_fps, chunk_size=500, device="cpu"):
    """Compare packed fingerprints in chunks and return a CPU similarity matrix."""
    query_fps = query_fps.to(device)
    dataset_fps = dataset_fps.to(device)
    query_popcount = bit_count(query_fps).sum(-1)
    dataset_popcount = bit_count(dataset_fps).sum(-1)

    n_queries = query_fps.shape[0]
    n_dataset = dataset_fps.shape[0]
    out = torch.empty((n_queries, n_dataset), dtype=torch.float32)
    for start in tqdm(range(0, n_dataset, chunk_size), desc="jaccard chunks"):
        end = min(start + chunk_size, n_dataset)
        chunk = dataset_fps[start:end]
        inter = bit_count(query_fps[:, None, :] & chunk[None, :, :]).sum(-1)
        union = query_popcount[:, None] + dataset_popcount[None, start:end] - inter
        out[:, start:end] = (inter / union).float().cpu()
    return out


# Molecular graphs and conformers


def mol_to_graph(mol, use_3d=True, sanitize=True):
    """Convert one RDKit conformer to an atomic graph, with total formal charge."""
    if sanitize:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)

    if use_3d and mol.GetNumConformers() > 0:
        pos = torch.tensor(mol.GetConformer().GetPositions()).float()
    else:
        pos = torch.zeros((mol.GetNumAtoms(), 3)).float()

    atom_types = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.uint8)
    all_charges = torch.tensor(
        [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.int8)

    return Data(
        x=atom_types,  # Keep x for compatibility with graph batching.
        numbers=atom_types,
        coord=pos,
        charge=torch.sum(all_charges),
    )


def smiles_to_batch(smiles, n_confs=10):
    """Generate explicit-H conformers with RDKit defaults and MMFF94; assign zero charge."""
    datalist = []
    for smile in tqdm(smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smile))
        AllChem.EmbedMultipleConfs(mol, numConfs=n_confs)
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        numbers = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.uint8)
        charge = torch.tensor(0, dtype=torch.uint8)

        for conf_id in range(mol.GetNumConformers()):
            conf = mol.GetConformer(conf_id)
            coords = torch.tensor(conf.GetPositions(), dtype=torch.float32)
            datalist.append(Data(x=numbers, numbers=numbers, coord=coords, charge=charge))
    res = Batch.from_data_list(datalist)
    del res.x
    return res


def generate_conformers(smiles, n_confs=32, seed=42, max_iters=2000):
    """Return seeded, explicit-H conformer graphs and their molecule indices.

    Optimize with MMFF94 when parameters are available, otherwise with UFF.
    """
    datalist, mol_idx = [], []
    for i, smi in enumerate(tqdm(smiles, desc="generating conformers")):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=seed))
        assert len(conf_ids) > 0, f"conformer generation failed for {smi}"
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
        else:
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
        numbers = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.uint8)
        charge = torch.tensor(Chem.GetFormalCharge(mol), dtype=torch.int64)
        for conf_id in conf_ids:
            coords = torch.tensor(mol.GetConformer(conf_id).GetPositions(), dtype=torch.float32)
            datalist.append(Data(x=numbers, numbers=numbers, coord=coords, charge=charge))
            mol_idx.append(i)
    return datalist, np.array(mol_idx)


def batch_smiles_conformers(smiles, n_confs, mols_per_batch=16):
    """Generate conformers in molecule batches and return their molecule indices."""
    batches = [smiles_to_batch(smiles[s:s + mols_per_batch], n_confs=n_confs)
               for s in range(0, len(smiles), mols_per_batch)]
    assert sum(batch.num_graphs for batch in batches) == len(smiles) * n_confs, (
        "conformer generation returned fewer conformers than requested")
    no_mmff = [
        smi for smi in smiles
        if not AllChem.MMFFHasAllMoleculeParams(Chem.AddHs(Chem.MolFromSmiles(smi)))
    ]
    if no_mmff:
        print(f"no MMFF parameters, left at the ETKDG geometry: {', '.join(no_mmff)}")
    return batches, np.repeat(np.arange(len(smiles)), n_confs)


def load_smiles_lookup(path):
    """Load molecule-ID-to-SMILES mappings from preprocessed NumPy files."""
    lookup = {}
    for file in Path(path).iterdir():
        data = np.load(file)
        ids = data["id"].tolist()
        smiles = np.char.decode(data["smiles"], "ascii")
        lookup.update(zip(ids, smiles))
    return lookup


# Embedding generation and checkpoints


@torch.inference_mode()
def encode_batches(model, batches, return_ids=False, id_attr="smiles", device=None,
                   pooled_aimnet2=False):
    """Encode batches in order and return CPU tensors, optionally with identifiers.

    pooled_aimnet2 uses the normalized mean atomic features instead of the
    aggregation network. Convert the returned embeddings with .numpy() when needed.
    """
    device = model.device if device is None else device
    embeddings, ids = [], []
    for batch in batches:
        if return_ids:
            ids.append(getattr(batch, id_attr).detach().cpu().reshape(-1))
        batch = batch.to(device)
        if pooled_aimnet2:
            atomic = model.initializer(batch)
            atomic = atomic[0] if isinstance(atomic, tuple) else atomic
            encoded = torch.nn.functional.normalize(
                scatter(atomic.x.float(), atomic.batch, dim=0, reduce="mean"), dim=1)
        else:
            encoded = model.encode(batch)
        embeddings.append(encoded.detach().cpu())
    embeddings = torch.cat(embeddings)
    return (embeddings, torch.cat(ids)) if return_ids else embeddings


@torch.inference_mode()
def encode_graphs(model, graphs, chunk_size=256):
    """Batch conformer graphs lazily and return CPU embeddings in input order."""
    def batches():
        for start in range(0, len(graphs), chunk_size):
            batch = Batch.from_data_list(graphs[start:start + chunk_size])
            del batch.x
            yield batch

    return encode_batches(model, batches())


def normalize_embeddings(embeddings):
    """L2-normalize finite, nonzero tensor rows and return a float32 NumPy array."""
    embeddings = embeddings.detach().cpu().float()
    norms = embeddings.norm(dim=1, keepdim=True)
    if not torch.isfinite(embeddings).all() or (norms == 0).any():
        raise ValueError("Cosine comparisons require finite, nonzero embeddings")
    return (embeddings / norms).numpy()


def average_embeddings(embeddings, n_mols):
    """Average equal-sized conformer groups and L2-normalize each molecular vector."""
    means = embeddings.reshape(n_mols, -1, embeddings.shape[1]).mean(1)
    return means / np.linalg.norm(means, axis=1, keepdims=True)


@torch.inference_mode()
def encode_molecules(model, mols, pooled_aimnet2=False, chunk_size=256):
    """Return NumPy molecular embeddings averaged over each molecule's conformers.

    Conformer and molecular vectors are L2-normalized. pooled_aimnet2 uses
    the normalized mean atomic features for each conformer.
    """
    graphs, molecule_indices = [], []
    for index, mol in enumerate(mols):
        for conformer in mol.GetConformers():
            # Copy the conformer because graph conversion kekulizes the molecule.
            graphs.append(mol_to_graph(Chem.Mol(mol, confId=conformer.GetId())))
            molecule_indices.append(index)
    model.eval()
    batches = (Batch.from_data_list(graphs[start:start + chunk_size])
               for start in range(0, len(graphs), chunk_size))
    embeddings = encode_batches(model, batches, pooled_aimnet2=pooled_aimnet2).float()
    means = scatter(embeddings, torch.tensor(molecule_indices), dim=0, reduce="mean")
    return torch.nn.functional.normalize(means, dim=1).numpy()


def encode_molecules_with_models(model_paths, mols, device):
    """Embed molecules with each checkpoint and add a pooled AIMNet2 baseline."""
    from model import EncoderModel
    out = {}
    for name, path in model_paths.items():
        checkpoint = EncoderModel.load_model(path, device=device)
        out[name] = encode_molecules(checkpoint, mols)
    # AIMNet2 is frozen: same in every checkpoint
    out["Pooled AIMNet2"] = encode_molecules(checkpoint, mols, pooled_aimnet2=True)
    return out


@torch.inference_mode()
def save_embeddings(model, dataset, collate_fn, device, out_dir,
                    batch_size=64, num_workers=4, out_name="embeddings"):
    """Save one embedding and molecule ID per dataset item.

    Select one stored conformer per molecule and preserve dataset order.
    """
    model.eval()
    dataset.sample_per_batch = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    embeddings, ids = encode_batches(
        model, tqdm(loader, desc="Embedding molecules"), return_ids=True, device=device)
    result = {"embeddings": embeddings, "id": ids}
    torch.save(result, Path(out_dir) / f"{out_name}.pt")
    return result


def load_or_compute_embeddings(path, checkpoint_path, dataset_dir, device,
                               batch_size=64, num_workers=4, num_conformers=16,
                               mmap=False):
    """Load cached embeddings, or encode and save the first stored conformer per molecule."""
    path = Path(path)
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)

    from dataset import ConformerEdgeDataset, ConformerEdge_collate_fn
    from model import EncoderModel

    print(f"Computing missing embedding cache: {path.name}")
    dataset = ConformerEdgeDataset(
        str(dataset_dir), num_conformers=num_conformers, sample_per_batch=1,
        use_molecule_properties=False, randomize=False, preload=False)
    model = EncoderModel.load_model(checkpoint_path, device=device).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    result = save_embeddings(
        model, dataset, ConformerEdge_collate_fn, device, path.parent,
        batch_size=batch_size, num_workers=num_workers, out_name=path.stem)
    if mmap:
        del result
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return result


def create_untrained_checkpoint(model_class, args, save_path, seed=0):
    """Initialize and save an EncoderModel checkpoint without training."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model_class(args).to(args["device"])
    model.eval()
    torch.save({
        "args": args,
        "initializer": model.initializer.__class__.__name__,
        "encoder": model.encoder.__class__.__name__,
        "aggregator": model.aggregator.__class__.__name__,
        "evaluator": model.evaluator.__class__.__name__,
        "model_state_dict": model.state_dict(),
    }, save_path)
    return model


# Geometric distances


def conformer_cosine_similarities(x: torch.Tensor) -> torch.Tensor:
    """Return cosine similarities for unordered conformer pairs in each molecule.

    Input shape is [molecules, conformers, dimensions]; pairs follow torch.triu_indices.
    """
    n_conformers = x.shape[1]
    if not torch.isfinite(x).all() or torch.any(x.norm(dim=-1) == 0):
        raise ValueError("Cosine distance requires finite, nonzero embeddings.")
    x = torch.nn.functional.normalize(x.double(), p=2, dim=-1)
    sim = torch.matmul(x, x.transpose(-2, -1))  # [N, S, S]
    i, j = torch.triu_indices(n_conformers, n_conformers, offset=1)
    return sim[:, i, j].clamp(-1.0, 1.0)


def aligned_rmsd(P, Q, num_atoms, mask):
    """Return proper-rotation RMSDs for padded coordinate batches.

    P and Q have shape [batch, atoms, 3], with atom counts and a Boolean padding mask.
    """
    assert P.shape == Q.shape
    P, Q = P.double(), Q.double()
    mask = mask.unsqueeze(-1)
    centroid_P = torch.sum(P, dim=1, keepdims=True) / num_atoms.reshape(-1, 1, 1)
    centroid_Q = torch.sum(Q, dim=1, keepdims=True) / num_atoms.reshape(-1, 1, 1)
    p = (P - centroid_P) * mask
    q = (Q - centroid_Q) * mask
    H = torch.matmul(p.transpose(1, 2), q)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1, :] *= -1.0
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
    rmsd = torch.sqrt(
        torch.sum(torch.square(torch.matmul(p, R.transpose(1, 2)) - q), dim=(1, 2)) / num_atoms)
    return rmsd.detach()


def conformer_rmsd(coords, pairs, mapping=None):
    """Compute RMSD for selected conformer pairs, optionally remapping the second geometry."""
    coords = torch.as_tensor(coords, dtype=torch.float64)
    if coords.shape[1] == 0:
        return np.full(len(pairs[0]), np.nan)
    p, q = coords[pairs[0]], coords[pairs[1]]
    if mapping is not None:
        q = q[:, list(mapping), :]
    num_atoms = torch.full((len(pairs[0]),), coords.shape[1], dtype=torch.long)
    mask = torch.ones(p.shape[:2], dtype=torch.bool)
    return aligned_rmsd(p, q, num_atoms, mask).numpy()


def conformer_torsion_distances(angles, pairs):
    """RMS wrapped angular difference in radians; undefined without torsions."""
    angles = np.asarray(angles, dtype=np.float64)
    if angles.shape[1] == 0 or not np.isfinite(angles).all():
        return np.full(len(pairs[0]), np.nan)
    difference = angles[pairs[0]] - angles[pairs[1]]
    wrapped = np.arctan2(np.sin(difference), np.cos(difference))
    return np.sqrt(np.mean(wrapped ** 2, axis=1))


def compute_conformer_distances(model, dataset, collate_fn, device, sample_per_batch,
                                 batch_size=64, num_workers=4):
    """Return aligned RMSD, cosine distance, and IDs for grouped conformers.

    Distances have shape [molecules, pairs] with identical torch.triu_indices
    ordering. Each dataset item supplies sample_per_batch conformers of one molecule.
    """
    model.eval()
    model.device = torch.device(device).type  # autocast expects cpu/cuda, without an index
    S = sample_per_batch
    i, j = torch.triu_indices(S, S, offset=1)  # shared pair index/order

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_rmsd, all_emb_dist, all_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing RMSD vs embedding distance"):
            batch = batch.to(device)
            total_confs = batch.num_graphs
            n_mols = total_confs // S
            assert total_confs % S == 0, "batch didn't divide evenly into groups of S conformers"

            conf_embeddings = model.encode(batch)  # [total_confs, D] -- one embedding per conformer
            conf_embeddings = conf_embeddings.reshape(n_mols, S, -1)
            emb_dist = 1.0 - conformer_cosine_similarities(conf_embeddings)  # [n_mols, P]

            natoms = torch.bincount(batch.batch)  # [total_confs]
            # [total_confs, Amax, 3], [total_confs, Amax]
            pad_coord, pad_mask = to_dense_batch(batch.coord, batch.batch)
            pad_coord = pad_coord.reshape(n_mols, S, pad_coord.shape[1], 3)
            pad_mask = pad_mask.reshape(n_mols, S, pad_mask.shape[1])
            natoms = natoms.reshape(n_mols, S)
            P_coord = pad_coord[:, i, :, :]   # [n_mols, P, Amax, 3]
            Q_coord = pad_coord[:, j, :, :]
            # [n_mols, P, Amax] -- assumes i/j atoms line up positionally
            pair_mask = pad_mask[:, i, :]
            pair_natoms = natoms[:, i]        # [n_mols, P]
            n_pairs = i.shape[0]
            Amax = P_coord.shape[2]
            rmsd = aligned_rmsd(
                P_coord.reshape(n_mols * n_pairs, Amax, 3),
                Q_coord.reshape(n_mols * n_pairs, Amax, 3),
                pair_natoms.reshape(n_mols * n_pairs),
                pair_mask.reshape(n_mols * n_pairs, Amax),
            ).reshape(n_mols, n_pairs)

            ids = batch.smiles.reshape(n_mols, S)[:, 0]
            all_rmsd.append(rmsd.cpu())
            all_emb_dist.append(emb_dist.cpu())
            all_ids.append(ids.cpu())

    return torch.cat(all_rmsd, dim=0), torch.cat(all_emb_dist, dim=0), torch.cat(all_ids, dim=0)


# Correlations and summaries


def spearman_correlations(a, b, min_pairs=3, nan_policy="omit"):
    """Return one Spearman coefficient per paired row.

    omit excludes nonfinite pairs. propagate returns NaN for a row containing NaNs.
    """
    if nan_policy not in ("omit", "propagate"):
        raise ValueError(f"Unsupported NaN policy: {nan_policy}")
    correlations = np.full(len(a), np.nan)
    for index, (x, y) in enumerate(zip(a, b)):
        x, y = np.asarray(x), np.asarray(y)
        if nan_policy == "omit":
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
        elif np.isnan(x).any() or np.isnan(y).any():
            continue
        if len(x) < min_pairs or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        correlations[index], _ = spearmanr(x, y)
    return correlations


def normalize_ranks(values):
    """Center and L2-normalize average ranks, returning None for constant inputs."""
    ranks = rankdata(values, method="average")
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm > 0 else None


def rank_correlation(x, y):
    """Correlate normalized rank vectors, returning NaN when either is undefined."""
    return float(np.clip(x @ y, -1, 1)) if x is not None and y is not None else np.nan


def partial_rank_correlation(x, y, control):
    """Correlate normalized rank vectors after projecting out one control rank vector."""
    if x is None or y is None or control is None:
        return np.nan
    x_residual = x - (x @ control) * control
    y_residual = y - (y @ control) * control
    denominator = np.linalg.norm(x_residual) * np.linalg.norm(y_residual)
    return (float(np.clip((x_residual @ y_residual) / denominator, -1, 1))
            if denominator > 1e-12 else np.nan)


def summarize_distribution(values):
    """Summarize a numeric distribution, excluding NaNs from the statistics."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    q1, median, q3 = np.quantile(valid, [0.25, 0.5, 0.75])
    return {
        "n": len(valid), "n_nan": int(np.isnan(arr).sum()),
        "mean": valid.mean(), "median": median, "stdev": valid.std(ddof=1),
        "q1": q1, "q3": q3, "iqr": q3 - q1,
        "min": valid.min(), "max": valid.max(),
    }


def bootstrap_median_interval(values, n_bootstrap=2000, seed=0):
    """Return the finite count, median, and 95% percentile-bootstrap interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if count < 2:
        return {"n": count, "median": float(np.median(values)) if count else np.nan,
                "lower": np.nan, "upper": np.nan}
    rng = np.random.default_rng(seed)
    bootstrap = []
    for start in range(0, n_bootstrap, 128):
        indices = rng.integers(0, count, size=(min(128, n_bootstrap - start), count))
        bootstrap.append(np.median(values[indices], axis=1))
    lower, upper = np.percentile(np.concatenate(bootstrap), (2.5, 97.5))
    return {"n": count, "median": float(np.median(values)),
            "lower": float(lower), "upper": float(upper)}


# Retrieval and projections


def retrieve_neighbors(model, smiles, embeddings, molecule_ids, id_to_smiles,
                       topk=20, fp_radius=3, fp_nbits=2048):
    """Retrieve molecules by cosine similarity and report Morgan fingerprint agreement."""
    if isinstance(smiles, str):
        smiles = [smiles]
    batch = smiles_to_batch(smiles, n_confs=1)
    query_embeddings = model.encode(batch.to(model.device)).cpu()
    embeddings = embeddings.cpu()
    results = {}
    for query_smiles, query in tqdm(zip(smiles, query_embeddings),
                                    total=len(smiles), desc="finding nearest neighbors"):
        similarities = torch.nn.CosineSimilarity(dim=1)(query, embeddings)
        indices = torch.argsort(similarities, descending=True)[:topk]
        query_fp = smiles_to_fingerprint(
            query_smiles, radius=fp_radius, n_bits=fp_nbits, sanitize_properties=False)
        neighbors = {"neighbor_smiles": [], "similarity": [], "fp_similarity": []}
        for index in indices:
            neighbor = id_to_smiles[molecule_ids[index].item()]
            neighbor_fp = smiles_to_fingerprint(
                neighbor, radius=fp_radius, n_bits=fp_nbits, sanitize_properties=False)
            neighbors["neighbor_smiles"].append(neighbor)
            neighbors["similarity"].append(similarities[index].item())
            neighbors["fp_similarity"].append(jaccard_similarity(query_fp, neighbor_fp))
        results[query_smiles] = neighbors
    return results


def compute_tsne(x, seed=None, perplexity=30):
    """Project embeddings into two t-SNE dimensions."""
    return TSNE(n_components=2, learning_rate="auto", init="random",
                perplexity=perplexity, random_state=seed).fit_transform(x)


def compute_pca(x, norm: bool = True, pcs=2):
    """Project embeddings with PCA, optionally standardizing each input dimension."""
    if norm:
        x = StandardScaler().fit_transform(x)
    return PCA(n_components=pcs).fit_transform(x)


def compute_umap(x, n_neighbors=15, min_dist=0.1, n_components=2,
                 metric="euclidean", seed=None):
    """Project embeddings with UMAP, importing it only when requested."""
    from umap import UMAP

    return UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                n_components=n_components, metric=metric,
                random_state=seed).fit_transform(x)
