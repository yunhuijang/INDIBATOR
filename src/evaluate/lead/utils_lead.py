# Lead optimization evaluation following GenMol paper methodology
# Reference: https://github.com/NVIDIA-Digital-Bio/genmol

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import rdkit.Chem.QED as QED
import sys
import os
from rdkit.Chem import RDConfig
from rdkit.Chem import rdFingerprintGenerator
# Add the SA_Score folder to the system path
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from src.evaluate.lead.docking.docking import DockingVina
from src.utils import get_lead_optimization_mood_seed_molecules, is_valid_smiles


# Path to seed molecules CSV
SEEDS_PATH = Path(__file__).parent.parent.parent.parent / "data" / "lead_seeds.csv"
ZINC250K_PATH = Path(__file__).parent.parent.parent.parent / "data" / "zinc250k.csv"

# MOOD evaluation constants
MOOD_HIT_THRESHOLDS = {
    'parp1': 10.0,
    'fa7': 8.5,
    '5ht1b': 8.7845,
    'jak2': 9.1,
    'braf': 10.3,
}
MOOD_SA_THRESHOLD = 4
MOOD_QED_THRESHOLD = 0.5
MOOD_SIMILARITY_THRESHOLD = 0.4

# Cache for training fingerprints (loaded once per protein/full)
_training_fps_cache = {}


def load_seed_smiles(protein: str, seed_idx: int = 0) -> str:
    """Load seed SMILES for a given protein and seed index.

    Args:
        protein: Target protein name (parp1, fa7, 5ht1b, braf, jak2, sars_cov_2)
        seed_idx: Seed molecule index (0, 1, or 2; for sars_cov_2, seed_idx to 8)

    Returns:
        SMILES string of the seed molecule
    """
    df = pd.read_csv(SEEDS_PATH)
    row = df[(df['target'] == protein) & (df['seed_idx'] == seed_idx)]
    if row.empty:
        raise ValueError(f"No seed found for protein={protein}, seed_idx={seed_idx}")
    return row['smiles'].iloc[0]


def compute_tanimoto_similarity(smiles: str, seed_smiles: str) -> float:
    """Compute Tanimoto similarity between two molecules using Morgan fingerprints.

    Args:
        smiles: SMILES string of the molecule
        seed_smiles: SMILES string of the seed molecule

    Returns:
        Tanimoto similarity score (0-1)
    """
    mol = Chem.MolFromSmiles(smiles)
    seed_mol = Chem.MolFromSmiles(seed_smiles)

    if mol is None or seed_mol is None:
        return 0.0

    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = mfgen.GetFingerprint(mol)
    seed_fp = mfgen.GetFingerprint(seed_mol)

    return DataStructs.TanimotoSimilarity(fp, seed_fp)


def compute_qed(mol) -> float:
    """Compute QED (Quantitative Estimation of Drug-likeness) score."""
    try:
        return QED.qed(mol)
    except:
        return 0.0


def compute_sa(mol) -> float:
    """Compute SA (Synthetic Accessibility) score.

    Returns:
        Raw SA score (1-10, lower is better, more synthesizable)
    """
    try:
        return sascorer.calculateScore(mol)
    except:
        return 10.0

def compute_docking_scores(protein: str, smiles: list) -> list:
    predictor = DockingVina(protein)
    
    reward = - np.array(predictor.predict(smiles))
    reward = np.clip(reward, 0, None)
    return reward

def compute_lead(
    protein: str,
    smiles_list: list,
    seed_idx: int,
    sim_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate lead optimization results following GenMol methodology.

    Constraints for qualification:
    - QED >= 0.6, SA <= 4, sim >= threshold

    Primary metric: Best docking score among qualifying molecules with valid docking.

    Args:
        protein: Target protein (parp1, fa7, 5ht1b, braf, jak2, sars_cov_2)
        smiles_list: List of generated SMILES strings
        seed_idx: Which seed molecule to use (0, 1, or 2; for sars_cov_2, seed_idx to 8)
        sim_threshold: Minimum Tanimoto similarity to seed (0.4 or 0.6)
        
    Returns:
        overall_score: dict with aggregated metrics (computed from qualifying + valid DS rows)
        all_results: DataFrame with ALL unique molecules and columns:
            - smiles, qed, sa, sim, ds (NaN if not computed)
            - is_valid: bool (molecule parsed successfully)
            - is_qualified: bool (QED >= 0.6, SA <= 4, sim >= threshold)
            - has_valid_ds: bool (docking score > 0)
    """
    num_total = len(smiles_list)
    seed_smiles = load_seed_smiles(protein, seed_idx)

    # Build DataFrame with unique non-empty SMILES
    valid_smiles = [s for s in smiles_list if s and s.strip()]
    unique_smiles = list(set(valid_smiles))
    df = pd.DataFrame({'smiles': unique_smiles})

    # Parse molecules and mark validity
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['is_valid'] = df['mol'].notna()

    # Compute properties only for valid molecules
    valid_mask = df['is_valid']
    df.loc[valid_mask, 'qed'] = df.loc[valid_mask, 'mol'].apply(compute_qed)
    df.loc[valid_mask, 'sa'] = df.loc[valid_mask, 'mol'].apply(compute_sa)
    df.loc[valid_mask, 'sim'] = df.loc[valid_mask, 'smiles'].apply(
        lambda s: compute_tanimoto_similarity(s, seed_smiles)
    )

    # Qualification check: QED >= 0.6, SA <= 4, sim >= threshold
    df['is_qualified'] = (
        df['is_valid'] &
        (df['qed'] >= 0.6) &
        (df['sa'] <= 4) &
        (df['sim'] >= sim_threshold)
    )

    # Compute docking for valid molecules
    df['ds'] = 0.0  # Initialize as float to avoid dtype warnings
    valid_smiles_list = df.loc[df['is_valid'], 'smiles'].tolist()
    if valid_smiles_list:
        docking_scores = compute_docking_scores(protein, valid_smiles_list)
        # Use explicit index-based assignment for guaranteed correct alignment
        valid_indices = df.loc[df['is_valid']].index
        for idx, score in zip(valid_indices, docking_scores):
            df.loc[idx, 'ds'] = score

    # Mark valid docking scores
    df['has_valid_ds'] = df['ds'].notna() & (df['ds'] > 0)

    # Compute overall_score from qualifying + valid DS rows
    num_unique = len(df)
    num_valid = int(df['is_valid'].sum())
    num_qualified = int(df['is_qualified'].sum())
    scoring_mask = df['is_qualified'] & df['has_valid_ds']
    num_qualified_valid_ds = int(scoring_mask.sum())

    best_ds = None
    best_smiles = None
    if scoring_mask.any():
        best_idx = df.loc[scoring_mask, 'ds'].idxmax()
        best_ds = df.loc[best_idx, 'ds']
        best_smiles = df.loc[best_idx, 'smiles']

    overall_score = {
        'num_total': num_total,
        'num_unique': num_unique,
        'num_valid': num_valid,
        'num_qualified': num_qualified,
        'num_qualified_valid_ds': num_qualified_valid_ds,
        'ratio_valid': num_valid / num_total if num_total > 0 else 0,
        'ratio_qualified': num_qualified / num_total if num_total > 0 else 0,
        'ratio_qualified_valid_ds': num_qualified_valid_ds / num_total if num_total > 0 else 0,
        'best_ds': best_ds,
        'best_smiles': best_smiles,
        'success': num_qualified_valid_ds > 0,
    }

    result_df = df.drop(columns=['mol'])
    return overall_score, result_df


def normalize_sa_score(sa_raw: float) -> float:
    """Convert raw SA score (1-10) to normalized score (0-1).

    Raw SA: 1 (easy) to 10 (hard)
    Normalized: (10 - raw) / 9, so higher is better
    """
    return (10 - sa_raw) / 9


def load_training_fingerprints(protein: str = None) -> list:
    """Load and compute fingerprints for training set with caching.

    Args:
        protein: If provided, loads protein-specific seed molecules.
                 If None, loads full ZINC250K dataset.

    Returns:
        List of Morgan fingerprints (cached after first load).
    """
    global _training_fps_cache

    # Use cache key based on protein
    cache_key = protein if protein else '__zinc250k__'
    if cache_key in _training_fps_cache:
        return _training_fps_cache[cache_key]

    # Load molecules based on protein or full ZINC250K
    if protein is None:
        zinc_train_df = pd.read_csv(ZINC250K_PATH)
        seed_molecules = zinc_train_df['smiles'].tolist()
    else:
        seed_molecules = get_lead_optimization_mood_seed_molecules(protein)

    # Compute fingerprints
    fps = []
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    for smiles in seed_molecules:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fps.append(mfgen.GetFingerprint(mol))
        except:
            continue

    # Cache and return
    _training_fps_cache[cache_key] = fps
    return fps


def compute_max_similarity_to_training(smiles: str, training_fps: list) -> float:
    """Compute max Tanimoto similarity between molecule and training set."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1.0  # Invalid molecules are not novel

    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = mfgen.GetFingerprint(mol)
    if not training_fps:
        return 0.0
    max_sim = max(DataStructs.TanimotoSimilarity(fp, train_fp) for train_fp in training_fps)
    return max_sim


def compute_max_similarity_to_training_bulk(smiles_list: list, training_fps: list) -> list:
    """Compute max Tanimoto similarity for multiple molecules efficiently.

    Uses RDKit's BulkTanimotoSimilarity for better performance.

    Args:
        smiles_list: List of SMILES strings
        training_fps: List of precomputed Morgan fingerprints

    Returns:
        List of max similarity values (same order as input)
    """
    if not training_fps:
        return [0.0] * len(smiles_list)

    results = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append(1.0)  # Invalid molecules are not novel
            continue
        mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fp = mfgen.GetFingerprint(mol)
        # BulkTanimotoSimilarity is much faster than individual comparisons
        similarities = DataStructs.BulkTanimotoSimilarity(fp, training_fps)
        results.append(max(similarities))
    return results


def compute_lead_mood(
    protein: str,
    smiles_list: list,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate lead optimization using MOOD methodology.

    MOOD criteria:
    - QED > 0.5
    - Normalized SA > 0.556 (raw SA < 5)
    - Novelty: max similarity to training < 0.4

    Args:
        protein: Target protein (parp1, fa7, 5ht1b, braf, jak2)
        smiles_list: List of generated SMILES strings

    Returns:
        overall_score: dict with MOOD metrics
        result_df: DataFrame with all molecules and computed properties
    """
    num_total = len(smiles_list)

    # Remove empty SMILES
    valid_smiles = [s for s in smiles_list if is_valid_smiles(s)]
    validity = len(valid_smiles) / num_total if num_total > 0 else 0

    # Build DataFrame
    df = pd.DataFrame({'smiles': valid_smiles})

    # Parse molecules
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['is_valid'] = df['mol'].notna()

    # Uniqueness
    uniqueness = len(set(df['smiles'])) / len(df) if len(df) > 0 else 0

    # Compute properties for valid molecules
    valid_mask = df['is_valid']
    df.loc[valid_mask, 'qed'] = df.loc[valid_mask, 'mol'].apply(compute_qed)
    df.loc[valid_mask, 'sa'] = df.loc[valid_mask, 'mol'].apply(compute_sa)

    # Compute similarity to training set using bulk operations for speed
    valid_smiles_list = df.loc[valid_mask, 'smiles'].tolist()

    training_fps_protein = load_training_fingerprints(protein)
    if training_fps_protein:
        similarities_protein = compute_max_similarity_to_training_bulk(valid_smiles_list, training_fps_protein)
        df.loc[valid_mask, 'sim'] = similarities_protein
    else:
        df['sim'] = 0.0  # If no training data, assume all novel

    training_fps_all = load_training_fingerprints()
    if training_fps_all:
        similarities_all = compute_max_similarity_to_training_bulk(valid_smiles_list, training_fps_all)
        df.loc[valid_mask, 'sim_total'] = similarities_all
    else:
        df['sim_total'] = 0.0  # If no training data, assume all novel
    
    novelty = len(df[df['sim'] < MOOD_SIMILARITY_THRESHOLD]) / len(df) if len(df) > 0 else 0
    novelty_total = len(df[df['sim_total'] < MOOD_SIMILARITY_THRESHOLD]) / len(df) if len(df) > 0 else 0
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['smiles'])

    # Compute docking scores
    df['ds'] = 0.0
    valid_smiles_for_docking = df.loc[df['is_valid'], 'smiles'].tolist()
    if valid_smiles_for_docking:
        docking_scores = compute_docking_scores(protein, valid_smiles_for_docking)
        valid_indices = df.loc[df['is_valid']].index
        for idx, score in zip(valid_indices, docking_scores):
            df.loc[idx, 'ds'] = score

    # Apply MOOD filtering
    filtered_df = df[
        (df['qed'] > MOOD_QED_THRESHOLD) &
        (df['sa'] < MOOD_SA_THRESHOLD) &
        (df['sim'] < MOOD_SIMILARITY_THRESHOLD)
    ].copy()
    filtered_df = filtered_df.sort_values(by='ds', ascending=False)
    
    filtered_df_total = df[
        (df['qed'] > MOOD_QED_THRESHOLD) &
        (df['sa'] < MOOD_SA_THRESHOLD) &
        (df['sim_total'] < MOOD_SIMILARITY_THRESHOLD)
    ].copy()
    filtered_df_total = filtered_df_total.sort_values(by='ds', ascending=False)
    overall_score = {'num_total': num_total, 'seed/novelty': novelty, 'total/novelty': novelty_total}

    for i, df_now in enumerate([filtered_df, filtered_df_total]):
        # Top 5% statistics
        num_top5 = max(1, int(num_total * 0.05))
        if len(df_now) >= num_top5:
            top5_ds = df_now.iloc[:num_top5]['ds']
            top_ds_mean = top5_ds.mean()
            top_ds_std = top5_ds.std()
        else:
            top_ds_mean = df_now['ds'].mean() if len(df_now) > 0 else 0
            top_ds_std = df_now['ds'].std() if len(df_now) > 1 else 0

        # Hit ratio
        hit_thr = MOOD_HIT_THRESHOLDS.get(protein)
        num_hits = len(df_now[df_now['ds'] > hit_thr])
        hit_ratio = num_hits / num_total if num_total > 0 else 0
        if i == 0:
            index = 'seed'
        else:
            index = 'total'
        overall_score[f'{index}/validity'] = validity
        overall_score[f'{index}/uniqueness'] = uniqueness
        overall_score[f'{index}/top_ds_mean'] = top_ds_mean
        overall_score[f'{index}/top_ds_std'] = top_ds_std
        overall_score[f'{index}/hit_ratio'] = hit_ratio
        overall_score[f'{index}/num_filtered'] = len(df_now)
        overall_score[f'{index}/num_hits'] = num_hits
        overall_score[f'{index}/hit_threshold'] = hit_thr
        
    result_df = df.drop(columns=['mol'], errors='ignore')
    return overall_score, result_df