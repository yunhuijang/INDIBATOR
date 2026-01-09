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
# Add the SA_Score folder to the system path
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from src.evaluate.lead.docking.docking import DockingVina


# Path to seed molecules CSV
SEEDS_PATH = Path(__file__).parent.parent.parent.parent / "data" / "lead_seeds.csv"


def load_seed_smiles(protein: str, seed_idx: int = 0) -> str:
    """Load seed SMILES for a given protein and seed index.

    Args:
        protein: Target protein name (parp1, fa7, 5ht1b, braf, jak2)
        seed_idx: Seed molecule index (0, 1, or 2)

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

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol, 2, 2048)

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
    sim_threshold: float
) -> tuple[dict, pd.DataFrame]:
    """Evaluate lead optimization results following GenMol methodology.

    Constraints for qualification:
    - QED >= 0.6
    - SA <= 4
    - Tanimoto similarity to seed >= sim_threshold

    Primary metric: Best docking score among qualifying molecules with valid docking.

    Args:
        protein: Target protein (parp1, fa7, 5ht1b, braf, jak2)
        smiles_list: List of generated SMILES strings
        seed_idx: Which seed molecule to use (0, 1, or 2)
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

    # Compute docking only for qualified molecules
    qualified_smiles = df.loc[df['is_qualified'], 'smiles'].tolist()
    if qualified_smiles:
        docking_scores = compute_docking_scores(protein, qualified_smiles)
        df.loc[df['is_qualified'], 'ds'] = docking_scores

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
