# Boltz binding affinity evaluation for molecular optimization

import subprocess
import tempfile
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from multiprocessing import Pool
import pandas as pd
import yaml
from tqdm import tqdm

from src.utils import is_valid_smiles

logger = logging.getLogger(__name__)

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BOLTZ_YAML_DIR = PROJECT_ROOT / "boltz" / "yaml"
BOLTZ_MSA_DIR = PROJECT_ROOT / "boltz" / "msa"

# Valid proteins (all 8 with MSA files)
VALID_PROTEINS = ['CA2', 'CDK2', 'DHFR', 'FABP4', 'JNK1', 'P38', 'THROMBIN', 'TYK2']


def load_protein_template(protein: str) -> Dict[str, Any]:
    """Load the YAML template for a protein.

    Args:
        protein: Protein name (e.g., 'CA2')

    Returns:
        Dictionary with protein sequence and MSA path
    """
    yaml_path = BOLTZ_YAML_DIR / f"{protein}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML template not found: {yaml_path}")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    return config


def create_prediction_yaml(protein: str, smiles: str, output_path: Path) -> None:
    """Create a YAML file for boltz prediction with a single ligand.

    Args:
        protein: Protein name
        smiles: SMILES string for the ligand
        output_path: Path to write the YAML file
    """
    template = load_protein_template(protein)

    # Create single-ligand configuration
    # Get version from template (default to 1)
    version = 1

    # Get protein info from template
    protein_info = template['sequences'][0]['protein']

    # Create new config with single ligand
    config = {
        'version': version,
        'sequences': [
            {
                'protein': {
                    'id': protein_info['id'],
                    'sequence': protein_info['sequence'],
                    'msa': protein_info['msa']
                }
            },
            {
                'ligand': {
                    'id': 'B',
                    'smiles': smiles
                }
            }
        ],
        'properties': [
            {
                'affinity': {
                    'binder': 'B'
                }
            }
        ]
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_boltz_single(args: Tuple[str, str, str, int], run_name: str) -> Dict[str, Any]:
    """Run boltz prediction for a single molecule.

    Args:
        args: Tuple of (smiles, protein, temp_dir, index)

    Returns:
        Dictionary with prediction results
    """
    smiles, protein, temp_dir, idx = args

    temp_path = Path(temp_dir)
    yaml_name = f"{run_name}_{protein}_{idx}"
    yaml_path = temp_path / f"{yaml_name}.yaml"
    # Save output to boltz/output/{protein} folder instead of temp directory
    out_dir = PROJECT_ROOT / "boltz" / "output" / protein / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving boltz output to: {out_dir}")

    try:
        # Create YAML file
        create_prediction_yaml(protein, smiles, yaml_path)

        # Run boltz predict (using uv run since boltz is managed by uv)
        # uv run needs to be executed from project root where pyproject.toml is located
        # Use absolute paths to avoid issues with working directory
        yaml_path_abs = yaml_path.resolve()
        out_dir_abs = out_dir.resolve()
        cmd = f"uv run boltz predict {yaml_path_abs} --out_dir {out_dir_abs}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=PROJECT_ROOT,  # Run from project root where pyproject.toml is
            env=os.environ.copy()  # Preserve environment variables
        )

        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else result.stdout[:500]
            logger.warning(f"Boltz failed for {smiles[:50]}... (returncode={result.returncode}): {error_msg}")
            return {
                'smiles': smiles,
                'affinity_pred_value': 0,
                'affinity_probability_binary': 0,
                'is_valid': False,
                'error': error_msg
            }

        # Parse results - boltz creates output in: {out_dir}/boltz_results_{yaml_name}/predictions/{yaml_name}/affinity_{yaml_name}.json
        # Try multiple possible output locations
        possible_paths = [
            out_dir / "boltz_results" / yaml_name / "predictions" / yaml_name / f"affinity_{yaml_name}.json",
            out_dir / f"boltz_results_{yaml_name}" / "predictions" / yaml_name / f"affinity_{yaml_name}.json",
            out_dir / "predictions" / yaml_name / f"affinity_{yaml_name}.json",
            out_dir / yaml_name / f"affinity_{yaml_name}.json",
        ]
        
        affinity_file = None
        for path in possible_paths:
            if path.exists():
                affinity_file = path
                break
        
        # If still not found, search recursively
        if affinity_file is None and out_dir.exists():
            for json_file in out_dir.rglob(f"affinity_{yaml_name}.json"):
                affinity_file = json_file
                break
        
        if affinity_file is None or not affinity_file.exists():
            error_msg = f"Affinity file not found. Searched: {possible_paths}"
            if result.stdout:
                logger.warning(f"Boltz stdout: {result.stdout[:500]}")
            if result.stderr:
                logger.warning(f"Boltz stderr: {result.stderr[:500]}")
            logger.warning(error_msg)
            return {
                'smiles': smiles,
                'affinity_pred_value': 0,
                'affinity_probability_binary': 0,
                'is_valid': False,
                'error': error_msg
            }

        with open(affinity_file) as f:
            affinity_data = json.load(f)

        return {
            'smiles': smiles,
            'affinity_pred_value': affinity_data.get('affinity_pred_value'),
            'affinity_probability_binary': affinity_data.get('affinity_probability_binary'),
            'is_valid': True,
            'error': ''
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"Boltz timeout for {smiles[:50]}...")
        return {
            'smiles': smiles,
            'affinity_pred_value': 0,
            'affinity_probability_binary': 0,
            'is_valid': False,
            'error': 'Timeout'
        }
    except Exception as e:
        logger.error(f"Error processing {smiles[:50]}...: {e}")
        return {
            'smiles': smiles,
            'affinity_pred_value': 0,
            'affinity_probability_binary': 0,
            'is_valid': False,
            'error': str(e)
        }


def compute_boltz(
    protein: str,
    candidates: List[Dict[str, Any]],
    run_name: str,
    num_workers: int = 16
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Compute boltz binding affinity for a list of molecules.

    Args:
        protein: Target protein name (e.g., 'CA2')
        smiles_list: List of SMILES strings to evaluate
        num_workers: Number of parallel workers (default: 16)

    Returns:
        overall_score: dict with aggregated metrics
        result_df: DataFrame with per-molecule results
    """
    protein_upper = protein.upper()
    if protein_upper not in VALID_PROTEINS:
        raise ValueError(f"Invalid protein: {protein}. Must be one of {VALID_PROTEINS}")

    num_total = len(candidates)
    logger.info(f"Computing boltz affinity for {num_total} molecules against {protein_upper} with {num_workers} workers")

    # Create temp directory and process molecules
    
    valid_candidates = [(i, c) for i, c in enumerate(candidates) if is_valid_smiles(c['smiles'])]
    with tempfile.TemporaryDirectory() as base_temp:
        # Create individual temp directories for each molecule
        args_list = []
        for original_idx, candidate in valid_candidates:
            smiles = candidate['smiles']
            score = candidate.get('score', -1)
            score_details = candidate.get('score_details', {'affinity_pred_value': -1, 'affinity_probability_binary': -1})
            mol_temp_dir = Path(base_temp) / f"mol_{original_idx}"
            mol_temp_dir.mkdir(parents=True, exist_ok=True)
            args_list.append((smiles, protein_upper, str(mol_temp_dir), original_idx, score, score_details))

        # Run predictions in parallel with limited workers (each worker uses GPU)
        results = []
        for i, args in enumerate(tqdm(args_list, desc="Running Boltz predictions")):
            score = args[-2]
            score_details = args[-1]
            if score > 0:
                result = {
                    'smiles': smiles,
                    'affinity_pred_value': score_details['affinity_pred_value'],
                    'affinity_probability_binary': score_details['affinity_probability_binary'],
                    'is_valid': True,
                    'error': '',
                }
            else:
                result = run_boltz_single(args[:-2], run_name)
                results.append(result)
            logger.info(f"Processed {i+1}/{len(args_list)}: {result['smiles'][:30]}... -> {result.get('affinity_pred_value')}")

    # Build DataFrame
    df = pd.DataFrame(results)

    # Compute overall metrics
    valid_df = df[df['is_valid'] == True]
    num_valid = len(valid_df)

    if num_valid > 0:
        avg_affinity = valid_df['affinity_pred_value'].mean()
        avg_probability = valid_df['affinity_probability_binary'].mean()
        # Best affinity is the LOWEST value (strongest binding)
        best_idx = valid_df['affinity_pred_value'].idxmin()
        best_affinity = valid_df.loc[best_idx, 'affinity_pred_value']
        best_smiles = valid_df.loc[best_idx, 'smiles']
    else:
        avg_affinity = 0
        avg_probability = 0
        best_affinity = 0
        best_smiles = ''

    overall_score = {
        'num_total': num_total,
        'num_valid': num_valid,
        'ratio_valid': num_valid / num_total if num_total > 0 else 0,
        'avg_affinity': avg_affinity,
        'avg_probability': avg_probability,
        'best_affinity': best_affinity,
        'best_smiles': best_smiles,
        'score_avg': (6-avg_affinity)*1.364,
        'score_best': (6-best_affinity)*1.364,
    }

    # Add score column (kcal/mol affinity so higher = better)
    df['score'] = df['affinity_pred_value'].apply(lambda x: (6-x)*1.364 if x is not None else None)

    logger.info(f"Boltz evaluation complete: {num_valid}/{num_total} valid, best affinity: {best_affinity}")

    return overall_score, df
