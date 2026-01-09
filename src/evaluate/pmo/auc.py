
import numpy as np
from rdkit import Chem


def top_auc(buffer:dict, top_n:int, finish:bool, freq_log:int, max_oracle_calls:int):
    sum_auc = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1]))
    buffer_max_idx = ordered_results[-1][1][1]  # last components' [1][1]: score

    for idx in range(freq_log, min(buffer_max_idx, max_oracle_calls), freq_log):
        temp_result = [item for item in ordered_results if item[1][1] <= idx]
        if len(temp_result) == 0:
            continue
        top_n_now = np.mean(
            [
                item[1][0]
                for item in sorted(temp_result, key=lambda kv: kv[1][0], reverse=True)[
                    :top_n
                ]
            ]
        )
        sum_auc += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx

    final_result = sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True)[
        :top_n
    ]
    top_n_now = np.mean([item[1][0] for item in final_result])
    sum_auc += (buffer_max_idx - called) * (top_n_now + prev) / 2

    if finish and buffer_max_idx < max_oracle_calls:
        sum_auc += (max_oracle_calls - buffer_max_idx) * top_n_now

    return sum_auc / max_oracle_calls

def compute_topk_auc(smiles_scores: list[(str, float)], top_k:int=5, max_oracle_calls:int=1000,
    freq_log:int=1, finish:bool=False
):
    # smiles_scores = []

    print("number of generated smiles", len(smiles_scores))
    # Step 2: Remove invalid or duplicate SMILES
    seen = set()
    mol_buffer = {}
    for i, (smi, score) in enumerate(smiles_scores):
        mol = Chem.MolFromSmiles(smi)
        if mol:  # valid SMILES
            canonical_smi = Chem.MolToSmiles(mol)  # Canonicalize
            if canonical_smi not in seen:
                # Store as {canonical_smi: [score, generation index]}
                mol_buffer[canonical_smi] = [score, i + 1]
                seen.add(canonical_smi)
            else:
                continue  # duplicated canonical SMILES
        else:
            continue  # invalid SMILES

    # Step 3: Compute AUC
    auc = top_auc(
        mol_buffer,
        top_k,
        finish=finish,
        freq_log=freq_log,
        max_oracle_calls=max_oracle_calls,
    )
    print(f"Top-{top_k} AUC Score: {auc:.4f}")
    return auc, mol_buffer