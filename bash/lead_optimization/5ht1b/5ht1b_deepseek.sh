for seed_mol_index in 0 1 2; do
        uv run python -m src.agent.main \
        --task_name lead_optimization/5ht1b \
        --seed_mol_index $seed_mol_index \
        --sim_threshold 0.4 \
        --num_candidates 1000 \
        --scientists 50 \
        --wandb_mode online \
        --model deepseek-chat \
        --is_rag_keyword \
        --num_mols_per_scientist 30 \
        --rounds 10 \
        --is_self_critique_on
done
