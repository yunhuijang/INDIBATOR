for i in {1..2}; do
uv run python -m src.agent.main \
        --task_name pmo/GSK3B \
        --num_candidates 1000 \
        --scientists 50 \
        --wandb_mode online \
        --num_mols_per_scientist 30 \
        --model deepseek-chat \
        --rounds 20 \
        --is_rag_keyword
done