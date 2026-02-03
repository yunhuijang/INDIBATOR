CUDA_VISIBLE_DEVICES=3 uv run python -m src.agent.main \
    --task_name boltz/TYK2 \
    --num_candidates 1000 \
    --scientists 50 \
    --wandb_mode online \
    --model deepseek-chat \
    --num_mols_per_scientist 50 \
    --is_rag_keyword