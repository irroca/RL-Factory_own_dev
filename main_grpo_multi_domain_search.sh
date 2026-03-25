#!/bin/bash
# GRPO Training Script for Multi-Domain Search R1 (Agent Lightning aligned)
# This script uses the MultiDomainSearchR1Manager that supports domain-aware retrieval:
# - Uses <search>query</search> + <domain>domain_name</domain> tags
# - Calls multi-domain retrieval server (POST /retrieve with domain parameter)
# - Supports biomedical, financial, science domains
# - Uses standard Search R1 instruction format extended with domain routing
#
# Prerequisites:
#   1. Start the multi-domain retrieval server:
#      cd /path/to/Search_agent_checkpoints/multi_domain_retriever
#      bash multi_domain_launch.sh train 8001   # training data
#      (Optionally on another port for test data)
#
#   2. Prepare training data (if not already available):
#      python scripts/multi_domain_search_data.py \
#          --from_local \
#          --train_files /path/to/train.jsonl \
#          --test_files /path/to/test.jsonl \
#          --local_dir ./data/multi_domain_search
#
# Hyperparameters are aligned with main_grpo_searchr1.sh for fair comparison.
#
# Configuration switches (append to command line to override):
#   Reward mode:
#     actor_rollout_ref.env.reward_mode=agl        # (default) pure Exact Match
#     actor_rollout_ref.env.reward_mode=multi_dim   # EM + format reward (search/domain/answer tag validity)
#
#   Interaction format (tool_manager):
#     actor_rollout_ref.env.tool_manager=multi_domain_searchr1            # (default) raw text concatenation (Search R1 style)
#     actor_rollout_ref.env.tool_manager=multi_domain_searchr1_multistep  # user→assistant turn alternation
#
#   Example: use multistep format with multi-dim reward:
#     bash main_grpo_multi_domain_search.sh \
#         actor_rollout_ref.env.tool_manager=multi_domain_searchr1_multistep \
#         actor_rollout_ref.env.reward_mode=multi_dim

set -e -x

export MODEL_PATH=./data/models/Qwen3-0.6B  # Student model
export REWARD_MODEL_PATH=./data/models/Qwen3-0.6B  # Teacher model
export RESULT_DIR=./results/rl_factory/multi_domain_search_aligned

# Multi-domain retrieval server URL (must be started before training)
export RETRIEVAL_URL=http://127.0.0.1:8001

# ---- Data paths ----
# Option A: Use pre-built multi-domain parquet files
TRAIN_DATA=./data/multi_domain_search/train.parquet
VAL_DATA=./data/multi_domain_search/test.parquet

# Option B (fallback): Use same data as SearchR1 for testing the pipeline
# TRAIN_DATA=/scratch/azureml/cr/j/7e6b762e2e0d44f990d5daffc11d8310/exe/wd/agent-lightning/contrib/recipes/search_r1/data/train.parquet
# VAL_DATA=/scratch/azureml/cr/j/7e6b762e2e0d44f990d5daffc11d8310/exe/wd/Search_agent_checkpoints/test_dev_2000.parquet

python3 -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \
    algorithm.adv_estimator=grpo\
    data.train_files=$TRAIN_DATA\
    data.val_files=$VAL_DATA\
    data.train_batch_size=128\
    data.max_prompt_length=4096\
    data.max_response_length=4096\
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.model.use_remove_padding=True\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.optim.lr=1e-6\
    actor_rollout_ref.actor.ppo_mini_batch_size=32\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8\
    actor_rollout_ref.actor.use_kl_loss=True\
    actor_rollout_ref.actor.kl_loss_coef=0.001\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl\
    actor_rollout_ref.actor.fsdp_config.param_offload=True\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
    actor_rollout_ref.actor.state_masking=True\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.name=vllm\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75\
    actor_rollout_ref.rollout.n=5\
    actor_rollout_ref.rollout.max_turns=4\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16\
    actor_rollout_ref.ref.fsdp_config.param_offload=False\
    actor_rollout_ref.rollout.enforce_eager=False\
    actor_rollout_ref.rollout.free_cache_engine=True\
    actor_rollout_ref.env.name=multi_domain_search\
    actor_rollout_ref.env.tool_manager=multi_domain_searchr1\
    actor_rollout_ref.env.enable_thinking=True\
    actor_rollout_ref.env.config_path=null\
    actor_rollout_ref.env.use_process_reward=False\
    actor_rollout_ref.env.retrieval_url=$RETRIEVAL_URL\
    actor_rollout_ref.env.retrieval_topk=3\
    reward_rollout.if_use_reward_rollout=False\
    reward_rollout.rollout.tensor_model_parallel_size=4\
    reward_rollout.rollout.gpu_memory_utilization=0.75\
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\
    reward_rollout.rollout.free_cache_engine=True\
    reward_rollout.rollout.response_length=2048\
    reward_model.reward_manager=parallel\
    algorithm.kl_ctrl.kl_coef=0.001\
    trainer.critic_warmup=0\
    trainer.logger=['console','wandb']\
    trainer.project_name='MultiDomain_SearchR1_with_RL-Factory'\
    trainer.experiment_name='multi_domain_search_r1_Qwen3-0.6B_aligned'\
    trainer.n_gpus_per_node=4\
    trainer.nnodes=1\
    trainer.val_before_train=True\
    trainer.default_local_dir=$RESULT_DIR\
    trainer.default_hdfs_dir=null\
    trainer.save_freq=40\
    trainer.test_freq=40\
    trainer.total_training_steps=400 $@ 2>&1 | tee grpo_multi_domain_search.log
