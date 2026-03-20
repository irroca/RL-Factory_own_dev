#!/bin/bash
# GRPO Training Script for Search R1 (Agent Lightning aligned)
# This script uses the SearchR1 tool manager that aligns with Agent Lightning's approach:
# - Uses <search>query</search> tags instead of <tool_call>
# - No JSON schema tool definitions in system prompt
# - Uses standard Search R1 instruction format from the paper

set -e -x

export MODEL_PATH=./data/models/Qwen3-0.6B # Student model 
export REWARD_MODEL_PATH=./data/models/Qwen3-0.6B # Teacher model 
export RESULT_DIR=./results/rl_factory/searchr1_aligned

python3 -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \
    algorithm.adv_estimator=grpo\
    data.train_files=/scratch/azureml/cr/j/7e6b762e2e0d44f990d5daffc11d8310/exe/wd/agent-lightning/contrib/recipes/search_r1/data/train.parquet\
    data.val_files=/scratch/azureml/cr/j/7e6b762e2e0d44f990d5daffc11d8310/exe/wd/agent-lightning/contrib/recipes/search_r1/data/test_dev_2000.parquet\
    data.train_batch_size=128\
    data.max_prompt_length=4096\
    data.max_response_length=512\
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
    actor_rollout_ref.env.name=search\
    actor_rollout_ref.env.tool_manager=searchr1\
    actor_rollout_ref.env.enable_thinking=True\
    actor_rollout_ref.env.config_path=null\
    actor_rollout_ref.env.use_process_reward=False\
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
    trainer.project_name='SearchR1_with_RL-Factory'\
    trainer.experiment_name='search_r1_Qwen3-0.6B_agl_aligned'\
    trainer.n_gpus_per_node=4\
    trainer.nnodes=1\
    trainer.val_before_train=False\
    trainer.default_local_dir=$RESULT_DIR\
    trainer.default_hdfs_dir=null\
    trainer.save_freq=50\
    trainer.test_freq=50\
    trainer.total_training_steps=600 $@ 2>&1 | tee grpo_searchr1.log
