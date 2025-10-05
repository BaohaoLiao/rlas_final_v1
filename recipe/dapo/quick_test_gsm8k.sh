 #!/usr/bin/env bash
set -xeuo pipefail
export NCCL_P2P_DISABLE=1
# --- 主要配置 (请根据你的环境修改) ---
# 1. 设置你想使用的GPU数量和设备ID
NGPUS=4 # 你准备在这台机器上使用的GPU总数
export CUDA_VISIBLE_DEVICES="6,7,8,9" # GPU的ID，数量应与 NGPUS 匹配

# 2. 确认模型和数据路径
# home目录，用于存放数据和模型，避免硬编码
DATA_HOME="${HOME}/verl"
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" # 确保这里的模型大小 (7B) 和实验名称一致
# ---

# 工作目录和项目设置
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
project_name='DAPO'
exp_name='DAPO-Qwen2.5-7b-MATH-single-node' # 更新了实验名称以作区分

# 算法超参数 (保留原始逻辑)
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 1))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
train_prompt_bsz=256 # 注意：如果GPU显存不足，可能需要减小这个值
n_resp_per_prompt=12
train_prompt_mini_bsz=64

# 路径设置 (使用更灵活的方式)
CKPTS_DIR="${DATA_HOME}/ckpts/${project_name}/${exp_name}"
TRAIN_FILE=/home/wx13/data/gsm8k_new/train.parquet
#"${DATA_HOME}/data/dapo-math-17k.parquet"
TEST_FILE=/home/wx13/data/gsm8k_new/test.parquet
#"${DATA_HOME}/data/aime-2024.parquet"

# 算法和性能相关参数
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
sp_size=1 # 单节点上一般不使用或使用很小的序列并行，设为1=禁用
tp_size=1 # 在单节点内，可以根据模型大小和GPU数量设置张量并行，例如2
fsdp_size=${NGPUS} # 在单节点上，FSDP的大小通常就是你使用的GPU总数
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=False # 对于单节点，当显存紧张时，offload到CPU内存很有用

# 启动训练
python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=100 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
