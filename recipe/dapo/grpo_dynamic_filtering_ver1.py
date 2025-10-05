# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                uids = [str(uuid.uuid4()) for _ in range(len(new_batch))]
                uids_array = np.array(uids, dtype=object)

                # 2. 将 UID 赋给 new_batch
                new_batch.non_tensor_batch["uid"] = uids_array


                # pop those keys for generation
                if True:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                #gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                gen_batch.non_tensor_batch["uid"] = uids_array
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    if True:
                        # Wrap the entire training step to measure timing
                        # [MODIFIED] START: 使用新的动态采样算法替换原有逻辑
                        # =================================================================
                        # 动态采样算法描述:
                        # 1. 对一个批次的 prompt 进行多轮 (最多4轮) 数据生成。
                        # 2. 每个 prompt 的退出条件是收集到 >= 2 个正样本 (reward > 0.9)。
                        # 3. 4轮结束后，为每个 prompt 构建最终的训练样本（最多8条）。
                        # 4. 构建策略：优先选最多4条正样本，然后用负样本补齐到8条。
                        # 5. 记录第一轮iteration的平均reward作为critic/real_score（未过滤前的真实奖励）。
                        # 
                        # 核心思想：对难题多采样，确保每个prompt都有足够的正样本用于训练。
                        # =================================================================
                        max_rounds = 4  # 最多生成轮数
                        positive_threshold = 0.9 # 正样本的 reward 阈值
                        min_positive_samples = 4 # 每个 prompt 的退出条件
                        max_total_samples_per_prompt = 8 # 每个 prompt 最终采样数
                        max_positive_samples_per_prompt = 4 # 最终采样中最多包含的正样本数

                        original_gen_batch = gen_batch
                        original_prompt_uids = list(original_gen_batch.non_tensor_batch["uid"])
                        # 为原始 prompt batch 中的每个 prompt 分配一个唯一的整数索引，方便后续操作
                        #original_prompt_uids = [str(uuid.uuid4()) for _ in range(len(original_gen_batch))]
                        #original_gen_batch.non_tensor_batch["uid"] = np.array(original_prompt_uids, dtype=object)

                        all_generated_data = [] # 存储所有轮次生成的数据
                        active_gen_batch = original_gen_batch # 当前需要进行生成的 prompt batch
                        first_iteration_rewards = [] # 存储第一次iteration的所有reward，用于计算critic/real_score
                        # [MODIFIED] START: 循环生成与动态退出
                        for round_idx in range(max_rounds):
                            print(f"Generation Round {round_idx + 1}/{max_rounds}. Active prompts: {len(active_gen_batch)}")
                            
                            if active_gen_batch is None or len(active_gen_batch) == 0:
                                print("All prompts have met the exit condition. Stopping generation early.")
                                break

                            # --- 生成和评估流程 (与原代码类似) ---
                            # a. 为当前活跃的 prompt 生成 response
                            active_gen_batch_repeated = active_gen_batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True)
                            
                            # [FIXED] Ensure batch size is divisible by number of workers to avoid chunking error
                            dp_size = self.actor_rollout_wg.dp_size if hasattr(self.actor_rollout_wg, 'dp_size') else 8
                            batch_size = len(active_gen_batch_repeated)
                            padding_applied = False
                            if batch_size % dp_size != 0:
                                # Pad the batch to make it divisible by dp_size
                                padding_needed = dp_size - (batch_size % dp_size)
                                print(f"Padding batch from {batch_size} to {batch_size + padding_needed} to make it divisible by {dp_size}")
                                # Repeat the last few samples to pad
                                indices_to_repeat = list(range(batch_size - padding_needed, batch_size))
                                if len(indices_to_repeat) == 0:
                                    indices_to_repeat = [batch_size - 1] * padding_needed
                                padding_batch = active_gen_batch_repeated[indices_to_repeat]
                                active_gen_batch_repeated = DataProto.concat([active_gen_batch_repeated, padding_batch])
                                padding_applied = True
                                
                            with marked_timer(f"gen_round_{round_idx}", timing_raw, "red"):
                                gen_output = self.actor_rollout_wg.generate_sequences(active_gen_batch_repeated)
                                
                            # Remove padding if we added any
                            if padding_applied:
                                gen_output = gen_output[:batch_size]
                                # Also trim the active_gen_batch_repeated to match
                                active_gen_batch_repeated = active_gen_batch_repeated[:batch_size]
                            #print('hi!')
                            # b. 准备计算 reward
                            if "timing" in gen_output.meta_info:
                                # [FIXED] Accumulate timing from multiple rounds instead of overwriting
                                round_timing = gen_output.meta_info["timing"]
                                for key, value in round_timing.items():
                                    timing_raw[key] += value  # Accumulate timing across rounds
                                gen_output.meta_info.pop("timing")

                            #keys_to_remove = list(active_gen_batch_repeated.batch.keys())
                            #gen_output.pop(batch_keys=keys_to_remove)
                            #current_round_data = active_gen_batch_repeated
                            #current_round_data = current_round_data.union(gen_output)
                            # [FIXED] Better approach: Use gen_output as base (complete sequences)
                            # and selectively add non-conflicting data from active_gen_batch_repeated
                            
                            # Start with gen_output which has complete sequences
                            current_round_data = gen_output
                            
                            # Add non-tensor data from active_gen_batch_repeated (like UIDs)
                            for key, value in active_gen_batch_repeated.non_tensor_batch.items():
                                if key not in current_round_data.non_tensor_batch:
                                    current_round_data.non_tensor_batch[key] = value
                                # If key exists, keep gen_output's version (more complete)
                            # Note: gen_output already contains complete input_ids, attention_mask, etc.
                            # so we don't need to reconstruct them


                            # 获取元信息
                            active_uids = active_gen_batch.non_tensor_batch["uid"]
                            all_metadata_uids = list(new_batch.non_tensor_batch["uid"])
                            indices_to_keep = [all_metadata_uids.index(uid) for uid in active_uids]
                            active_metadata_batch = new_batch[indices_to_keep]
                            active_metadata_repeated = active_metadata_batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True)
                            # Ensure metadata batch size matches current_round_data
                            if padding_applied:
                                active_metadata_repeated = active_metadata_repeated[:batch_size]
                            current_round_data = current_round_data.union(active_metadata_repeated)

                            # c. 计算 reward
                            #print(current_round_data[:10])
                            #while True:
                            #    pass
                            with marked_timer(f"reward_round_{round_idx}", timing_raw, "yellow"):
                                # we combine with rule-based rm
                                reward_extra_infos_dict: dict[str, list]
                                try:
                                    reward_result = self.reward_fn(current_round_data, return_dict=True)
                                    reward_tensor = reward_result["reward_tensor"]
                                    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                                except Exception as e:
                                    print(f"Error in reward_fn: {e}")
                                    reward_tensor = self.reward_fn(current_round_data)
                                    reward_extra_infos_dict = {}
                                    
                                current_round_data.batch["token_level_scores"] = reward_tensor

                                # 同样应用KL惩罚
                                if self.config.algorithm.use_kl_in_reward:
                                    current_round_data, kl_metrics = apply_kl_penalty(
                                        current_round_data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                    )
                                    metrics.update(kl_metrics) # 注意：metrics可能会被覆盖，但对于最终的batch是准确的
                                else:
                                    current_round_data.batch["token_level_rewards"] = current_round_data.batch["token_level_scores"]


                            # d. 将当前轮次生成的数据存起来
                            all_generated_data.append(current_round_data)
                            
                            # 记录第一次iteration的所有reward用于计算critic/real_score
                            # 这里记录的是未经过任何过滤的原始奖励，反映模型的真实表现
                            if round_idx == 0:
                                # 计算当前轮次的sequence-level rewards
                                seq_rewards = current_round_data.batch["token_level_rewards"].sum(dim=-1)
                                first_iteration_rewards.extend(seq_rewards.tolist())

                            # --- 检查退出条件 ---
                            # e. 合并至今为止所有生成的数据
                            consolidated_data = DataProto.concat(all_generated_data)
                            consolidated_data.non_tensor_batch["seq_final_reward"] = (
                                consolidated_data.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )

                            prompt_uid_to_pos_count = defaultdict(int)
                            for uid, reward in zip(consolidated_data.non_tensor_batch["uid"], consolidated_data.non_tensor_batch["seq_final_reward"]):
                                if reward > positive_threshold:
                                    prompt_uid_to_pos_count[uid] += 1
                            
                            remaining_uids = []
                            # 注意：这里应该遍历 original_prompt_uids 来确保遍历所有prompt
                            for uid in original_prompt_uids:
                                if prompt_uid_to_pos_count.get(uid, 0) < min_positive_samples:
                                    remaining_uids.append(uid)
                            
                            if remaining_uids:
                                keep_indices = [i for i, uid in enumerate(original_gen_batch.non_tensor_batch["uid"]) if uid in remaining_uids]
                                if not keep_indices: # 如果所有prompt都已完成
                                     active_gen_batch = None
                                else:
                                     active_gen_batch = original_gen_batch[keep_indices]
                            else:
                                active_gen_batch = None
                        

                        #########################################################
                        if not all_generated_data:
                            # 如果4轮都没有生成任何数据（不太可能发生，除非输入为空）
                            print("Warning: No data was generated after all rounds. Skipping training step.")
                            continue

                        final_data = DataProto.concat(all_generated_data)
                        final_data.non_tensor_batch["seq_final_reward"] = (
                            final_data.batch["token_level_rewards"].sum(dim=-1).numpy()
                        )
                        #########################################################
                        # save final data to file

                        #save_path = "/home/wx13/reinforceflow/verl/final_data_debug.pt"  # 定义保存路径和文件名

                        #torch.save(final_data, save_path)
                        
                        #########################################################
                        uid_to_traj_indices = defaultdict(list)
                        for i, uid in enumerate(final_data.non_tensor_batch["uid"]):
                            uid_to_traj_indices[uid].append(i)
                        
                        final_kept_indices = []
                        for uid in original_prompt_uids:
                            indices = uid_to_traj_indices.get(uid, [])
                            if not indices: 
                                continue

                            positive_indices = [i for i in indices if final_data.non_tensor_batch["seq_final_reward"][i] > positive_threshold]
                            negative_indices = [i for i in indices if final_data.non_tensor_batch["seq_final_reward"][i] <= positive_threshold]
                            
                            # 2. 按新策略进行采样
                            # Step A: 优先选择最多 `max_positive_samples_per_prompt` (4) 个正样本
                            final_samples_for_prompt = positive_indices[:max_positive_samples_per_prompt]

                            # Step B: 尝试用负样本补足到 `max_total_samples_per_prompt` (8) 个
                            num_neg_needed = max_total_samples_per_prompt - len(final_samples_for_prompt)
                            final_samples_for_prompt.extend(negative_indices[:num_neg_needed])

                            # Step C: [核心修改] 如果负样本不够，导致总数仍不足8个，则用剩余的正样本来“兜底”
                            num_still_needed = max_total_samples_per_prompt - len(final_samples_for_prompt)
                            if num_still_needed > 0:
                                # 找出在 Step A 中没有被选中的那些正样本
                                remaining_positives = positive_indices[max_positive_samples_per_prompt:]
                                # 用它们来补足最后的空缺
                                final_samples_for_prompt.extend(remaining_positives[:num_still_needed])
                            
                            final_kept_indices.extend(final_samples_for_prompt)

                        batch = final_data[final_kept_indices]
                        def restore_interleave_structure(batch, n_samples_per_prompt=8):
                            # 按prompt重新分组
                            prompt_groups = defaultdict(list)
                            for i, uid in enumerate(batch.non_tensor_batch["uid"]):
                                prompt_groups[uid].append(i)
                            
                            # 重新按interleave方式排列
                            interleaved_indices = []
                            for sample_idx in range(n_samples_per_prompt):
                                for uid in sorted(prompt_groups.keys()):
                                    if sample_idx < len(prompt_groups[uid]):
                                        interleaved_indices.append(prompt_groups[uid][sample_idx])
                            
                            return batch[interleaved_indices]

                        #batch = restore_interleave_structure(batch)

                        ### make data contiguous, important for a faster training 
                        for key, tensor in batch.batch.items():
                            if isinstance(tensor, torch.Tensor):
                                batch.batch[key] = tensor.contiguous()
                        
                        # 计算并记录第一次iteration的平均reward作为critic/real_score
                        # 这个指标反映了模型在未经过任何采样过滤前的真实表现水平
                        if first_iteration_rewards:
                            critic_real_score = np.mean(first_iteration_rewards)
                            metrics["critic/real_score"] = critic_real_score
                            print(f"First iteration average reward (critic/real_score): {critic_real_score:.4f}")
                            print(f"Total samples in first iteration: {len(first_iteration_rewards)}")
                        else:
                            print("Warning: No first iteration rewards recorded")
                            
                    # === Updating ===

                    # [FIXED] Simple response_mask computation with detailed analysis
                    #print(f"DEBUG: Computing response_mask - responses: {batch.batch['responses'].shape}, attention_mask: {batch.batch['attention_mask'].shape}")
                    
                    response_mask = compute_response_mask(batch)
                    '''
                    # [DEBUG] Analyze the response_mask to see if it's all ones
                    unique_values = torch.unique(response_mask)
                    num_zeros = torch.sum(response_mask == 0).item()
                    num_ones = torch.sum(response_mask == 1).item()
                    total_elements = response_mask.numel()
                    
                    print(f"DEBUG: Response mask analysis:")
                    print(f"  Shape: {response_mask.shape}")
                    print(f"  Unique values: {unique_values.tolist()}")
                    print(f"  Zeros: {num_zeros}/{total_elements} ({100*num_zeros/total_elements:.1f}%)")
                    print(f"  Ones: {num_ones}/{total_elements} ({100*num_ones/total_elements:.1f}%)")
                    
                    if num_zeros == 0:
                        print("WARNING: Response mask is all ones! This means no masking is happening.")
                        print("This could indicate padding tokens are being included in loss calculation.")
                    '''
                    batch.batch["response_mask"] = response_mask

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]

                        #### [MODIFIED]
                        
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # [FIXED] Manually set step timing since structure is complex
                total_step_time = sum(timing_raw.values())  # Sum all individual timings
                timing_raw["step"] = max(total_step_time, 0.001)  # Ensure non-zero
                
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                # [FIXED] Add safety check for timing_raw to avoid ZeroDivisionError
                if timing_raw.get("step", 0) > 0:
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                else:
                    print("WARNING: step timing is 0, skipping throughput metrics")
                    # Add default metrics to avoid missing keys
                    metrics.update({
                        "perf/total_num_tokens": sum(batch.meta_info["global_token_num"]),
                        "perf/time_per_step": 0.0,
                        "perf/throughput": 0.0,
                    })
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
