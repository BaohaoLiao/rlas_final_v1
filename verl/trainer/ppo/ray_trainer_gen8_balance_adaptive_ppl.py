# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
ray_trainer_gen8_balance_ppl, 正样本选perplexity最大的负样本选random
"""

import json
import os
import random
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    
    Downsampling Configuration:
    The trainer supports intelligent downsampling from N samples per prompt to a fixed number.
    Configure via trainer config:
    - positive_threshold: float = 0.9  # Threshold for positive samples (reward > threshold)
    - max_total_samples_per_prompt: int = 8  # Final number of samples per prompt
    - max_positive_samples_per_prompt: int = 4  # Max positive samples in final set
    
    Downsampling Strategy:
    1. Prioritize up to max_positive_samples_per_prompt positive samples
    2. Fill remaining slots with negative samples  
    3. If insufficient negative samples, use remaining positive samples as fallback
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # Initialize PPL tracking for adaptive sample selection
        self.N_long = getattr(config.trainer, 'ppl_long_window', 50)  # 长期均线窗口
        self.N_short = getattr(config.trainer, 'ppl_short_window', 10)  # 短期均线窗口
        self.ppl_tolerance = getattr(config.trainer, 'ppl_tolerance', 0.06)  # 容忍度
        self.ppl_warmup_steps = getattr(config.trainer, 'ppl_warmup_steps', 50)  # warmup步数，默认为N_long
        if self.ppl_warmup_steps is None:
            self.ppl_warmup_steps = self.N_long
        
        # PPL移动平均线计算
        self.alpha_long = 2 / (self.N_long + 1)
        self.alpha_short = 2 / (self.N_short + 1)
        self.ema_long = 0.0
        self.ema_short = 0.0
        self.ppl_initialized = False

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    # def _generate_multi_round_with_early_downsampling(
    #     self,
    #     orig_prompt_batch: DataProto,
    #     positive_threshold: float = 0.7,
    #     actual_repeat: int = 32,
    #     round_repeat: int = 4,
    #     final_keep_per_prompt: int = 4,
    #     timing_raw: dict | None = None,
    #     context_batch: DataProto | None = None,
    # ):
    #     """
    #     迭代式多轮生成 + 早停下采样（片段缓存，按 uid 对齐补字段；不依赖 DataProto '+'）
    #     """
    #     import time
    #     import numpy as np
    #     import torch
    #     from collections import defaultdict
    #     import math

    #     assert actual_repeat % round_repeat == 0, "actual_repeat 必须能被 round_repeat 整除"
    #     max_rounds = actual_repeat // round_repeat

    #     # -------------------- 工具函数 --------------------
    #     def _first_dim_size(dp: DataProto) -> int:
    #         if hasattr(dp, "batch") and isinstance(dp.batch, dict) and dp.batch:
    #             for v in dp.batch.values():
    #                 if isinstance(v, torch.Tensor):
    #                     return v.shape[0]
    #         if hasattr(dp, "non_tensor_batch") and isinstance(dp.non_tensor_batch, dict) and dp.non_tensor_batch:
    #             for v in dp.non_tensor_batch.values():
    #                 try:
    #                     return len(v)
    #                 except Exception:
    #                     continue
    #         raise RuntimeError("Cannot infer batch size from DataProto")

    #     def _dp_rows(dp: DataProto) -> int:
    #         return _first_dim_size(dp)

    #     def _dp_cat(frags: list[DataProto]) -> DataProto:
    #         """按行拼接若干 DataProto 片段，返回新的 DataProto。"""
    #         assert len(frags) > 0, "空片段列表"
    #         # 1) 统一 keys
    #         tensor_keys_sets = [set(f.batch.keys()) for f in frags]
    #         nontensor_keys_sets = [set(f.non_tensor_batch.keys()) for f in frags]
    #         tensor_keys = set.intersection(*tensor_keys_sets) if tensor_keys_sets else set()
    #         nontensor_keys = set.intersection(*nontensor_keys_sets) if nontensor_keys_sets else set()

    #         # 如果有不相交的键，优先取交集；需要的话也可改成并集+填充默认值
    #         if any(set(f.batch.keys()) != tensor_keys for f in frags):
    #             missing = set.union(*tensor_keys_sets) - tensor_keys
    #             print(f"[warn] tensor keys 不一致，使用交集：忽略 {missing}")
    #         if any(set(f.non_tensor_batch.keys()) != nontensor_keys for f in frags):
    #             missing = set.union(*nontensor_keys_sets) - nontensor_keys
    #             print(f"[warn] non-tensor keys 不一致，使用交集：忽略 {missing}")

    #         # 2) 拼接
    #         out_batch = {}
    #         for k in tensor_keys:
    #             parts = [f.batch[k] for f in frags]
    #             # 设备/dtype 以第一个为准
    #             out_batch[k] = torch.cat(parts, dim=0)

    #         out_non_tensor = {}
    #         for k in nontensor_keys:
    #             parts = []
    #             for f in frags:
    #                 v = f.non_tensor_batch[k]
    #                 arr = v if isinstance(v, np.ndarray) else np.array(v, dtype=object)
    #                 if arr.dtype != object:
    #                     arr = arr.astype(object)
    #                 parts.append(arr)
    #             out_non_tensor[k] = np.concatenate(parts, axis=0)

    #         # 3) 构建新的 DataProto
    #         merged: DataProto = DataProto.from_single_dict({**out_batch, **out_non_tensor})
    #         # 4) meta_info（沿用首个片段）
    #         try:
    #             merged.meta_info = dict(getattr(frags[0], "meta_info", {}) or {})
    #         except Exception:
    #             pass
    #         return merged

    #     # -------------------- context_batch: uid -> fields 映射（全量 non-tensor 键） --------------------
    #     ctx_uid_to_fields: dict = {}
    #     if context_batch is not None:
    #         if "uid" not in context_batch.non_tensor_batch:
    #             raise KeyError("context_batch 缺少 uid；无法基于 uid 做字段补齐。")
    #         ctx_uids = list(context_batch.non_tensor_batch["uid"])
    #         ctx_keys = list(context_batch.non_tensor_batch.keys())
    #         for i, u in enumerate(ctx_uids):
    #             d = ctx_uid_to_fields.setdefault(u, {})
    #             for key in ctx_keys:
    #                 d[key] = context_batch.non_tensor_batch[key][i]

    #     # ✅ orig_prompt_batch 必须有 uid（只从 context 复制，不生成随机 uid）
    #     if "uid" not in orig_prompt_batch.non_tensor_batch:
    #         if context_batch is not None and "uid" in context_batch.non_tensor_batch and \
    #         _first_dim_size(context_batch) == _first_dim_size(orig_prompt_batch):
    #             orig_prompt_batch.non_tensor_batch["uid"] = np.array(
    #                 list(context_batch.non_tensor_batch["uid"]), dtype=object
    #             )
    #         else:
    #             raise KeyError("orig_prompt_batch 缺少 uid，且无法从 context_batch 对齐复制；请确保 _get_gen_batch 透传 uid。")

    #     uid_arr = list(orig_prompt_batch.non_tensor_batch["uid"])

    #     # 状态
    #     state = {
    #         uid: {"finished": False, "seen": 0, "pos": 0, "first4_gidx": [], "later_pos_gidx": []}
    #         for uid in uid_arr
    #     }

    #     # 片段缓存 & 结果累积
    #     first4_cache: dict[str, DataProto] = {}   # uid -> 第0轮的前round_repeat条片段
    #     first4_rewards: dict[str, list] = {}      # uid -> 第0轮样本的奖励列表
    #     pos_cache = defaultdict(list)             # uid -> [单行正样本片段, ...]
    #     neg_cache = defaultdict(list)             # uid -> [单行负样本片段, ...]
    #     selected_pool_batches: list[DataProto] = []
    #     selected_count_by_uid = defaultdict(int)
    #     rounds_info = {"per_round": []}

    #     # -------------------- 轮内：对齐并计算奖励 --------------------
    #     def compute_seq_rewards_for_round(mini_prompt_batch: DataProto, gen_out: DataProto):
    #         def _repeat_tensor(t: torch.Tensor, rep: int) -> torch.Tensor:
    #             return t.repeat_interleave(rep, dim=0)

    #         Bp = _first_dim_size(mini_prompt_batch)
    #         Bg = _first_dim_size(gen_out)
    #         if Bg % Bp != 0:
    #             raise ValueError(f"Batch mismatch: gen_out({Bg}) is not a multiple of mini_prompt_batch({Bp}).")
    #         rep = Bg // Bp

    #         if not hasattr(gen_out, "non_tensor_batch") or gen_out.non_tensor_batch is None:
    #             gen_out.non_tensor_batch = {}

    #         # 1) uid
    #         if "uid" not in gen_out.non_tensor_batch:
    #             if "uid" in mini_prompt_batch.non_tensor_batch:
    #                 gen_out.non_tensor_batch["uid"] = np.repeat(
    #                     np.array(mini_prompt_batch.non_tensor_batch["uid"], dtype=object), rep, axis=0
    #                 )
    #             else:
    #                 raise KeyError("无法在 gen_out 对齐 uid；mini_prompt_batch.non_tensor_batch 里也没有 uid。")

    #         # 2) 复制 mini_prompt_batch 的所有 non-tensor 键（按 rep 展开）
    #         for k, v in mini_prompt_batch.non_tensor_batch.items():
    #             if k in gen_out.non_tensor_batch:
    #                 continue
    #             arr = np.array(v, dtype=object)
    #             if arr.shape[0] != Bp:
    #                 raise ValueError(f"mini_prompt_batch.non_tensor_batch['{k}'] 长度 {arr.shape[0]} != {Bp}")
    #             gen_out.non_tensor_batch[k] = np.repeat(arr, rep, axis=0)

    #         # 3) context(uid-join) 补齐关键字段 + 其它可补字段
    #         uids_round = list(gen_out.non_tensor_batch["uid"])
    #         required_keys = ["reward_model"]
    #         rfk = getattr(self.reward_fn, "reward_fn_key", None)
    #         if isinstance(rfk, str) and len(rfk) > 0:
    #             required_keys.append(rfk)
    #         else:
    #             required_keys.append("data_source")
    #         for key in required_keys:
    #             if key in gen_out.non_tensor_batch:
    #                 continue
    #             filled, miss = [], 0
    #             for u in uids_round:
    #                 src = ctx_uid_to_fields.get(u, None)
    #                 if src is None or key not in src:
    #                     miss += 1; filled.append(None)
    #                 else:
    #                     filled.append(src[key])
    #             if miss == len(uids_round):
    #                 raise KeyError(f"关键字段 '{key}' 在 mini_prompt_batch 和 context_batch 中都拿不到。")
    #             if any(x is None for x in filled):
    #                 ids = [i for i, x in enumerate(filled) if x is None][:5]
    #                 raise KeyError(f"'{key}' 通过 uid 映射仍有缺失（样例索引: {ids}）。请确保 context_batch 覆盖所有活跃 uid。")
    #             gen_out.non_tensor_batch[key] = np.array(filled, dtype=object)

    #         if ctx_uid_to_fields:
    #             sample_any = next(iter(ctx_uid_to_fields.values()), {})
    #             ctx_all_keys = set(sample_any.keys()) if isinstance(sample_any, dict) else set()
    #             aux_keys = [k for k in ctx_all_keys if k not in gen_out.non_tensor_batch]
    #             for key in aux_keys:
    #                 try:
    #                     filled = [ctx_uid_to_fields.get(u, {}).get(key, None) for u in uids_round]
    #                     if all(v is None for v in filled):
    #                         continue
    #                     gen_out.non_tensor_batch[key] = np.array(filled, dtype=object)
    #                 except Exception:
    #                     pass

    #         # 4) 如需补张量键（attention_mask 等），可在此从 mini_prompt_batch 按 rep 补齐
    #         # for k in required_prompt_tensor_keys:
    #         #     if k not in gen_out.batch and k in mini_prompt_batch.batch:
    #         #         gen_out.batch[k] = _repeat_tensor(mini_prompt_batch.batch[k], rep)

    #         # 5) meta（可选）
    #         if hasattr(mini_prompt_batch, "meta_info") and isinstance(mini_prompt_batch.meta_info, dict):
    #             if not hasattr(gen_out, "meta_info") or gen_out.meta_info is None:
    #                 gen_out.meta_info = {}
    #             if "global_steps" in mini_prompt_batch.meta_info and "global_steps" not in gen_out.meta_info:
    #                 gen_out.meta_info["global_steps"] = mini_prompt_batch.meta_info["global_steps"]

    #         # ==== 奖励 / KL ====
    #         mini = gen_out
    #         if self.use_rm and "rm_scores" not in mini.batch.keys():
    #             rm_tensor = self.rm_wg.compute_rm_score(mini)
    #             mini = mini.union(rm_tensor)

    #         if self.config.reward_model.launch_reward_fn_async:
    #             future_r = compute_reward_async.remote(data=mini, reward_fn=self.reward_fn)
    #             reward_tensor, reward_extra_infos_dict = ray.get(future_r)
    #         else:
    #             reward_tensor, reward_extra_infos_dict = compute_reward(mini, self.reward_fn)

    #         mini.batch["token_level_scores"] = reward_tensor

    #         if self.config.algorithm.use_kl_in_reward:
    #             mini, _ = apply_kl_penalty(mini, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
    #             seq_reward = mini.batch["token_level_rewards"].sum(dim=-1)
    #         else:
    #             seq_reward = reward_tensor.sum(dim=-1)
    #             mini.batch["token_level_rewards"] = reward_tensor

    #         if reward_extra_infos_dict:
    #             for k, v in reward_extra_infos_dict.items():
    #                 try:
    #                     if len(v) == _first_dim_size(mini):
    #                         mini.non_tensor_batch[k] = np.array(v, dtype=object)
    #                 except Exception:
    #                     pass

    #         return mini, seq_reward, uids_round

    #     # -------------------- 轮询 --------------------
    #     active_uids = set(uid_arr)
    #     for r in range(max_rounds):
    #         t0 = time.time()
    #         if not active_uids:
    #             rounds_info["per_round"].append({
    #                 "round": r, "active_prompts": 0, "made_positive": 0,
    #                 "finished_prompts": sum(1 for s in state.values() if s["finished"]),
    #                 "sec": 0.0,
    #             })
    #             break

    #         # 活跃子批
    #         uid_to_idx = {uid: i for i, uid in enumerate(uid_arr)}
    #         active_indices = [uid_to_idx[uid] for uid in uid_arr if uid in active_uids]
    #         mini_prompt_batch = orig_prompt_batch[active_indices]

    #         # 生成
    #         round_inp = mini_prompt_batch.repeat(repeat_times=round_repeat, interleave=True)

    #         dp_size = self.actor_rollout_wg.dp_size if hasattr(self.actor_rollout_wg, 'dp_size') else 8
    #         batch_size = len(round_inp)
    #         padding_applied = False
    #         if batch_size % dp_size != 0:
    #             # Pad the batch to make it divisible by dp_size
    #             padding_needed = dp_size - (batch_size % dp_size)
    #             print(f"Padding batch from {batch_size} to {batch_size + padding_needed} to make it divisible by {dp_size}")
    #             # Repeat the last few samples to pad
    #             indices_to_repeat = list(range(batch_size - padding_needed, batch_size))
    #             if len(indices_to_repeat) == 0:
    #                 indices_to_repeat = [batch_size - 1] * padding_needed
    #             padding_batch = round_inp[indices_to_repeat]
    #             round_inp = DataProto.concat([round_inp, padding_batch])
    #             padding_applied = True

    #         gen_out = (self.actor_rollout_wg.generate_sequences(round_inp)
    #                 if not self.async_rollout_mode
    #                 else self.async_rollout_manager.generate_sequences(round_inp))

    #         if padding_applied:
    #             gen_out = gen_out[:batch_size]
    #             # Also trim the round_inp to match
    #             round_inp = round_inp[:batch_size]

    #         # 轮内奖励
    #         mini_with_out, seq_reward, uids_round = compute_seq_rewards_for_round(mini_prompt_batch, gen_out)
    #         seq_reward_np = seq_reward.detach().cpu().numpy().tolist()

    #         # 轮内 uid -> 局部行索引
    #         per_uid_local_idx = defaultdict(list)
    #         for j, uid in enumerate(uids_round):
    #             per_uid_local_idx[uid].append(j)

    #         # 按 uid 更新状态/缓存与收敛
    #         made_positive_this_round = 0
    #         for uid in list(active_uids):
    #             locs = per_uid_local_idx.get(uid, [])
    #             if not locs:
    #                 continue
    #             st = state[uid]

    #             # r==0：缓存前round_repeat个片段和奖励
    #             if r == 0:
    #                 first4 = locs[:round_repeat]
    #                 st["first4_gidx"].extend(first4)
    #                 if first4 and uid not in first4_cache:
    #                     first4_cache[uid] = mini_with_out[first4]
    #                     # 保存第0轮样本的奖励
    #                     first4_rewards[uid] = [seq_reward_np[j] for j in first4]

    #             # 本轮判定与缓存正样本片段
    #             for j in locs:
    #                 if st["finished"]:
    #                     break  # 该 uid 已完成，本轮不再处理更多片段，避免重复追加
    #                 st["seen"] += 1
    #                 if seq_reward_np[j] > positive_threshold:
    #                     st["pos"] += 1
    #                     # 所有正样本都缓存（包括第0轮的）
    #                     pos_cache[uid].append(mini_with_out[[j]])
    #                     made_positive_this_round += 1

    #             # 达到比例后，用缓存片段收敛
    #             if not st["finished"]:
    #                 ratio = (st["pos"] / st["seen"]) if st["seen"] > 0 else 0.0
    #                 target_pos = math.ceil(ratio * final_keep_per_prompt) #1 if ratio <= 0.375 else (2 if ratio <= 0.625 else 3)
    #                 target_pos = min(target_pos, final_keep_per_prompt - 1)
    #                 # 必须至少有1个正样本才能收敛
    #                 if target_pos > 0 and len(pos_cache[uid]) >= target_pos and uid in first4_cache:
    #                     # 选正样本片段
    #                     pos_frags = pos_cache[uid][:target_pos]
    #                     # 用前round_repeat个补负样本若干行
    #                     neg_need = final_keep_per_prompt - len(pos_frags)
    #                     frags_to_cat = []
    #                     frags_to_cat.extend(pos_frags)
    #                     if neg_need > 0:
    #                         # 从第0轮样本中选择负样本
    #                         # 需要找到第0轮中的负样本索引
    #                         if r == 0:
    #                             # 当前轮是第0轮，可以直接从当前轮的奖励中判断
    #                             first_round_negative_indices = []
    #                             for local_idx in range(len(locs)):
    #                                 j = locs[local_idx]
    #                                 if j in st["first4_gidx"] and seq_reward_np[j] <= positive_threshold:
    #                                     # 这是第0轮的负样本
    #                                     first_round_negative_indices.append(local_idx)
                                
    #                             # 从负样本中选择需要的数量
    #                             selected_neg_indices = first_round_negative_indices[:neg_need]
    #                             if selected_neg_indices:
    #                                 neg_samples = [mini_with_out[[locs[idx]]] for idx in selected_neg_indices]
    #                                 frags_to_cat.extend(neg_samples)
    #                         else:
    #                             # 非第0轮，从缓存的第0轮样本中选择负样本
    #                             if uid in first4_rewards:
    #                                 rewards = first4_rewards[uid]
    #                                 negative_indices = [i for i, reward in enumerate(rewards) 
    #                                                   if reward <= positive_threshold]
    #                                 selected_neg_indices = negative_indices[:neg_need]
    #                                 if selected_neg_indices:
    #                                     frags_to_cat.append(first4_cache[uid][selected_neg_indices])
    #                                 else:
    #                                     # 负样本不足，从所有第0轮样本中选择负样本（包括正样本）
    #                                     all_first4_indices = list(range(len(rewards)))
    #                                     # 排除已经选择的正样本
    #                                     used_indices = set(range(target_pos))
    #                                     available_indices = [i for i in all_first4_indices if i not in used_indices]
    #                                     selected_indices = available_indices[:neg_need]
    #                                     if selected_indices:
    #                                         frags_to_cat.append(first4_cache[uid][selected_indices])
    #                             else:
    #                                 # 降级方案：没有奖励信息，从所有第0轮样本中选择
    #                                 n_first4 = _dp_rows(first4_cache[uid])
    #                                 all_first4_indices = list(range(n_first4))
    #                                 # 排除已经选择的正样本
    #                                 used_indices = set(range(target_pos))
    #                                 available_indices = [i for i in all_first4_indices if i not in used_indices]
    #                                 selected_indices = available_indices[:neg_need]
    #                                 if selected_indices:
    #                                     frags_to_cat.append(first4_cache[uid][selected_indices])

    #                     if frags_to_cat and not st["finished"]:
    #                         merged = _dp_cat(frags_to_cat)
    #                         selected_pool_batches.append(merged)
    #                         selected_count_by_uid[uid] = _dp_rows(merged)
    #                         st["finished"] = True
    #                     else:
    #                         if not frags_to_cat:
    #                             print(f"[warn] uid={uid} 收敛时片段为空，请检查阈值/缓存。")

    #         # 本轮完成后的活跃集合
    #         active_uids = {u for u in active_uids if not state[u]["finished"]}

    #         sec = time.time() - t0
    #         if timing_raw is not None:
    #             timing_raw[f"gen_round_{r}_sec"] = sec

    #         rounds_info["per_round"].append({
    #             "round": r,
    #             "active_prompts": len(per_uid_local_idx),
    #             "made_positive": made_positive_this_round,
    #             "finished_prompts": sum(1 for s in state.values() if s["finished"]),
    #             "reward_mean": float(np.mean(seq_reward_np)) if seq_reward_np else 0.0,
    #             "sec": round(sec, 3),
    #         })
    #         print(f"[Gen-Round {r}] active_prompts={len(per_uid_local_idx)} "
    #             f"made_positive={made_positive_this_round} "
    #             f"finished={rounds_info['per_round'][-1]['finished_prompts']} "
    #             f"time={sec:.3f}s "
    #             f"reward_mean={rounds_info['per_round'][-1]['reward_mean']:.4f}")

    #         if not active_uids:
    #             break

    #     # 兜底：剩余 uid 用前round_repeat个片段（若存在）
    #     uids_that_need_fallback = {uid for uid in uid_arr if selected_count_by_uid.get(uid, 0) == 0}
    #     for uid in uids_that_need_fallback:
    #         if uid in first4_cache and first4_cache[uid] is not None:
    #             n_rows = _dp_rows(first4_cache[uid])
    #             take = min(final_keep_per_prompt, n_rows)
    #             frag = first4_cache[uid][:take] if take < n_rows else first4_cache[uid]
    #             selected_pool_batches.append(frag)
    #             selected_count_by_uid[uid] = take
    #         else:
    #             print(f"[warn] uid={uid} 没有 first4_cache，无法兜底。")

    #     assert len(selected_pool_batches) > 0, "早停后没有选中样本，请检查阈值/规则是否过严或数据是否异常"

    #     # 选中样本拼接
    #     selected_batch = _dp_cat(selected_pool_batches)
    #     # === 构造与 selected_batch 行数一致的“对齐 context 行视图”（仅用于取字段，不做 union）===
    #     def _align_ctx_rows_to_selected(selected: DataProto, ctx: DataProto) -> DataProto:
    #         import numpy as np
    #         # 取 selected 的 uid 序列
    #         if "uid" not in selected.non_tensor_batch:
    #             raise KeyError("selected_batch 缺少 uid，无法对齐 context。")
    #         sel_uids = list(selected.non_tensor_batch["uid"])

    #         # ctx 必须有 uid
    #         if "uid" not in ctx.non_tensor_batch:
    #             raise KeyError("context_batch 缺少 uid，无法对齐。")
    #         ctx_uids = list(ctx.non_tensor_batch["uid"])

    #         # 建立 uid -> 首次出现的行索引
    #         uid_to_idx = {}
    #         for i, u in enumerate(ctx_uids):
    #             if u not in uid_to_idx:
    #                 uid_to_idx[u] = i

    #         # 依顺序对齐到 selected 的行
    #         idxs = []
    #         miss = []
    #         for i, u in enumerate(sel_uids):
    #             j = uid_to_idx.get(u, None)
    #             if j is None:
    #                 miss.append((i, u))
    #             else:
    #                 idxs.append(j)
    #         if miss:
    #             examples = miss[:5]
    #             raise KeyError(f"context_batch 中找不到部分 uid，样例: {examples}")

    #         return ctx[idxs]

    #     # 选择 context 来源
    #     _context_src = context_batch if context_batch is not None else orig_prompt_batch
    #     ctx_rows = _align_ctx_rows_to_selected(selected_batch, _context_src)

    #     # === 把 ctx_rows 的缺失键直接 merge 进 selected_batch（不覆盖已有键）===
    #     # 1) 非张量键
    #     for k, v in ctx_rows.non_tensor_batch.items():
    #         if k not in selected_batch.non_tensor_batch:
    #             selected_batch.non_tensor_batch[k] = v

    #     # 2) 张量键（很少需要；仅当 selected_batch 里没有该张量时才补）
    #     # 2) 张量键（仅当 selected_batch 没有该张量时才补；显式判空，避免 TensorDict 布尔判断）
    #     ctx_batch = getattr(ctx_rows, "batch", None)
    #     if ctx_batch is not None:
    #         n_selected = _first_dim_size(selected_batch)
    #         # 兼容 dict / TensorDict：两者都有 .items()
    #         for k, v in ctx_batch.items():
    #             if k in selected_batch.batch:
    #                 continue
    #             if v.shape[0] != n_selected:
    #                 raise ValueError(
    #                     f"ctx_rows.batch['{k}'] 行数({v.shape[0]}) != selected_batch({n_selected})"
    #                 )
    #             selected_batch.batch[k] = v


    #     # 最终 batch 就是已经补齐上下文字段的 selected_batch
    #     final_batch = selected_batch


    #     # 🔒 兜底：确保最终 batch 一定带有 token_level_scores
    #     if "token_level_scores" not in final_batch.batch and "token_level_rewards" in final_batch.batch:
    #         final_batch.batch["token_level_scores"] = final_batch.batch["token_level_rewards"]

    #     return final_batch, rounds_info


    def _generate_multi_round_with_early_downsampling(
        self,
        orig_prompt_batch: DataProto,
        positive_threshold: float = 0.7,
        actual_repeat: int = 32,
        round_repeat: int = 4,
        final_keep_per_prompt: int = 4,
        timing_raw: dict | None = None,
        context_batch: DataProto | None = None,
    ):
        """
        迭代式多轮生成 + 早停下采样（片段缓存，按 uid 对齐补字段；不依赖 DataProto '+'）
        """
        import time
        import numpy as np
        import torch
        from collections import defaultdict
        import math
        from datasets import Dataset

        assert actual_repeat % round_repeat == 0, "actual_repeat 必须能被 round_repeat 整除"
        max_rounds = actual_repeat // round_repeat
        target_pos = final_keep_per_prompt // 2
        target_neg = final_keep_per_prompt - target_pos

        def _first_dim_size(dp: DataProto) -> int:
            if hasattr(dp, "batch") and isinstance(dp.batch, dict) and dp.batch:
                for v in dp.batch.values():
                    if isinstance(v, torch.Tensor):
                        return v.shape[0]
            if hasattr(dp, "non_tensor_batch") and isinstance(dp.non_tensor_batch, dict) and dp.non_tensor_batch:
                for v in dp.non_tensor_batch.values():
                    try:
                        return len(v)
                    except Exception:
                        continue
            raise RuntimeError("Cannot infer batch size from DataProto")

        def _dp_rows(dp: DataProto) -> int:
            return _first_dim_size(dp)

        def _dp_cat(frags: list[DataProto]) -> DataProto:
            """使用 DataProto.concat() 保持高效的 TensorDict 结构"""
            assert len(frags) > 0, "空片段列表"
            
            # 检查键一致性（保持原有的警告机制）
            tensor_keys_sets = [set(f.batch.keys()) for f in frags]
            nontensor_keys_sets = [set(f.non_tensor_batch.keys()) for f in frags]
            tensor_keys = set.intersection(*tensor_keys_sets) if tensor_keys_sets else set()
            nontensor_keys = set.intersection(*nontensor_keys_sets) if nontensor_keys_sets else set()
            
            if any(set(f.batch.keys()) != tensor_keys for f in frags):
                missing = set.union(*tensor_keys_sets) - tensor_keys
                print(f"[warn] tensor keys 不一致，使用交集：忽略 {missing}")
            if any(set(f.non_tensor_batch.keys()) != nontensor_keys for f in frags):
                missing = set.union(*nontensor_keys_sets) - nontensor_keys
                print(f"[warn] non-tensor keys 不一致，使用交集：忽略 {missing}")
            
            # 使用 DataProto.concat() 保持 TensorDict 优化
            try:
                merged = DataProto.concat(frags)
                return merged
            except Exception as e:
                # 如果 concat 失败，回退到原来的方法
                print(f"[warn] DataProto.concat() 失败，回退到手动拼接: {e}")
                out_batch = {k: torch.cat([f.batch[k] for f in frags], dim=0) for k in tensor_keys}
                out_non_tensor = {}
                for k in nontensor_keys:
                    parts = [np.array(f.non_tensor_batch[k], dtype=object) for f in frags]
                    out_non_tensor[k] = np.concatenate(parts, axis=0)
                merged = DataProto.from_single_dict({**out_batch, **out_non_tensor})
                try:
                    merged.meta_info = dict(getattr(frags[0], "meta_info", {}) or {})
                except Exception:
                    pass
                return merged

        ctx_uid_to_fields: dict = {}
        if context_batch is not None:
            if "uid" not in context_batch.non_tensor_batch:
                raise KeyError("context_batch 缺少 uid；无法基于 uid 做字段补齐。")
            ctx_uids = list(context_batch.non_tensor_batch["uid"])
            ctx_keys = list(context_batch.non_tensor_batch.keys())
            for i, u in enumerate(ctx_uids):
                d = ctx_uid_to_fields.setdefault(u, {})
                for key in ctx_keys:
                    d[key] = context_batch.non_tensor_batch[key][i]

        if "uid" not in orig_prompt_batch.non_tensor_batch:
            if context_batch is not None and "uid" in context_batch.non_tensor_batch and _first_dim_size(context_batch) == _first_dim_size(orig_prompt_batch):
                orig_prompt_batch.non_tensor_batch["uid"] = np.array(list(context_batch.non_tensor_batch["uid"]), dtype=object)
            else:
                raise KeyError("orig_prompt_batch 缺少 uid，且无法从 context_batch 对齐复制；请确保 _get_gen_batch 透传 uid。")

        uid_arr = list(orig_prompt_batch.non_tensor_batch["uid"])

        state = {uid: {"finished": False, "seen": 0, "pos": 0, 'neg': 0} for uid in uid_arr}
        #first8_cache: dict[str, DataProto] = {}   # uid -> 第0轮的前round_repeat条片段
        all_candidates_cache = []
        selected_pool_batches: list[DataProto] = []
        rounds_info = {"per_round": []}

        def compute_seq_rewards_for_round(mini_prompt_batch: DataProto, gen_out: DataProto):
            Bp = _first_dim_size(mini_prompt_batch)
            Bg = _first_dim_size(gen_out)
            if Bg % Bp != 0: raise ValueError(f"Batch mismatch: gen_out({Bg}) is not a multiple of mini_prompt_batch({Bp}).")
            rep = Bg // Bp
            if not hasattr(gen_out, "non_tensor_batch") or gen_out.non_tensor_batch is None: gen_out.non_tensor_batch = {}
            if "uid" not in gen_out.non_tensor_batch:
                if "uid" in mini_prompt_batch.non_tensor_batch: gen_out.non_tensor_batch["uid"] = np.repeat(np.array(mini_prompt_batch.non_tensor_batch["uid"], dtype=object), rep, axis=0)
                else: raise KeyError("无法在 gen_out 对齐 uid；mini_prompt_batch.non_tensor_batch 里也没有 uid。")
            for k, v in mini_prompt_batch.non_tensor_batch.items():
                if k in gen_out.non_tensor_batch: continue
                arr = np.array(v, dtype=object)
                if arr.shape[0] != Bp: raise ValueError(f"mini_prompt_batch.non_tensor_batch['{k}'] 长度 {arr.shape[0]} != {Bp}")
                gen_out.non_tensor_batch[k] = np.repeat(arr, rep, axis=0)
            uids_round = list(gen_out.non_tensor_batch["uid"])
            required_keys = ["reward_model"]
            rfk = getattr(self.reward_fn, "reward_fn_key", None)
            if isinstance(rfk, str) and len(rfk) > 0: required_keys.append(rfk)
            else: required_keys.append("data_source")
            for key in required_keys:
                if key in gen_out.non_tensor_batch: continue
                filled, miss = [], 0
                for u in uids_round:
                    src = ctx_uid_to_fields.get(u, None)
                    if src is None or key not in src: miss += 1; filled.append(None)
                    else: filled.append(src[key])
                if miss == len(uids_round): raise KeyError(f"关键字段 '{key}' 在 mini_prompt_batch 和 context_batch 中都拿不到。")
                if any(x is None for x in filled):
                    ids = [i for i, x in enumerate(filled) if x is None][:5]
                    raise KeyError(f"'{key}' 通过 uid 映射仍有缺失（样例索引: {ids}）。请确保 context_batch 覆盖所有活跃 uid。")
                gen_out.non_tensor_batch[key] = np.array(filled, dtype=object)
            if ctx_uid_to_fields:
                sample_any = next(iter(ctx_uid_to_fields.values()), {})
                ctx_all_keys = set(sample_any.keys()) if isinstance(sample_any, dict) else set()
                aux_keys = [k for k in ctx_all_keys if k not in gen_out.non_tensor_batch]
                for key in aux_keys:
                    try:
                        filled = [ctx_uid_to_fields.get(u, {}).get(key, None) for u in uids_round]
                        if all(v is None for v in filled): continue
                        gen_out.non_tensor_batch[key] = np.array(filled, dtype=object)
                    except Exception: pass
            if hasattr(mini_prompt_batch, "meta_info") and isinstance(mini_prompt_batch.meta_info, dict):
                if not hasattr(gen_out, "meta_info") or gen_out.meta_info is None: gen_out.meta_info = {}
                if "global_steps" in mini_prompt_batch.meta_info and "global_steps" not in gen_out.meta_info: gen_out.meta_info["global_steps"] = mini_prompt_batch.meta_info["global_steps"]
            mini = gen_out
            if self.use_rm and "rm_scores" not in mini.batch.keys(): mini = mini.union(self.rm_wg.compute_rm_score(mini))
            if self.config.reward_model.launch_reward_fn_async:
                reward_tensor, reward_extra_infos_dict = ray.get(compute_reward_async.remote(data=mini, reward_fn=self.reward_fn))
            else: reward_tensor, reward_extra_infos_dict = compute_reward(mini, self.reward_fn)
            mini.batch["token_level_scores"] = reward_tensor
            if self.config.algorithm.use_kl_in_reward:
                mini, _ = apply_kl_penalty(mini, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                seq_reward = mini.batch["token_level_rewards"].sum(dim=-1)
            else:
                seq_reward = reward_tensor.sum(dim=-1)
                mini.batch["token_level_rewards"] = reward_tensor
            if reward_extra_infos_dict:
                for k, v in reward_extra_infos_dict.items():
                    try:
                        if len(v) == _first_dim_size(mini): mini.non_tensor_batch[k] = np.array(v, dtype=object)
                    except Exception: pass
            return mini, seq_reward, uids_round

        # --- 轮询 ---
        active_uids = set(uid_arr)
        for r in range(max_rounds):
            t0 = time.time()
            if not active_uids:
                rounds_info["per_round"].append({"round": r, "active_prompts": 0, "completed": 0, "finished_prompts": sum(1 for s in state.values() if s["finished"]), "sec": 0.0})
                break

            uid_to_idx = {uid: i for i, uid in enumerate(uid_arr)}
            active_indices = [uid_to_idx[uid] for uid in uid_arr if uid in active_uids]
            mini_prompt_batch = orig_prompt_batch[active_indices]
            round_inp = mini_prompt_batch.repeat(repeat_times=round_repeat, interleave=True)
            
            dp_size = self.actor_rollout_wg.dp_size if hasattr(self.actor_rollout_wg, 'dp_size') else 8
            batch_size = len(round_inp)
            padding_applied = False
            if batch_size % dp_size != 0:
                padding_needed = dp_size - (batch_size % dp_size)
                print(f"Padding batch from {batch_size} to {batch_size + padding_needed} to make it divisible by {dp_size}")
                indices_to_repeat = list(range(batch_size - padding_needed, batch_size))
                if len(indices_to_repeat) == 0: indices_to_repeat = [batch_size - 1] * padding_needed
                padding_batch = round_inp[indices_to_repeat]
                round_inp = DataProto.concat([round_inp, padding_batch])
                padding_applied = True
            
            gen_out = (self.actor_rollout_wg.generate_sequences(round_inp) if not self.async_rollout_mode else self.async_rollout_manager.generate_sequences(round_inp))
            
            if padding_applied:
                gen_out = gen_out[:batch_size]
                round_inp = round_inp[:batch_size]

            mini_with_out, seq_reward, uids_round = compute_seq_rewards_for_round(mini_prompt_batch, gen_out)
            seq_reward_np = seq_reward.detach().cpu().numpy().tolist()
            per_uid_local_idx = defaultdict(list)
            for j, uid in enumerate(uids_round):
                per_uid_local_idx[uid].append(j)

            completed_this_round = 0
            for uid in list(active_uids):
                locs = per_uid_local_idx.get(uid, [])
                if not locs: continue
                st = state[uid]

                # r==0：缓存前round_repeat个片段
                # if r == 0:
                #     first8 = locs[:round_repeat]
                #     if first8 and uid not in first8_cache:
                #         first8_cache[uid] = mini_with_out[first8]

                for j in locs:
                    if st["finished"]: break
                    st["seen"] += 1
                    is_positive = seq_reward_np[j] > positive_threshold
                    if is_positive:
                        st["pos"] += 1
                    else:
                        st["neg"] += 1
                    all_candidates_cache.append(mini_with_out[[j]])

                if not st["finished"]:
                    
                    if st["pos"] >= target_pos and st["neg"] >= target_neg:
                        st["finished"] = True
                        completed_this_round += 1

            active_uids = {u for u in active_uids if not state[u]["finished"]}
            sec = time.time() - t0
            if timing_raw is not None: timing_raw[f"gen_round_{r}_sec"] = sec
            rounds_info["per_round"].append({"round": r, "active_prompts": len(per_uid_local_idx), "completed": completed_this_round, "finished_prompts": sum(1 for s in state.values() if s["finished"]), "reward_mean": float(np.mean(seq_reward_np)) if seq_reward_np else 0.0, "sec": round(sec, 3)})
            print(f"[Gen-Round {r}] active_prompts={len(per_uid_local_idx)} completed={completed_this_round} finished={rounds_info['per_round'][-1]['finished_prompts']} time={sec:.3f}s reward_mean={rounds_info['per_round'][-1]['reward_mean']:.4f}")
            if not active_uids: break

        all_candidates_cache = _dp_cat(all_candidates_cache)
        # 确保有response_mask字段
        if "response_mask" not in all_candidates_cache.batch:
            all_candidates_cache.batch["response_mask"] = compute_response_mask(all_candidates_cache)
        # 计算所有all_candidates_cache的old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(all_candidates_cache)
            entropys = old_log_prob.batch["entropys"]
            response_masks = all_candidates_cache.batch["response_mask"]
            # loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            # entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_probs = old_log_prob.batch["old_log_probs"]
            masked_log_probs = old_log_probs * response_masks
            sample_nlls = -torch.sum(masked_log_probs, dim=1)  # (batch_size,)
            sample_valid_tokens = torch.sum(response_masks, dim=1)  # (batch_size,)
            sample_perplexity = sample_nlls / sample_valid_tokens  # (batch_size,)
            # Handle division by zero
            sample_perplexity = torch.where(sample_valid_tokens > 0, sample_perplexity, torch.zeros_like(sample_perplexity))
            old_log_prob.batch.pop("entropys")
            all_candidates_cache = all_candidates_cache.union(old_log_prob)
            all_candidates_cache.batch["entropy"] = entropys
            all_candidates_cache.batch["perplexity"] = sample_perplexity

        # 计算当前步骤的平均PPL并更新自适应策略
        current_step_ppl = all_candidates_cache.batch["perplexity"].mean().item()
        selection_mode = self._update_ppl_tracking(current_step_ppl)
        
        # 详细的模式选择日志
        print(f"\n🎯 [Step {self.global_steps}] PPL自适应策略分析")
        print(f"   当前步骤PPL: {current_step_ppl:.4f}")
        print(f"   EMA短期: {self.ema_short:.4f}")
        print(f"   EMA长期: {self.ema_long:.4f}")
        
        if self.ppl_initialized and self.global_steps >= self.ppl_warmup_steps:
            consolidation_threshold = self.ema_long * (1 + self.ppl_tolerance)
            exploration_threshold = self.ema_long * (1 - self.ppl_tolerance)
            print(f"   阈值范围: [{exploration_threshold:.4f}, {consolidation_threshold:.4f}]")
            print(f"   容忍度: ±{self.ppl_tolerance*100:.1f}%")
            
            # 详细的模式判断原因
            if self.ema_short > consolidation_threshold:
                diff_pct = (self.ema_short - self.ema_long) / self.ema_long * 100
                print(f"   🔴 模式: {selection_mode} - 短期PPL比长期高{diff_pct:.1f}%，模型可能不稳定，选择低PPL样本")
            elif self.ema_short < exploration_threshold:
                diff_pct = (self.ema_long - self.ema_short) / self.ema_long * 100
                print(f"   🟡 模式: {selection_mode} - 短期PPL比长期低{diff_pct:.1f}%，PPL下降过快，选择高PPL样本保持多样性")
            else:
                ratio = self.ema_short / self.ema_long
                print(f"   🟢 模式: {selection_mode} - 短期/长期比率{ratio:.3f}，趋势健康，平衡选择高低PPL样本")
        else:
            warmup_remaining = self.ppl_warmup_steps - self.global_steps
            print(f"   🔵 模式: {selection_mode} (warmup阶段，剩余{warmup_remaining}步)")
            print(f"   说明: warmup期间使用均衡模式，让EMA有足够数据建立基线")

        # 按照uid分组
        uid_to_traj_indices = defaultdict(list)
        final_kept_indices = []
        for i, uid in enumerate(all_candidates_cache.non_tensor_batch["uid"]):
            uid_to_traj_indices[uid].append(i)

        for uid in uid_arr:
            indices = uid_to_traj_indices.get(uid, [])
            if not indices: continue
            pos_indices_perplexity = [(i, all_candidates_cache.batch["perplexity"][i]) for i in indices if (all_candidates_cache.batch["token_level_rewards"][i]).sum(dim=-1) > positive_threshold]
            neg_indices_perplexity = [(i, all_candidates_cache.batch["perplexity"][i]) for i in indices if (all_candidates_cache.batch["token_level_rewards"][i]).sum(dim=-1) <= positive_threshold]
            pos_num = len(pos_indices_perplexity)
            neg_num = len(neg_indices_perplexity)
            n_rows = pos_num + neg_num
            take = min(final_keep_per_prompt, n_rows)
            if n_rows < final_keep_per_prompt:
                print(f"[WARN] uid={uid} 样本{n_rows}不足目标{final_keep_per_prompt}，但继续处理")
            
            actual_pos = min(pos_num, target_pos)
            actual_neg = min(neg_num, target_neg)
            # 如果一种样本不足，用另一种补齐
            if actual_pos + actual_neg < take:
                if pos_num > actual_pos:
                    # 用正样本补齐
                    additional_pos = min(pos_num - actual_pos, take - actual_pos - actual_neg)
                    actual_pos += additional_pos
                elif neg_num > actual_neg:
                    # 用负样本补齐
                    additional_neg = min(neg_num - actual_neg, take - actual_pos - actual_neg)
                    actual_neg += additional_neg

            # 使用自适应样本选择策略
            selected_pos_indices, selected_neg_indices = self._select_samples_by_mode(
                pos_indices_perplexity, neg_indices_perplexity, 
                actual_pos, actual_neg, selection_mode
            )
            
            # 合并选中的样本索引
            selected_indices = selected_pos_indices + selected_neg_indices
            final_kept_indices.extend(selected_indices)

        # 使用下采样后的索引重新构建batch
        if final_kept_indices and len(final_kept_indices) > 0:
            # 确保final_kept_indices是列表格式
            if not isinstance(final_kept_indices, list):
                final_kept_indices = list(final_kept_indices)
            selected_batch = all_candidates_cache[final_kept_indices]
        else:
            # 如果没有选中任何样本，使用所有样本
            print(f"[WARN] No samples selected, using all {len(all_candidates_cache)} samples")
            selected_batch = all_candidates_cache

        
        def _align_ctx_rows_to_selected(selected: DataProto, ctx: DataProto) -> DataProto:
            import numpy as np
            # 取 selected 的 uid 序列
            if "uid" not in selected.non_tensor_batch:
                raise KeyError("selected_batch 缺少 uid，无法对齐 context。")
            sel_uids = list(selected.non_tensor_batch["uid"])

            # ctx 必须有 uid
            if "uid" not in ctx.non_tensor_batch:
                raise KeyError("context_batch 缺少 uid，无法对齐。")
            ctx_uids = list(ctx.non_tensor_batch["uid"])

            # 建立 uid -> 首次出现的行索引
            uid_to_idx = {}
            for i, u in enumerate(ctx_uids):
                if u not in uid_to_idx:
                    uid_to_idx[u] = i

            # 依顺序对齐到 selected 的行
            idxs = []
            miss = []
            for i, u in enumerate(sel_uids):
                j = uid_to_idx.get(u, None)
                if j is None:
                    miss.append((i, u))
                else:
                    idxs.append(j)
            if miss:
                examples = miss[:5]
                raise KeyError(f"context_batch 中找不到部分 uid，样例: {examples}")

            return ctx[idxs]

        _context_src = context_batch if context_batch is not None else orig_prompt_batch
        ctx_rows = _align_ctx_rows_to_selected(selected_batch, _context_src)

        for k, v in ctx_rows.non_tensor_batch.items():
            if k not in selected_batch.non_tensor_batch: selected_batch.non_tensor_batch[k] = v
        ctx_batch = getattr(ctx_rows, "batch", None)
        if ctx_batch is not None:
            n_selected = _first_dim_size(selected_batch)
            for k, v in ctx_batch.items():
                if k in selected_batch.batch: continue
                if v.shape[0] != n_selected: raise ValueError(f"ctx_rows.batch['{k}'] 行数({v.shape[0]}) != selected_batch({n_selected})")
                selected_batch.batch[k] = v
        final_batch = selected_batch
        if "token_level_scores" not in final_batch.batch and "token_level_rewards" in final_batch.batch:
            final_batch.batch["token_level_scores"] = final_batch.batch["token_level_rewards"]
        
        # 验证 final_batch 保持了高效的 TensorDict 结构
        if hasattr(final_batch.batch, '__class__'):
            batch_type = final_batch.batch.__class__.__name__
            if 'TensorDict' not in batch_type and 'dict' in batch_type.lower():
                print(f"[perf_warn] final_batch.batch 是普通 {batch_type}，可能影响性能")
            else:
                print(f"[perf_info] final_batch.batch 是高效的 {batch_type}")

        return final_batch, rounds_info, current_step_ppl, selection_mode

    def _update_ppl_tracking(self, current_ppl: float) -> str:
        """
        更新PPL移动平均线并返回当前的样本选择模式
        
        Args:
            current_ppl: 当前步骤的平均PPL
            
        Returns:
            selection_mode: "Consolidation", "Exploration", 或 "Balanced"
        """
        # 初始化EMA值
        if not self.ppl_initialized:
            self.ema_long = current_ppl
            self.ema_short = current_ppl
            self.ppl_initialized = True
        else:
            # 使用EMA公式进行迭代更新
            self.ema_long = self.alpha_long * current_ppl + (1 - self.alpha_long) * self.ema_long
            self.ema_short = self.alpha_short * current_ppl + (1 - self.alpha_short) * self.ema_short
        
        # warmup阶段处理
        if self.global_steps < self.ppl_warmup_steps:
            return "Balanced"
        
        # 自适应模式判断
        consolidation_threshold = self.ema_long * (1 + self.ppl_tolerance)
        exploration_threshold = self.ema_long * (1 - self.ppl_tolerance)
        
        if self.ema_short > consolidation_threshold:
            # 短期PPL显著高于长期趋势 -> 模型不稳定，需要巩固
            return "Consolidation"
        elif self.ema_short < exploration_threshold:
            # 短期PPL显著低于长期趋势 -> PPL下降过快，可能丧失多样性，需要探索
            return "Exploration"
        else:
            # 短期和长期趋势一致 -> 模型状态健康
            return "Balanced"

    def _select_samples_by_mode(self, pos_indices_perplexity: list, neg_indices_perplexity: list, 
                               target_pos: int, target_neg: int, selection_mode: str) -> tuple:
        """
        根据选择模式选择正样本和负样本
        
        Args:
            pos_indices_perplexity: [(index, perplexity), ...] 正样本索引和perplexity对
            neg_indices_perplexity: [(index, perplexity), ...] 负样本索引和perplexity对  
            target_pos: 目标正样本数量
            target_neg: 目标负样本数量
            selection_mode: 选择模式
            
        Returns:
            (selected_pos_indices, selected_neg_indices): 选中的正样本和负样本索引
        """
        actual_pos = min(len(pos_indices_perplexity), target_pos)
        actual_neg = min(len(neg_indices_perplexity), target_neg)
        
        # 选择正样本
        if actual_pos == 0:
            selected_pos_indices = []
        elif selection_mode == "Consolidation":
            # 巩固模式：选择PPL最低的正样本
            pos_sorted = sorted(pos_indices_perplexity, key=lambda x: x[1])  # 升序排序
            selected_pos_indices = [x[0] for x in pos_sorted[:actual_pos]]
            
        elif selection_mode == "Exploration":
            # 探索模式：选择PPL最高的正样本  
            pos_sorted = sorted(pos_indices_perplexity, key=lambda x: x[1], reverse=True)  # 降序排序
            selected_pos_indices = [x[0] for x in pos_sorted[:actual_pos]]
            
        else:  # Balanced Mode
            # 均衡模式：选择1个最高PPL和1个最低PPL的正样本（如果有足够样本的话）
            if actual_pos >= 2:
                pos_sorted_asc = sorted(pos_indices_perplexity, key=lambda x: x[1])  # 升序
                pos_sorted_desc = sorted(pos_indices_perplexity, key=lambda x: x[1], reverse=True)  # 降序
                
                selected_pos_indices = []
                # 先选最低PPL的
                low_count = actual_pos // 2
                selected_pos_indices.extend([x[0] for x in pos_sorted_asc[:low_count]])
                # 再选最高PPL的
                high_count = actual_pos - low_count
                selected_pos_indices.extend([x[0] for x in pos_sorted_desc[:high_count]])
            else:
                # 样本不足时随机选择
                if actual_pos > 0:
                    selected_pos_indices = [x[0] for x in random.sample(pos_indices_perplexity, actual_pos)]
                else:
                    selected_pos_indices = []
        
        # 负样本始终随机选择（保持原有逻辑）
        neg_indices_only = [x[0] for x in neg_indices_perplexity]
        if len(neg_indices_only) >= actual_neg:
            selected_neg_indices = random.sample(neg_indices_only, actual_neg)
        else:
            selected_neg_indices = neg_indices_only
            
        return selected_pos_indices, selected_neg_indices

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

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if False:
        #if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
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
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # 可调参数
                    positive_threshold = 0.7
                    actual_repeat = 32
                    round_repeat = 8           # 每轮为活跃 prompt 生成8条
                    final_keep_per_prompt = 4  # 每个 prompt 最终保留4条

                    with marked_timer("gen_multi_round", timing_raw, color="red"):
                        # 函数已返回对齐合并后的"最终 batch"
                        final_batch, rounds_info, current_step_ppl, selection_mode = self._generate_multi_round_with_early_downsampling(
                            orig_prompt_batch=gen_batch,
                            positive_threshold=positive_threshold,
                            actual_repeat=actual_repeat,
                            round_repeat=round_repeat,
                            final_keep_per_prompt=final_keep_per_prompt,
                            timing_raw=timing_raw,
                            context_batch=batch,  # 用于补齐非张量字段（uid 映射等）
                        )

                    # 小结打印（也可写到 metrics）
                    total_prompts = len(set(gen_batch.non_tensor_batch["uid"]))
                    print(f"[Summary] prompts={total_prompts}, selected_rows={len(final_batch)}, "
                        f"max_rounds={actual_repeat // round_repeat}")
                    if rounds_info.get("per_round"):
                        try:
                            finished_prompts = rounds_info["per_round"][-1]["finished_prompts"]
                        except Exception:
                            finished_prompts = 0
                        for info in rounds_info["per_round"]:
                            print(f"  - round {info['round']}: active={info['active_prompts']}, "
                                f"made_pos={info['completed']}, finished={info['finished_prompts']}, "
                                f"time={info['sec']}s")
                    # write into metrics
                    metrics["sampling/total_samples"] = np.sum([(info["active_prompts"] * round_repeat) for info in rounds_info["per_round"]])
                    metrics["sampling/prompts_active_only_1st_round"] = rounds_info["per_round"][0]["finished_prompts"]
                    
                    # 安全地访问第二轮信息
                    if len(rounds_info["per_round"]) > 1:
                        metrics["sampling/prompts_active_after_1st_round"] = rounds_info["per_round"][1]["active_prompts"] - (rounds_info["per_round"][0]["active_prompts"] - rounds_info["per_round"][-1]["finished_prompts"])
                    else:
                        metrics["sampling/prompts_active_after_1st_round"] = 0
                    
                    metrics["sampling/prompts_no_positive_anywhere"] = rounds_info["per_round"][0]["active_prompts"] - rounds_info["per_round"][-1]["finished_prompts"]
                    metrics['sampling/kept_samples'] = len(final_batch)
                    
                    # 添加PPL相关的metrics
                    metrics["ppl/current_step"] = current_step_ppl
                    metrics["ppl/ema_short"] = self.ema_short
                    metrics["ppl/ema_long"] = self.ema_long
                    metrics["ppl/selection_mode"] = {"Consolidation": 0, "Exploration": 1, "Balanced": 2}[selection_mode]
                    if self.ppl_initialized:
                        metrics["ppl/short_vs_long_ratio"] = self.ema_short / self.ema_long if self.ema_long > 0 else 1.0
                    
                    # ✅ 关键：不要再 repeat / union 了，直接用最终 batch
                    batch = final_batch
                    
                    # 最终训练样本统计 (使用final_batch中的token_level_rewards)
                    if "token_level_rewards" in batch.batch:
                        final_pos_count = sum(1 for i in range(len(batch)) if batch.batch["token_level_rewards"][i].sum() > positive_threshold)
                        final_neg_count = len(batch) - final_pos_count
                        print(f"\n📈 [Step {self.global_steps}] 最终训练样本统计:")
                        print(f"   最终样本分布: {final_pos_count}正 + {final_neg_count}负 = {len(batch)}总")
                    else:
                        print(f"\n📈 [Step {self.global_steps}] 最终训练样本统计:")
                        print(f"   最终样本数量: {len(batch)}总 (无奖励信息)")
                    print(f"   PPL自适应策略: {selection_mode}模式")
                    if self.ppl_initialized:
                        print(f"   EMA趋势: 短期({self.ema_short:.4f}) vs 长期({self.ema_long:.4f})")
                    print("=" * 80)

                    # 之后保持不变（mask/balance/kl/adv/损失等）...
                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        # Check if batch size is divisible by world_size for balancing
                        world_size = self.actor_rollout_wg.world_size
                        batch_size = len(batch)
                        if batch_size % world_size == 0:
                            self._balance_batch(batch, metrics=metrics)
                        else:
                            # Pad the batch to make it divisible by world_size
                            padding_needed = world_size - (batch_size % world_size)
                            print(f"Padding batch from {batch_size} to {batch_size + padding_needed} for balancing")
                            
                            # Randomly choose samples to pad
                            indices_to_repeat = random.choices(range(batch_size), k=padding_needed)
                            padding_batch = batch[indices_to_repeat]
                            batch = DataProto.concat([batch, padding_batch])
                            
                            # 验证padding后batch结构保持高效
                            if hasattr(batch.batch, '__class__'):
                                batch_type = batch.batch.__class__.__name__
                                if 'TensorDict' not in batch_type and 'dict' in batch_type.lower():
                                    print(f"[perf_warn] 填充后batch.batch是普通{batch_type}，可能影响性能")
                            
                            # Now balance the padded batch
                            self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.batch = batch.batch.contiguous()

                    # (可选) 记录真实平均奖励（以最终保留样本为准）
                    all_seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
                    metrics["critic/real_reward"] = rounds_info["per_round"][0]["reward_mean"]
                    metrics["sampling/downsampled_samples"] = len(batch)           # 实际用于训练的样本数
                    metrics["sampling/total_prompts"] = total_prompts

                    # log old_log_probs
                    if 'old_log_probs' in batch.batch.keys():
                        old_log_prob = batch.batch["old_log_probs"]
                        entropys = batch.batch['entropy']
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item(),
                            "actor/perplexity": batch.batch["perplexity"].mean().detach().item()
                        }
                        metrics.update(old_log_prob_metrics)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # Note: reward processing and downsampling already done above
                        #######################
                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            reward_extra_infos_dict = {}
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict["request_id"] = batch.non_tensor_batch["request_id"].tolist()

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
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

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
