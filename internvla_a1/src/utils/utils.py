import os
import shutil
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from transformers.trainer_utils import _re_checkpoint

import numpy as np
from typing import Union

import torch
from torch.utils.data import Sampler

from policies.mwm_policy import MWMPolicy


def save_training_args(training_args, policy_config, config):
    os.makedirs(training_args.output_dir, exist_ok=True)
    policy_config.save_pretrained(Path(training_args.output_dir))

    if not os.path.exists(Path(training_args.output_dir) / "config.yaml"):
        OmegaConf.save(config, Path(training_args.output_dir) / "config.yaml")


def clean_overrides(override_args):
    cleaned_args = []
    for arg in override_args:
        if arg.startswith("--"):
            cleaned_args.append(arg[2:])
        else:
            cleaned_args.append(arg)
    return cleaned_args


def load_ckpt(policy, config):
    if config.policy.use_world_model:
        if config.exp.stage == "stage1_infer_wm":
            print(f"\033[93mLoading mwm from {config.policy.mwm_pretrained_path}\033[0m")
            MWMPolicy._load_as_safetensor(policy, config.policy.mwm_pretrained_path, "cpu", False)
        elif config.exp.stage == "stage2_pretrain_vla" and config.policy.use_world_model:
            assert config.exp.load_ckpt is not None, f"load_ckpt is not set for stage {config.exp.stage}"
            if config.exp.load_ckpt is not None:
                print(f"\033[93mLoading ckpt from {config.exp.load_ckpt}\033[0m")
                MWMPolicy._load_as_safetensor(policy, config.exp.load_ckpt, "cpu", False)
        elif config.exp.stage == "stage1_pretrain_wm" and config.exp.load_ckpt is not None:
            print(f"\033[93mLoading mwm from {config.exp.load_ckpt}\033[0m")
            MWMPolicy._load_as_safetensor(policy, config.exp.load_ckpt, "cpu", False)
        elif config.exp.stage == "pretrain_onestage" and config.exp.load_ckpt is not None:
            if config.exp.load_ckpt is not None:
                print(f"\033[93mLoading ckpt from {config.exp.load_ckpt}\033[0m")
                MWMPolicy._load_as_safetensor(policy, config.exp.load_ckpt, "cpu", False)
        elif "stage3" in config.exp.stage:
            assert config.exp.load_ckpt is not None, f"load_ckpt is not set for stage {config.exp.stage}"
            print(f"\033[93mLoading ckpt from {config.exp.load_ckpt}\033[0m")
            MWMPolicy._load_as_safetensor(policy, config.exp.load_ckpt, "cpu", False)
    else:
        if config.exp.load_ckpt is not None:
            MWMPolicy._load_as_safetensor(policy, config.exp.load_ckpt, "cpu", False)
        
    return policy


def set_policy_config(policy_config, src_config):
    """
    Set the policy config from the config file
    Args:
        policy_config: The policy config to set which is used to initialize the policy
        src_config: The policy config from the local config file
    """
    policy_config.pretrained_path = src_config.path
    policy_config.language_tokenizer_path = src_config.language_tokenizer_path

    policy_config.use_world_model = src_config.use_world_model

    if policy_config.use_world_model:
        policy_config.gen_expert_config.temporal_conv_kernel_size = src_config.temporal_conv_kernel_size
        policy_config.gen_expert_config.temporal_conv_stride = src_config.temporal_conv_stride
        policy_config.gen_expert_config.spatial_conv_kernel_size = src_config.spatial_conv_kernel_size
        policy_config.gen_expert_config.spatial_conv_stride = src_config.spatial_conv_stride
        policy_config.image_tokenizer_path = src_config.image_tokenizer_path

    policy_config.resize_imgs_with_padding = eval(src_config.resize_imgs_with_padding)

    policy_config.attention_implementation = src_config.attention_implementation
    policy_config.chunk_size = src_config.chunk_size

    return policy_config


def get_second_last_checkpoint(folder):
    worker_idx = int(os.environ.get("MLP_ROLE_INDEX", 0))
    local_rank_idx = int(os.environ.get('LOCAL_RANK', -1))

    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) < 2:
        if worker_idx == 0 and local_rank_idx in [-1, 0]:
            for checkpoint in checkpoints:
                shutil.rmtree(os.path.join(folder, checkpoint))
        return None

    sorted_checkpoints = sorted(
        checkpoints,
        key=lambda x: int(_re_checkpoint.search(x).groups()[0]),
        reverse=True
    )
    # if worker_idx == 0 and local_rank_idx in [-1, 0]:
    #     shutil.rmtree(os.path.join(folder, sorted_checkpoints[0]))

    return os.path.join(folder, sorted_checkpoints[0])


class LargeScaleWeightedRandomSampler(Sampler):
    def __init__(self, weights: Union[torch.Tensor, list, np.ndarray], num_samples: int, replacement: bool = True, max_block: int = 2**24 - 1):
        if isinstance(weights, list):
            weights = torch.tensor(weights)
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.max_block = max_block

    def __iter__(self):
        return iter(self._sample_indices().tolist())

    def _sample_indices(self) -> torch.Tensor:
        weights = self.weights
        total_weight = weights.sum()
        indices = []
        n = len(weights)
        num_blocks = (n + self.max_block - 1) // self.max_block

        for i in range(num_blocks):
            start = i * self.max_block
            end = min((i + 1) * self.max_block, n)
            block_weights = weights[start:end].float()
            block_weight_sum = block_weights.sum()

            if block_weight_sum == 0:
                continue

            block_prob = block_weight_sum / total_weight
            block_sample_count = int(round(self.num_samples * block_prob.item()))
            sampled = torch.multinomial(block_weights, block_sample_count, self.replacement)
            indices.append(sampled + start)

        return torch.cat(indices)[:self.num_samples]  # truncate in case of rounding error

    def __len__(self):
        return self.num_samples


def convert_ds_stats_to_dict(ds_stats):
    for k, v in ds_stats.items():
        for _k, _v in v.items():
            if isinstance(_v, np.ndarray):
                ds_stats[k][_k] = _v.tolist()
    return ds_stats