from tqdm import tqdm
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from glob import glob
import torch
import json
from pathlib import Path
from lerobot.common.datasets.compute_stats import aggregate_stats


SUFFIX = "_delta"

ACTION_KEYS = [
    "actions.joint.position",
    "actions.effector.position",
]

task_ids = [
    354,
    356,
    359,
    360,
    361,
    362,
    363,
    365,
    366,
    372,
    374,
    376,
    377,
    380,
    385,
    398,
    410,
    421,
    424,
    438,
    444,
    470,
    474,
    477,
    483,
    486,
    487,
    503,
    512,
    520,
    521,
    555,
    558,
    561,
    570,
    582,
    597,
    599,
    602,
    609,
    616,
    619,
    658,
    681,
]


def set_delta_action(ds: LeRobotDataset):
    episode_lengths = [ep_dict["length"] for ep_dict in ds.meta.episodes.values()]
    cumulative_lengths = np.cumsum(episode_lengths)

    for k in ACTION_KEYS:
        action = torch.stack(ds.hf_dataset[k]).numpy()
        delta_action = np.diff(action, axis=0)

        delta_action = np.concatenate([delta_action, delta_action[-1:]], axis=0)
        for end_idx in cumulative_lengths:
            delta_action[end_idx - 1] = delta_action[end_idx - 2]

        ds.meta.stats[k]["min"] = delta_action.min(0)
        ds.meta.stats[k]["max"] = delta_action.max(0)
        ds.meta.stats[k]["mean"] = delta_action.mean(0)
        ds.meta.stats[k]["std"] = delta_action.std(0)

        for st in ["min", "max", "mean", "std"]:
            if ds.meta.stats[k][st].ndim == 0:
                raise
                ds.meta.stats[k][st] = ds.meta.stats[k][st][None]

def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(elem) for elem in obj]
    else:
        return obj

def main():
    # base_dir = "/fs-computility/efm/shared/datasets/agibot-world/AgiBotWorld-Beta/0416_agibotworld_beta_lerobotv2_ray/agibotworld"
    # sub_ds_stats = []
    # fp_paths = [f"{base_dir}/task_{i}" for i in task_ids]
    # for i, ds_fp_path in enumerate(tqdm(fp_paths)):
    #     ds = LeRobotDataset(ds_fp_path)
    #     set_delta_action(ds)
    #     sub_ds_stats.append(ds.meta.stats)

    # ds_stats = aggregate_stats(sub_ds_stats)

    # output_file_path = Path(base_dir) / f"stats_delta.json"
    # ds_stats = convert_numpy_to_list(ds_stats)
    # with open(output_file_path, "w") as f:
    #     json.dump(ds_stats, f, indent=2)

    base_dir = "/fs-computility/efm/shared/datasets/agibot-world/AgiBotWorld-Beta/0416_agibotworld_beta_lerobotv2_ray/agibotworld"
    sub_ds_stats = []
    fp_paths = [f"{base_dir}/task_{i}" for i in task_ids]
    for i, ds_fp_path in enumerate(tqdm(fp_paths)):
        ds = LeRobotDatasetMetadata(ds_fp_path)
        sub_ds_stats.append(ds.stats)

    ds_stats = aggregate_stats(sub_ds_stats)

    output_file_path = Path(base_dir) / f"stats_ds.json"
    ds_stats = convert_numpy_to_list(ds_stats)
    with open(output_file_path, "w") as f:
        json.dump(ds_stats, f, indent=2)


if __name__ == "__main__":
    main()