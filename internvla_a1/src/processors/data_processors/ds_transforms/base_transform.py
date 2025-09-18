import dataclasses
import torch
import numpy as np
from src.utils.transforms import Normalize, Unnormalize

@dataclasses.dataclass
class BaseTransform:
    action_dim: int = 7

    camera_keys: list[str] = dataclasses.field(default_factory=lambda: ["observation.images.image_0"])
    state_keys: list[str] = dataclasses.field(default_factory=lambda: ["observation.states"])
    action_keys: list[str] = dataclasses.field(default_factory=lambda: ["action"])
    keys_to_keep: list[str] = dataclasses.field(default_factory=lambda: ["timestamp", "index", "frame_index", "episode_index", "task_index", "task"])

    action_mode: str = "abs"

    do_normalize: bool = True
    norm_method: str = "mean_std"
    norm_stats: dict | None = None

    def __post_init__(self):
        if self.do_normalize:
            if self.norm_stats is None:
                raise ValueError("norm_stats must be provided if do_normalize is True")
            # Convert numpy arrays to torch tensors
            for key, value in self.norm_stats.items():
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        self.norm_stats[key][k] = torch.from_numpy(v).to(torch.float32)
                    elif isinstance(v, list):
                        self.norm_stats[key][k] = torch.from_numpy(np.array(v)).to(torch.float32)
                    elif isinstance(v, torch.Tensor):
                        self.norm_stats[key][k] = v
                    else:
                        raise ValueError(f"Unsupported type: {type(v)}")

            self.normalize_state = Normalize(self.norm_method, self.norm_stats, self.state_keys)
            self.normalize_action = Normalize(self.norm_method, self.norm_stats, self.action_keys)
        
        valid_camera_keys = set(self.camera_keys)
        valid_camera_keys.update(k + "_is_pad" for k in self.camera_keys)

        valid_state_keys = set(self.state_keys)

        valid_action_keys = set(self.action_keys)
        valid_action_keys.update(k + "_is_pad" for k in self.action_keys)

        self.valid_keys = valid_camera_keys | valid_state_keys | valid_action_keys | set(self.keys_to_keep)


@dataclasses.dataclass
class BaseTransformOutputs:
    action_dim: int = 7

    camera_keys: list[str] = dataclasses.field(default_factory=lambda: ["observation.images.image_0"])
    state_keys: list[str] = dataclasses.field(default_factory=lambda: ["observation.states"])
    action_keys: list[str] = dataclasses.field(default_factory=lambda: ["action"])

    do_normalize: bool = False
    norm_method: str = "mean_std"
    norm_stats: dict | None = None

    def __post_init__(self):
        if self.do_normalize:
            if self.norm_stats is None:
                raise ValueError("norm_stats must be provided if do_normalize is True")
            # Convert numpy arrays to torch tensors
            for key, value in self.norm_stats.items():
                self.norm_stats[key] = {
                    k: torch.from_numpy(v).to(torch.float32) if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            
            self.unnorm_action = Unnormalize(self.norm_method, self.norm_stats, self.action_keys)

