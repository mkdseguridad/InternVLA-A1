import dataclasses
import numpy as np
from collections.abc import Sequence, Callable

import torchvision.transforms.functional as TF

from torchvision.transforms import transforms as tv_transforms

from typing import (
    Protocol,
    Dict,
    List,
    Any,
    Tuple,
)

from PIL import Image

import torch

import src.utils.tokenizer as _tokenizer
import src.utils.image_tools as image_tools

from lerobot.configs.policies import PreTrainedConfig


@dataclasses.dataclass
class TensorSpec:
    shape: Tuple[int, ...]
    dtype: torch.dtype


class DataTransformFn(Protocol):
    def __call__(self, data: Dict) -> Dict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: Dict) -> Dict:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclasses.dataclass
class RepackTransform:
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    def __init__(self, structure: Dict[str, Any]):
        self.structure = structure

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flat_item = flatten_dict(data)
        return {new_key: flat_item[old_key] for new_key, old_key in self._flatten_structure(self.structure).items()}

    def _flatten_structure(self, structure: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, str]:
        """Flatten the structure dictionary."""
        items = []
        for k, v in structure.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_structure(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


@dataclasses.dataclass
class InjectHistoryObservation:
    n_obs_img_steps: int
    n_pred_img_steps: int
    obs_img_stride: int
    suffix: str = "history"

    def __post_init__(self):
        if self.n_obs_img_steps % self.obs_img_stride != 0:
            raise ValueError(f"n_obs_img_steps {self.n_obs_img_steps} must be divisible by obs_img_stride {self.obs_img_stride}")
        if self.n_pred_img_steps % self.obs_img_stride != 0:
            raise ValueError(f"n_pred_img_steps {self.n_pred_img_steps} must be divisible by obs_img_stride {self.obs_img_stride}")

        self.cur_n_obs_img_steps = self.n_obs_img_steps // self.obs_img_stride
        self.cur_n_pred_img_steps = self.n_pred_img_steps // self.obs_img_stride
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sample = {}
        for key, value in data.items():
            if key.startswith("observation") and "image" in key and self.suffix not in key:
                sample[key] = value[-self.cur_n_pred_img_steps-1]
            else:
                sample[key] = value
        return sample


@dataclasses.dataclass
class TransformWorldModelObservation:
    gen_obs_transforms: list
    suffix: str = "history"
    def __call__(self, data: Dict) -> Dict:
        for k, v in data.items():
            if k.endswith(self.suffix):
                v = [image_tools.convert_to_uint8(x) for x in v.numpy()]
                v = [Image.fromarray(x.transpose(1, 2, 0)) for x in v]
                for t in self.gen_obs_transforms:
                    if isinstance(t, ConsistentRandomCrop):
                        v = t(v)
                    else:
                        v = [t(x) for x in v]
                data[k] = torch.stack(v) # torch.Size([3, 3, 256, 256])
        return data

# @dataclasses.dataclass
# class TransformWorldModelObservation:
#     gen_obs_transforms: list
#     suffix: str = "history"

#     def __call__(self, data: Dict) -> Dict:
#         for k, v in data.items():
#             if k.endswith(self.suffix):
#                 gen_bos = v
#                 for _trans in self.gen_obs_transforms:
#                     gen_bos =  _trans({k: gen_bos})    
#                 data[k] = gen_bos[k]
#         return data


@dataclasses.dataclass
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: Dict) -> Dict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


class ConsistentRandomCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = tuple(tv_transforms._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

    def forward(self, img_list: List[Image.Image]):
        img = img_list[0]

        i, j, h_crop, w_crop = tv_transforms.RandomCrop.get_params(img, output_size=self.size)

        cropped_frames = [TF.crop(f, i, j, h_crop, w_crop) for f in img_list]
        return cropped_frames


@dataclasses.dataclass
class Normalize(DataTransformFn):
    norm_method: str
    norm_stats: dict
    norm_keys: list[str]

    def __call__(self, data: Dict) -> Dict:
        if self.norm_stats is None:
            if self.norm_method is not None and self.norm_method != "none":
                raise ValueError(f"norm_method is {self.norm_method} but norm_stats is None")
            return data

        if self.norm_method == "mean_std":
            for key in self.norm_keys:
                data[key] = (data[key] - self.norm_stats[key]["mean"]) / (self.norm_stats[key]["std"] + 1e-8)
        elif self.norm_method == "min_max":
            for key in self.norm_keys:
                data[key] = (data[key] - self.norm_stats[key]["min"]) / (self.norm_stats[key]["max"] - self.norm_stats[key]["min"] + 1e-8)
        elif self.norm_method == "quantile":
            for key in self.norm_keys:
                data[key] = (data[key] - self.norm_stats[key]["q01"]) / (self.norm_stats[key]["q99"] - self.norm_stats[key]["q01"] + 1e-8) * 2.0 - 1.0
        else:
            raise ValueError(f"norm_method is {self.norm_method} but norm_stats is None")
        
        return data


@dataclasses.dataclass
class Unnormalize(DataTransformFn):
    norm_method: str
    norm_stats: dict
    norm_keys: list[str]

    def __call__(self, data: Dict) -> Dict:
        if self.norm_stats is None:
            if self.norm_method is not None and self.norm_method != "none":
                raise ValueError(f"norm_method is {self.norm_method} but norm_stats is None")
            return data

        if self.norm_method == "mean_std":
            for key in self.norm_keys:
                data[key] = data[key] * (self.norm_stats[key]["std"] + 1e-6) + self.norm_stats[key]["mean"]
        elif self.norm_method == "min_max":
            for key in self.norm_keys:
                data[key] = data[key] * (self.norm_stats[key]["max"] - self.norm_stats[key]["min"] + 1e-8) + self.norm_stats[key]["min"]
        elif self.norm_method == "quantile":
            for key in self.norm_keys:
                data[key] = data[key] * (self.norm_stats[key]["q99"] - self.norm_stats[key]["q01"] + 1e-8) + self.norm_stats[key]["q01"]
        else:
            raise ValueError(f"norm_method is {self.norm_method} but norm_stats is None")

        return data

    # def _unnormalize_quantile(self, x, stats):
    #     assert stats.q01 is not None
    #     assert stats.q99 is not None
    #     return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


@dataclasses.dataclass
class ResizeImages(DataTransformFn):
    height: int
    width: int
    suffix: str = "history"

    def __call__(self, data: Dict) -> Dict:
        for key, value in data.items():
            if self.suffix in key or "pad" in key: continue
            if key.startswith("observation.images."):
                data[key] = image_tools.resize_with_pad(value, self.width, self.height, pad_value=0)

        return data


@dataclasses.dataclass
class DeltaActions(DataTransformFn):
    """Convert absolute actions to delta actions (action - current_state).
    
    This transform converts actions from absolute values to relative changes
    from the current state: delta_action = action - current_state

    The delta action is: [a0-s, a1-a0, a2-a1, ...]
    """
    state_key: str = "observation.state"
    action_key: str = "action"

    def __call__(self, data: Dict) -> Dict:
        if self.action_key not in data:
            raise ValueError(f"Action key {self.action_key} not found in data")

        state, actions = data[self.state_key], data[self.action_key]
        abs_pre_actions = torch.cat((state[None, :], actions[:-1]), dim=0)
        abs_post_actions = actions.clone()
        actions = abs_post_actions - abs_pre_actions

        data[self.action_key] = actions

        return data


@dataclasses.dataclass
class AbsoluteActions(DataTransformFn):
    """Convert delta actions back to absolute actions.
    
    This transform converts delta actions back to absolute values by accumulating
    the deltas starting from the initial state: abs_action[i] = state + cumsum(delta_actions[0:i+1])
    This is the inverse operation of DeltaActions.

    The absolute action is: [s, s+delta_a0, s+delta_a0+delta_a1, s+delta_a0+delta_a1+delta_a2, ...]
    """
    state_key: str = "observation.state"
    action_key: str = "action"

    def __call__(self, data: Dict) -> Dict:
        if self.action_key not in data:
            raise ValueError(f"Action key {self.action_key} not found in data")

        state, delta_actions = data[self.state_key], data[self.action_key]

        # Convert delta actions back to absolute actions:
        #     abs_action[i] = state + cumsum(delta_actions[0:i+1])
        cumulative_deltas = torch.cumsum(delta_actions, dim=0)
        abs_actions = state[None, :] + cumulative_deltas
        
        data[self.action_key] = abs_actions

        return data


@dataclasses.dataclass
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer

    def __call__(self, data: Dict) -> Dict:
        if (prompt := data.pop("task", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: Dict) -> Dict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: Dict) -> Dict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


def _assert_quantile_stats(norm_stats) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )


def flatten_dict(tree: dict, parent_key: str = '', sep: str = '/') -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    items = []
    for k, v in tree.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, return_mask: bool = False) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        x = np.pad(x, pad_width)
        if return_mask:
            mask = np.ones_like(x)
            mask[..., :current_dim] = 0
            return x, mask
        return x
    return x, np.zeros_like(x)


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def unflatten_dict(tree: dict, sep: str = '/') -> dict:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    result = {}
    for key, value in tree.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def apply_tree(
    tree: Dict, selector: Dict, fn: Callable[[Any, Any], Any], *, strict: bool = False
) -> Dict:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: Any) -> Any:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


class ModelTransformGroup(Group):
    default_prompt: str | None = None
    fast_tokenizer_path: str = "fast"

    def __init__(
        self, 
        policy_config: PreTrainedConfig, 
        n_obs_img_steps: int, 
        n_pred_img_steps: int, 
        obs_img_stride: int,
        und_obs_transforms: tv_transforms.Compose,
        gen_obs_transforms: tv_transforms.Compose,
    ) -> Group:
        super().__init__()
        match policy_config.model_type:
            case "mwm" | "pi0":
                self.inputs = [
                    InjectHistoryObservation(
                        n_obs_img_steps=n_obs_img_steps,
                        n_pred_img_steps=n_pred_img_steps,
                        obs_img_stride=obs_img_stride,
                    ),
                    TransformWorldModelObservation(
                        gen_obs_transforms=gen_obs_transforms,
                    ),
                    UndExpertObsTransform(
                        und_obs_transforms=und_obs_transforms,
                    )
                ]
            case "pi0":
                self.inputs = [
                    # InjectDefaultPrompt(self.default_prompt),
                    # ResizeImages(224, 224),
                    # TokenizePrompt(
                    #     _tokenizer.PaligemmaTokenizer(policy_config.tokenizer_max_length),
                    # ),
                ]

            case "pi0_fast":
                self.inputs = [
                    # InjectDefaultPrompt(self.default_prompt),
                    # ResizeImages(224, 224),
                    TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(policy_config.tokenizer_max_length, fast_tokenizer_path=self.fast_tokenizer_path),
                        ),
                ]
                self.outputs = [
                    ExtractFASTActions(
                        _tokenizer.FASTTokenizer(policy_config.tokenizer_max_length, fast_tokenizer_path=self.fast_tokenizer_path),
                        action_horizon=policy_config.action_horizon,
                        action_dim=policy_config.action_dim,
                    ),
                ]


@dataclasses.dataclass
class UndExpertObsTransform(DataTransformFn):
    und_obs_transforms: tv_transforms.Compose
    suffix: str = "history"

    def __call__(self, data: Dict) -> Dict:
        for k, v in data.items():
            if k.startswith("observation.images.") and "pad" not in k and self.suffix not in k:
                v = image_tools.convert_to_uint8(v.numpy())
                v = Image.fromarray(v.transpose(1, 2, 0))
                for transform in self.und_obs_transforms:
                    v = transform(v)
                # TODO: if max_num_images is not 1, we need to handle this
                data[k] = torch.cat(v)
        return data
