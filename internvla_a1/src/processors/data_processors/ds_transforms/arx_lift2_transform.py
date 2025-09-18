import dataclasses
import numpy as np
import torch

from src.processors.data_processors.ds_transforms.base_transform import (
    BaseTransform,
    BaseTransformOutputs
)
from src.utils.action_utils import (
    convert_action_abs2delta,
    convert_action_abs2rel
)
from copy import deepcopy


@dataclasses.dataclass
class ARXLift2Inputs(BaseTransform):
    single_arm_dim: int = 7

    def __call__(self, data: dict) -> dict:

        keys_to_delete = [k for k in data.keys() if k not in self.valid_keys]
        for k in keys_to_delete:
            del data[k]

        assert self.action_mode in ["abs", "delta", "rel"]
        if self.action_mode == "delta":
            data = convert_action_abs2delta(data, self.action_keys)
        elif self.action_mode == "rel":
            data = convert_action_abs2rel(data, self.action_keys)

        if self.do_normalize:
            data = self.normalize_state(data)
            data = self.normalize_action(data)

        state_left_arm = data["states.left_joint.position"]
        state_left_gripper = data["states.left_gripper.position"]
        state_right_arm = data["states.right_joint.position"]
        state_right_gripper = data["states.right_gripper.position"]

        state = torch.cat([
            state_left_arm, state_left_gripper, state_right_arm, state_right_gripper
        ], dim=0)

        action_left_arm = data["actions.left_joint.position"]
        action_left_gripper = data["actions.left_gripper.position"]
        action_right_arm = data["actions.right_joint.position"]
        action_right_gripper = data["actions.right_gripper.position"]

        if action_left_gripper.ndim == 1:
            action_left_gripper = action_left_gripper[:, None]
        if action_right_gripper.ndim == 1:
            action_right_gripper = action_right_gripper[:, None]

        action = torch.cat([
            action_left_arm, action_left_gripper, action_right_arm, action_right_gripper
        ], dim=1)
        action_is_pad = deepcopy(data["actions.left_joint.position_is_pad"])

        data["observation.state"] = state
        data["action"] = action
        data["action_is_pad"] = action_is_pad

        for k in self.state_keys:
            del data[k]

        for k in self.action_keys:
            del data[k]
            del data[f"{k}_is_pad"]

        return data


@dataclasses.dataclass
class ARXLift2Outputs(BaseTransformOutputs):
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"action": np.asarray(data["action"][:, :self.action_dim])}
