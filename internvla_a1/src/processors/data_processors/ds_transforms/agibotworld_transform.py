import dataclasses
import torch

from src.processors.data_processors.ds_transforms.base_transform import (
    BaseTransform,
    BaseTransformOutputs
)
from src.utils.action_utils import (
    convert_action_abs2delta,
    convert_action_abs2rel
)


@dataclasses.dataclass
class AgiBotWorldInputs(BaseTransform):
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

        # reformulate the task
        data["task"] = data["task"].split("|")[0].strip()

        # reformulate state and action
        # state
        state_left_arm_gripper = torch.cat(
            [
                data["observation.states.joint.position"][:self.single_arm_dim],
                data["observation.states.effector.position"][0].unsqueeze(-1),
            ],
            dim=-1
        )
        state_right_arm_gripper = torch.cat(
            [
                data["observation.states.joint.position"][self.single_arm_dim:2*self.single_arm_dim],
                data["observation.states.effector.position"][1].unsqueeze(-1),
            ],
            dim=-1
        )
        state_arm_gripper = torch.cat(
            [
                state_left_arm_gripper,
                state_right_arm_gripper,
            ],
            dim=-1
        )
        # state_head = data["observation.states.head.position"]
        # state_waist = data["observation.states.waist.position"]
        state = torch.cat(
            [
                state_arm_gripper,
                # state_head, 
                # state_waist
            ], 
            dim=-1
        )
        assert state.shape[0] == 16, f"state shape is {state.shape[0]}"
        data["observation.state"] = state
        for k in self.state_keys:
            del data[k]

        # action
        action_left_arm_gripper = torch.cat(
            (
                data["actions.joint.position"][:, :self.single_arm_dim],
                data["actions.effector.position"][:, 0].unsqueeze(-1),
            ),
            dim=-1
        )
        action_right_arm_gripper = torch.cat(
            (
                data["actions.joint.position"][:, self.single_arm_dim:2*self.single_arm_dim],
                data["actions.effector.position"][:, 1].unsqueeze(-1),
            ),
            dim=-1
        )
        action_arm_gripper = torch.cat(
            [
                action_left_arm_gripper,
                action_right_arm_gripper,
            ],
            dim=-1
        )
        action = torch.cat(
            [
                action_arm_gripper,
                # data["actions.head.position"],
                # data["actions.waist.position"],
            ], 
            dim=-1
        )
        # assert action.shape[-1] == 8 * 2 + 2 + 2, f"action shape is {action.shape[-1]}"
        data["action"] = action
        data["action_is_pad"] = data["actions.joint.position_is_pad"]
        for k in self.action_keys:
            del data[k]
            del data[f"{k}_is_pad"]    

        return data


@dataclasses.dataclass
class AgiBotWorldOutputs(BaseTransformOutputs):
    action_dim: int = 16
    single_arm_dim: int = 7
    single_gripper_dim: int = 1

    def __call__(self, action: torch.Tensor, raw_format: bool = True) -> list:
        assert len(action.shape) == 2, f"action shape is {action.shape}"
        action = action[:, :self.action_dim]

        # split the action into left and right arm
        action_left_arm_gripper = action[:, :self.single_arm_dim + self.single_gripper_dim]
        action_right_arm_gripper = action[:, self.single_arm_dim + self.single_gripper_dim:2*(self.single_arm_dim + self.single_gripper_dim)]

        action_left_joint = action_left_arm_gripper[:, :self.single_arm_dim]
        action_left_gripper = action_left_arm_gripper[:, self.single_arm_dim:self.single_arm_dim + self.single_gripper_dim]

        action_right_joint = action_right_arm_gripper[:, :self.single_arm_dim]
        action_right_gripper = action_right_arm_gripper[:, self.single_arm_dim:self.single_arm_dim + self.single_gripper_dim]

        # merge the action into the joint and gripper
        action_joint = torch.cat(
            [
                action_left_joint,
                action_right_joint,
            ],
            dim=-1  
        )
        action_gripper = torch.cat(
            [
                action_left_gripper,
                action_right_gripper,
            ],
            dim=-1
        )

        # compose the action output
        action = {
            "actions.joint.position": action_joint,
            "actions.effector.position": action_gripper,
        }

        if self.do_normalize:
            action = self.unnorm_action(action)

        if raw_format:
            action = torch.cat(
                [
                    action["actions.joint.position"],
                    action["actions.effector.position"],
                ],
                dim=-1
            )
            return action
    
        raise NotImplementedError("Raw format is not implemented for AgiBotWorldOutputs")
