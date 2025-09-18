from dataclasses import dataclass, field

from src.configs.base_config import LeRobotDataConfig

import src.utils.transforms as _transforms

from src.processors.data_processors.ds_transforms.libero_transform import LiberoInputs, LiberoOutputs
from src.processors.data_processors.ds_transforms.bridge_transform import BridgeV2Inputs, BridgeV2Outputs
from src.processors.data_processors.ds_transforms.agibotworld_transform import AgiBotWorldInputs, AgiBotWorldOutputs
from src.processors.data_processors.ds_transforms.fractal_transform import FractalInputs, FractalOutputs
from src.processors.data_processors.ds_transforms.split_aloha_transform import SplitAlohaInputs, SplitAlohaOutputs
from src.processors.data_processors.ds_transforms.arx_lift2_transform import ARXLift2Inputs, ARXLift2Outputs

@dataclass
class LeRobotLiberoDataConfig(LeRobotDataConfig):

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method, action_mode="abs"):
        """Create Libero-specific data transforms."""
        return _transforms.Group(
            inputs=[
                LiberoInputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    action_mode=action_mode,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[LiberoOutputs()],
        )


@dataclass
class LeRobotAgiBotWorldDataConfig(LeRobotDataConfig):
    gripper_dim: list[int] = field(default_factory=lambda: [0, 1])

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method, action_mode="delta"):
        """Create AgiBotWorld-specific data transforms."""
        return _transforms.Group(
            inputs=[
                AgiBotWorldInputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    action_mode=action_mode,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[AgiBotWorldOutputs()],
        )


@dataclass
class LeRobotBridgeV2DataConfig(LeRobotDataConfig):

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method, action_mode="abs"):
        """Create BridgeV2-specific data transforms."""
        return _transforms.Group(
            inputs=[
                BridgeV2Inputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    action_mode=action_mode,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[BridgeV2Outputs()],
        )


@dataclass
class LeRobotFractalDataConfig(LeRobotDataConfig):

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method, action_mode="abs"):
        """Create Fractal-specific data transforms."""
        return _transforms.Group(
            inputs=[
                FractalInputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    action_mode=action_mode,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[FractalOutputs()],
        )


@dataclass
class LeRobotSplitAlohaDataConfig(LeRobotDataConfig):
    gripper_dim: list[int] = field(default_factory=lambda: [6, 13])

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method, action_mode="abs"):
        """Create SplitAloha-specific data transforms."""
        return _transforms.Group(
            inputs=[
                SplitAlohaInputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    action_mode=action_mode,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[SplitAlohaOutputs()],
        )
    
@dataclass
class LeRobotARXLift2DataConfig(LeRobotDataConfig):
    gripper_dim: list[int] = field(default_factory=lambda: [6, 13])

    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method, action_mode="abs"):
        """Create SplitAloha-specific data transforms."""
        return _transforms.Group(
            inputs=[
                ARXLift2Inputs(
                    action_dim=policy_config.max_action_dim, 
                    camera_keys=self.camera_keys,
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                    action_mode=action_mode,
                    norm_stats=ds_stats,
                    do_normalize=do_normalize,
                    norm_method=norm_method,
                ),
            ],
            outputs=[ARXLift2Outputs()],
        )
