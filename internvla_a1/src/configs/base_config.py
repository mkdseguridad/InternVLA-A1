import dataclasses
from collections.abc import Sequence
import src.utils.transforms as _transforms
from src.utils.transforms import ModelTransformGroup
from abc import abstractmethod

@dataclasses.dataclass
class DataConfig:
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    camera_names: Sequence[str] = None
    state_keys: Sequence[str] = None
    action_keys: Sequence[str] = None
    predict_img_keys: Sequence[str] = None

    local_path: str | None = None
    weight: float = 1

    n_obs_img_steps: int = 12
    n_pred_img_steps: int = 3
    obs_img_stride: int = 3

    num_indices: int = 50

    image_size: tuple[int, int] = (224, 224)
    tolerance_s: float = 1e-4  # Tolerance for video timestamp synchronization (default 0.02 seconds)


@dataclasses.dataclass
class LeRobotDataConfig(DataConfig):
    """Base class for LeRobot dataset configurations with common initialization logic."""
    
    def initialize(
        self, 
        policy_config, 
        camera_keys: list[str], 
        state_keys: list[str],
        action_keys: list[str],
        predict_img_keys: list[str],
        gripper_action_keys: list[str],
        image_size: dict,
        gen_obs_suffix: str,
        local_path: str, 
        ds_stats: dict,
        weight: int = 1,
        action_mode: str = "abs",
        do_normalize: bool = True,
        norm_method: str | None = None,
        n_obs_img_steps: int | None = None, 
        n_pred_img_steps: int | None = None, 
        obs_img_stride: int | None = None,
        num_indices: int | None = None,
        und_expert_obs_transforms: list | None = None,
        gen_obs_transforms: list | None = None,
        gripper_dim: list[int] | None = None,
        tolerance_s: float | None = None,
    ):
        # Store basic attributes
        self.local_path = local_path
        self.camera_keys = camera_keys
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.predict_img_keys = predict_img_keys
        self.gripper_action_keys = gripper_action_keys
        if gripper_dim is None:
            self.gripper_dim = getattr(self, 'gripper_dim', [-1])
        else:
            self.gripper_dim = gripper_dim
        self.image_size = image_size
        self.suffix = gen_obs_suffix
        self.weight = weight
        self.action_mode = action_mode

        # Set optional attributes if provided
        if n_obs_img_steps: self.n_obs_img_steps = n_obs_img_steps
        if n_pred_img_steps: self.n_pred_img_steps = n_pred_img_steps
        if obs_img_stride: self.obs_img_stride = obs_img_stride
        if num_indices: self.num_indices = num_indices
        if und_expert_obs_transforms: self.und_expert_obs_transforms = und_expert_obs_transforms
        if gen_obs_transforms: self.gen_obs_transforms = gen_obs_transforms
        if tolerance_s is not None: self.tolerance_s = tolerance_s

        # Create dataset-specific transforms (to be implemented by subclasses)
        self.data_transforms = self._create_data_transforms(
            policy_config=policy_config, 
            action_mode=action_mode,
            ds_stats=ds_stats, 
            do_normalize=do_normalize, 
            norm_method=norm_method
        )

        # Create common repack transforms
        self.repack_transforms = self._create_repack_transforms(
            camera_keys=camera_keys, 
            predict_img_keys=predict_img_keys, 
            gen_obs_suffix=gen_obs_suffix
        )

        # Create model transforms
        self.model_transforms = ModelTransformGroup(
            policy_config=policy_config, 
            n_obs_img_steps=self.n_obs_img_steps, 
            n_pred_img_steps=self.n_pred_img_steps, 
            obs_img_stride=self.obs_img_stride,
            und_obs_transforms=self.und_expert_obs_transforms,
            gen_obs_transforms=self.gen_obs_transforms
        )

        return self

    @abstractmethod
    def _create_data_transforms(self, policy_config, ds_stats, do_normalize, norm_method, action_mode="abs"):
        """Create dataset-specific data transforms. To be implemented by subclasses."""
        pass

    def _create_repack_transforms(self, camera_keys, predict_img_keys, gen_obs_suffix):
        """Create common repack transforms."""
        
        repack_image_keys = {f"observation.images.image{idx}": key for idx, key in enumerate(camera_keys)}
        repack_pred_image_keys = {f"observation.images.image{idx}_{gen_obs_suffix}": key for idx, key in enumerate(predict_img_keys)}
        repack_pred_image_pad_keys = {f"observation.images.image{idx}_{gen_obs_suffix}_is_pad": key for idx, key in enumerate(predict_img_keys)}
        return _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        **repack_image_keys,
                        **repack_pred_image_keys,
                        **repack_pred_image_pad_keys,
                        "observation.state": "observation.state",

                        "action": "action",
                        "action_is_pad": "action_is_pad",

                        "task": "task",
                    }
                ),
                _transforms.ResizeImages(
                    height=self.image_size.height, 
                    width=self.image_size.width, 
                    suffix=self.suffix
                ),
            ]
        )

    @abstractmethod
    def _apply_delta_actions(self):
        """Apply delta actions if needed. Can be overridden by subclasses."""
        pass
        
