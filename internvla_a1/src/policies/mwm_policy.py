import os
import logging
import packaging
from pathlib import Path
from collections import deque

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

import safetensors
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi0.modeling_pi0 import pad_vector

from transformers import AutoTokenizer
from transformers.utils import logging

from src.models.modeling_mwm import MWMFlowMatching
from src.models.configuration_mwm import MWMConfig
from src.utils.image_tools import resize_with_pad

from src.models.cosmos_tokenizer.image_lib import ImageTokenizer


logger = logging.get_logger(__name__)

class MWMPolicy(nn.Module):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = MWMConfig
    cache_action_steps = 5

    def __init__(
        self,
        config: MWMConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        input_features: dict[str, FeatureType] | None = None,
        output_features: dict[str, FeatureType] | None = None,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__()
        self.config = config
        self.use_world_model = config.use_world_model
        if input_features is not None and output_features is not None:
            # Hack
            print("HACK: the normalization type is mean std !!!!")
            normalization_mapping: dict[str, NormalizationMode] = {
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MEAN_STD,
                "ACTION": NormalizationMode.MEAN_STD,
            }
            self.normalize_inputs = Normalize(
                input_features, normalization_mapping, dataset_stats
            )
            self.normalize_targets = Normalize(
                output_features, normalization_mapping, dataset_stats
            )
            self.unnormalize_outputs = Unnormalize(
                output_features, normalization_mapping, dataset_stats
            )

        if self.use_world_model:
            self.cosmos_encoder = ImageTokenizer(checkpoint_enc=f"{config.image_tokenizer_path}/encoder.jit")
            self.cosmos_decoder = ImageTokenizer(checkpoint_dec=f"{config.image_tokenizer_path}/decoder.jit")
            self.gen_loss_fct = nn.CrossEntropyLoss(reduction="mean")

        self.language_tokenizer = AutoTokenizer.from_pretrained(config.language_tokenizer_path)
        self.model = MWMFlowMatching(config, **kwargs)

        self.reset()

        self.param_info()

    def param_info(self):
        total_params = 0
        for _, p in self.named_parameters():
            total_params += p.data.numel()
        print(f"Total parameters: {total_params:,}")

        und_expert_params = 0
        for _, p in self.model.internvl_with_expert.und_expert.named_parameters():
            und_expert_params += p.data.numel()
        print(f"Und expert parameters: {und_expert_params:,}")

        gen_expert_params = 0
        for _, p in self.model.internvl_with_expert.gen_expert.named_parameters():
            gen_expert_params += p.data.numel()
        print(f"Gen expert parameters: {gen_expert_params:,}")

        act_expert_params = 0
        for _, p in self.model.internvl_with_expert.act_expert.named_parameters():
            act_expert_params += p.data.numel()
        print(f"Act expert parameters: {act_expert_params:,}")

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.cache_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None, return_loss: bool = False) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])

        # pad the state
        if batch["observation.state"].shape[-1] == 7:
            pad_state = torch.tensor([0]).unsqueeze(0).cuda()
            batch["observation.state"] = torch.cat((batch["observation.state"][:, :6], pad_state, batch["observation.state"][:, 6].unsqueeze(1)), dim=-1).cuda()

        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            images, img_masks = self.prepare_mix_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )

            # if self.cache_action_steps is not None:
            #     actions = actions[:, :self.cache_action_steps]

            # Unpad actions
            original_action_dim = 7
            actions = actions[:, :, :original_action_dim]

            actions = self.unnormalize_outputs({"action": actions})["action"]

            return actions

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1)[:5])

        return self._action_queue.popleft()

    @torch.no_grad
    def select_action_with_world_model(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        predict_action_only: bool = False,
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        # batch = self.normalize_inputs(batch)
        # action = {"action": self.normalize_targets(batch)["action"]}

        if len(self._action_queue) == 0:
            images, image_masks = self.prepare_mix_images(batch)
            world_model_images, world_model_image_masks = self.prepare_mix_history_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            B, N_view, T, C, H, W = world_model_images.shape
            world_model_images = world_model_images.reshape(B * N_view * T, C, H, W)

            with torch.no_grad():
                world_model_image_latents, _ = self.cosmos_encoder.encode(world_model_images)

            # prepare the input of world model
            world_model_embs = self.model.wm_embeddings(world_model_image_latents)
            world_model_embs = world_model_embs.reshape(B, N_view, T, *world_model_embs.shape[1:])
            world_model_input_embs = world_model_embs[:, :, :2]

            action_output = self.model.sample_actions_with_world_model(
                images=images, 
                image_masks=image_masks, 
                lang_tokens=lang_tokens, 
                lang_masks=lang_masks, 
                state=state, 
                world_model_input_embs=world_model_input_embs, 
                world_model_input_masks=world_model_image_masks,
                predict_action_only=False, 
                noise=noise,
                # actions=self.prepare_action(action),
            )
            actions = action_output.actions
            pred_img_indicies = action_output.pred_img_indices

            # if self.cache_action_steps is not None:
            #     actions = actions[:, :self.cache_action_steps]

            # Unpad actions
            # TODO: (aopolin) fix this bug, include a param to decide the action dim
            # original_action_dim = 7 # self.config.action_feature.shape[0]
            # actions = actions[:, :, :original_action_dim]

            # actions = self.unnormalize_outputs({"action": actions})["action"]

            if predict_action_only:
                return actions

            pred_imgs = self.cosmos_decoder.decode(pred_img_indicies)
            # end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")

            return (actions, pred_imgs)

    def forward_with_world_model(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        time: Tensor | None = None, 
        cur_n_obs_img_steps: int | None = None, 
        cur_n_pred_img_steps: int | None = None,  
        train_gen_expert_only: bool = False, 
        gen_out_loss_ratio: float = 0.1
    ) -> dict[str, Tensor]:

        #########################################################
        # prepare the inputs
        #########################################################
        # batch = self.normalize_inputs(batch)
        # batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_mix_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")

        world_model_images, world_model_image_masks = self.prepare_mix_history_images(batch)
        B, N_view, T, C, H, W = world_model_images.shape
        world_model_images = world_model_images.reshape(B * N_view * T, C, H, W) # (bs * 3 * 3, 3, 256, 256)
        with torch.no_grad():
            world_model_image_latents, _ = self.cosmos_encoder.encode(world_model_images) # (bs * 3 * 3, 32, 32)

        # prepare the output of world model
        gt_world_model_indices = world_model_image_latents.reshape(B, N_view, T, -1)
        gt_world_model_indices = gt_world_model_indices[:, :, cur_n_obs_img_steps: cur_n_obs_img_steps + cur_n_pred_img_steps].contiguous()
        gt_world_model_indices = gt_world_model_indices.squeeze(2)
        gt_world_model_indices = torch.where(
            world_model_image_masks[:, :, None].repeat(1, 1, gt_world_model_indices.shape[2]), 
            gt_world_model_indices, 
            torch.ones_like(gt_world_model_indices) * -100
        )

        # prepare the input of world model
        world_model_embs = self.model.wm_embeddings(world_model_image_latents)
        # split the input and output of world model
        world_model_embs = world_model_embs.reshape(B, N_view, T, *world_model_embs.shape[1:])
        world_model_input_embs = world_model_embs[:, :, :cur_n_obs_img_steps]

        # visualize for debug
        # from torchvision.utils import save_image
        # save_image(samples, "samples.png", nrow=T, normalize=True, value_range=(-1, 1))

        #########################################################
        # Forward and compute the loss
        #########################################################
        action_losses, gen_logits = self.model.forward_with_world_model(
            images=images, # list<Tensor(bs, 3, 448, 448)>
            img_masks=img_masks, # list<Tensor(bs,)>
            lang_tokens=lang_tokens, # (bs, seq_len), seq_len: 48
            lang_masks=lang_masks, # (bs, seq_len)
            state=state, # (bs, 32)
            world_model_input_embs=world_model_input_embs, # (bs, n_cam, 2, h//8, w//8, hidden_dim)
            world_model_input_masks=world_model_image_masks, # (bs, n_cam)
            actions=actions, # (bs, chunk_size, 32)
            noise=noise, 
            time=time
        )

        gen_logits = gen_logits.reshape(B, -1, gen_logits.shape[-1])
        gt_world_model_indices = gt_world_model_indices.reshape(B, -1).long()

        gen_logits = gen_logits.view(-1, gen_logits.shape[-1])
        gt_world_model_indices = gt_world_model_indices.view(-1)

        gen_loss = self.gen_loss_fct(gen_logits, gt_world_model_indices)

        loss_dict = {}
        loss_dict["gen_acc_mean"] = (gen_logits.argmax(dim=-1) == gt_world_model_indices).float().mean()

        if train_gen_expert_only:
            loss_dict["loss"] = gen_loss
            loss_dict["gen_loss"] = gen_loss
            return loss_dict

        loss_dict["action_losses_after_forward"] = action_losses.clone()

        if action_is_pad is not None:
            in_episode_bound = ~action_is_pad
            if action_losses.shape == in_episode_bound.shape:
                action_losses = action_losses * in_episode_bound
            else:
                action_losses = action_losses * in_episode_bound.unsqueeze(-1)
            loss_dict["action_losses_after_in_ep_bound"] = action_losses.clone()

        # Remove padding
        action_losses = action_losses[:, :, : self.config.max_action_dim]
        loss_dict["action_losses_after_rm_padding"] = action_losses.clone()
        loss_dict["action_loss"] = action_losses.mean().clone()
        loss_dict["gen_loss"] = gen_loss.clone()

        loss_dict["loss"] = loss_dict["action_loss"] + gen_out_loss_ratio * loss_dict["gen_loss"]

        return loss_dict

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""

        # batch = self.normalize_inputs(batch)
        # batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_mix_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if action_is_pad is not None:
            in_episode_bound = ~action_is_pad
            if losses.shape == in_episode_bound.shape:
                losses = losses * in_episode_bound
            else:
                losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss_dict["original_loss"] = losses.mean()
        loss = loss_dict["original_loss"]

        loss_dict["loss"] = loss
        loss_dict["action_loss"] = losses.mean()

        return loss_dict

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            
            if len(img.shape) == 5:
                # only use the current step image
                img = img[:, -1]

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            # img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_mix_images(self, batch):
        images = []
        image_masks = []

        # Hack
        img_keys = [
            "observation.images.image0",
            "observation.images.image1",
            "observation.images.image2",
        ]

        for key in img_keys:
            if key not in batch:
                img = torch.zeros_like(batch["observation.images.image0"])
                mask = torch.zeros_like(batch["observation.images.image0_mask"])
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                images.append(img)
                image_masks.append(mask)
                continue
            img = batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            images.append(img)
            image_masks.append(batch[f"{key}_mask"])
        
        # delete the empty images
        for i in range(len(images) - 1, -1, -1):
            if images[i].sum() == 0:
                images.pop(i)
                image_masks.pop(i)

        return images, image_masks

    def prepare_mix_history_images(self, batch):
        images = []
        image_masks = []

        # Hack
        img_keys = [
            "observation.images.image0_history",
            "observation.images.image1_history",
            "observation.images.image2_history",
        ]

        for key in img_keys:
            if key not in batch:
                img = torch.zeros_like(batch["observation.images.image0_history"])
                mask = torch.zeros_like(batch["observation.images.image0_history_mask"])
                images.append(img)
                image_masks.append(mask)
                continue
            img = batch[key]
            images.append(img)
            image_masks.append(batch[f"{key}_mask"])

        for i in range(len(images) - 1, -1, -1):
            if images[i].sum() == 0:
                images.pop(i)
                image_masks.pop(i)

        images = torch.stack(images, dim=0)
        images = images.permute(1, 0, 2, 3, 4, 5)  # B, N_cam, T, C, H, W

        image_masks = torch.stack(image_masks, dim=0)
        image_masks = image_masks.permute(1, 0)  # B, N_cam

        return images, image_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_ROBOT].device
        tasks = batch["task"]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: MWMConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> "MWMPolicy":
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = MWMConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)

        if os.path.isdir(model_id):
            print(f"Loading weights from local directory: {model_id}")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, "cpu", strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, "cpu", strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):
            load_model_as_safetensor(model, model_file, strict=strict)
            if map_location != "cpu":
                logger.warning(
                    "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                    " This means that the model is loaded on 'cpu' first and then copied to the device."
                    " This leads to a slower loading time."
                    " Please update safetensors to version 0.4.3 or above for improved performance."
                )
                model.to(map_location)
        else:
            safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
        return model


    def _save_pretrained(self, save_directory: Path) -> None:
        self.config.save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))