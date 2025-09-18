import os
import io
import time
import base64
import random
import imageio
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple

from openai import OpenAI

import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.utils.utils import LargeScaleWeightedRandomSampler
from src.policies.mwm_policy import MWMPolicy

from lerobot.common.policies.pretrained import PreTrainedPolicy
from transformers import Trainer, __version__, PretrainedConfig
from transformers.trainer import (
    logger, 
    FSDP_MODEL_NAME, 
    TRAINING_ARGS_NAME, 
    is_peft_available, 
    _get_fsdp_ckpt_kwargs,
    _is_peft_model,
)
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
    is_torch_xla_available
)
from transformers.trainer_utils import (
    EvalLoopOutput, 
    has_length, 
    denumpify_detensorize,
    PREFIX_CHECKPOINT_DIR
)
from transformers.trainer_pt_utils import (
    EvalLoopContainer, 
    find_batch_size, 
    IterableDatasetShard
)
from transformers.integrations.deepspeed import deepspeed_init
from transformers.integrations.integration_utils import is_wandb_available, rewrite_logs

if is_peft_available():
    from peft import PeftModel

if is_accelerate_available():
    from accelerate.utils import load_fsdp_model


@dataclass
class PI0TrainingArguments(TrainingArguments):
    train_dir: str | None = None
    eval_dir: str | None = None
    num_eval_episodes: int = 50
    stage: str = "stage_vla"
    language_tokenizer_path: str | None = None

    freeze_vision_encoder: bool = False
    train_act_expert_only: bool = False
    train_gen_expert_only: bool = False
    train_state_proj: bool = True

    gen_out_loss_ratio: float = 0.0

    resize_imgs_with_padding: Tuple[int, int] = (224, 224)

    image_transforms_enabled: bool = True
    image_transforms_max_num_transforms: int = 3
    image_transforms_random_order: bool = True
    image_transforms_type: List[str] = field(default_factory=lambda: ["brightness", "contrast", "saturation", "random_crop", "random_rotation"])

    und_expert_lr: float = 0.0
    act_expert_lr: float = 0.0
    gen_expert_lr: float = 0.0
    vision_encoder_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        random.seed(self.seed)


class PolicyTrainerCallback(TrainerCallback):
    policy: None
    image_transforms: None
    def __init__(self, policy, image_transforms):
        self.policy = policy
        self.image_transforms = image_transforms

    def on_train_begin(self, args, state, control, **kwargs):
        """ move the normalize_inputs and normalize_targets to the device """
        if self.policy is not None:
            if hasattr(self.policy, "normalize_inputs"):
                self.policy.normalize_inputs.to(args.device)
            if hasattr(self.policy, "normalize_targets"):
                self.policy.normalize_targets.to(args.device)
            if hasattr(self.policy, "unnormalize_outputs"):
                self.policy.unnormalize_outputs.to(args.device)
        if self.image_transforms is not None:
            self.image_transforms.to(args.device)


class PolicyTrainer(Trainer):
    def __init__(
        self, 
        policy: Union[PreTrainedPolicy, MWMPolicy], 
        image_transforms=None, 
        use_world_model=True,
        cur_n_obs_img_steps=None, 
        cur_n_pred_img_steps=None, 
        training_ds_sample_weights=None,
        eval_ds_sample_weights=None,
        save_pred_img=True,
        save_gt_img=True,
        save_rec_img=True,
        save_wm_comparison=True,
        max_eval_samples=None,
        eval_url=None,
        eval_mllm_model_name="Qwen/Qwen2.5-VL-32B-Instruct",
        *args, 
        **kwargs
    ):
        self.policy = policy
        self.image_transforms = image_transforms
        self.use_world_model = use_world_model
        # TODO: make this configurable
        self.pred_img_keys = ["observation.images.image0_history"]
        assert len(self.pred_img_keys) == 1, "Only one image key is supported for now"

        self.cur_n_obs_img_steps = cur_n_obs_img_steps
        self.cur_n_pred_img_steps = cur_n_pred_img_steps

        move_callbacks = [PolicyTrainerCallback(policy=policy, image_transforms=image_transforms)]

        self.training_ds_sample_weights = training_ds_sample_weights
        self.eval_ds_sample_weights = eval_ds_sample_weights

        self.save_pred_img = save_pred_img
        self.save_gt_img = save_gt_img
        self.save_rec_img = save_rec_img
        self.save_gen_comparison = save_wm_comparison
        self.save_pred_img_idx = 0
        self.save_gt_img_idx = 0
        self.save_rec_img_idx = 0
        self.save_wm_in_img_idx = 0
        self.max_eval_samples = max_eval_samples

        self.worker_idx = int(os.environ.get("MLP_ROLE_INDEX", 0))
        self.local_rank_idx = int(os.environ.get('LOCAL_RANK', -1))

        self.eval_if = False
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"{eval_url}",
        )
        self.eval_mllm_model_name = eval_mllm_model_name

        super().__init__(model=policy, callbacks=move_callbacks, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # apply image transforms to the inputs of understanding expert
        if self.image_transforms is not None:
            for key, value in inputs.items():
                if "history" in key or "mask" in key:
                    continue
                if key.startswith("observation.images"):
                    inputs[key] = self.image_transforms(value)

        if self.policy.use_world_model:
            outputs = self.policy.forward_with_world_model(
                inputs, 
                cur_n_obs_img_steps=self.cur_n_obs_img_steps, 
                cur_n_pred_img_steps=self.cur_n_pred_img_steps,
                train_gen_expert_only=self.args.train_gen_expert_only,
                gen_out_loss_ratio=self.args.gen_out_loss_ratio,
            )
        else:
            outputs = self.policy.forward(inputs)

        loss = outputs["loss"]

        if self.state.is_local_process_zero and self.state.is_world_process_zero:
            if self.state.global_step % self.state.logging_steps == 0 and self.state.global_step != 0:
                action_lr_log = {
                    "action_learning_rate": self.optimizer.param_groups[-1]["lr"],
                }
                action_log = {
                    "action_loss": outputs.get("action_loss", torch.tensor(0)).cpu().item(),
                }
                if self.policy.use_world_model:
                    wm_log = {
                        "gen_loss": outputs.get("gen_loss", torch.tensor(0)).cpu().item(),
                        "gen_acc_mean": outputs.get("gen_acc_mean", torch.tensor(0)).cpu().item(),
                        "gen_learning_rate": self.optimizer.param_groups[4]["lr"],
                    }
                    vit_log = {
                        "vit_learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                    if self.policy.model.train_gen_expert_only:
                        loss_dict = {**wm_log, **vit_log}
                    else:
                        loss_dict = {**wm_log, **vit_log, **action_lr_log, **action_log}
                else:
                    loss_dict = {**action_lr_log, **action_log}

                self.log(loss_dict)

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_action_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_gen_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_pred_wm_indices = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_pred_wm_gt_indices = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_l2_loss_imgs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if self.max_eval_samples is not None and observed_num_examples >= self.max_eval_samples:
                break
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            (
                loss, 
                loss_action, 
                loss_gen, 
                action_pred, 
                action_gt, 
                pred_wm_indices, 
                gt_wm_indices,
                l2_loss_imgs,
            ) = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if loss is not None:
                loss = self.gather_function((loss.repeat(batch_size)))
                all_losses.add(loss)
            if loss is not None:
                action_losses = self.gather_function((loss_action.repeat(batch_size)))
                all_action_losses.add(action_losses)
            if loss is not None:
                gen_losses = self.gather_function((loss_gen.repeat(batch_size)))
                all_gen_losses.add(gen_losses)
            if pred_wm_indices is not None:
                pred_wm_indices = self.gather_function((pred_wm_indices))
                all_pred_wm_indices.add(pred_wm_indices)
            if gt_wm_indices is not None:
                gt_wm_indices = self.gather_function((gt_wm_indices))
                all_pred_wm_gt_indices.add(gt_wm_indices)
            if l2_loss_imgs is not None:
                l2_loss_imgs = self.gather_function((l2_loss_imgs))
                all_l2_loss_imgs.add(l2_loss_imgs)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if action_gt is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                action_gt = self.accelerator.pad_across_processes(action_gt, dim=1, pad_index=-100)
            if action_pred is not None:
                action_pred = self.accelerator.pad_across_processes(action_pred, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    action_pred = self.preprocess_logits_for_metrics(action_pred, action_gt)
                action_pred = self.gather_function((action_pred))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(action_pred)
            if action_gt is not None:
                action_gt = self.gather_function((action_gt))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(action_gt)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and action_pred is not None and action_gt is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = loss if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=action_pred, label_ids=action_gt, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del loss, action_pred, action_gt, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_action_losses.to_cpu_and_numpy()
                all_gen_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                all_pred_wm_indices.to_cpu_and_numpy()
                all_pred_wm_gt_indices.to_cpu_and_numpy()
                all_l2_loss_imgs.to_cpu_and_numpy()

                del loss, action_losses, gen_losses, action_pred, action_gt, inputs, pred_wm_indices, gt_wm_indices
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_action_losses = all_action_losses.get_arrays()
        all_gen_losses = all_gen_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        all_pred_wm_indices = all_pred_wm_indices.get_arrays()
        all_pred_wm_gt_indices = all_pred_wm_gt_indices.get_arrays()
        all_l2_loss_imgs = all_l2_loss_imgs.get_arrays()
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["action_losses"] = all_action_losses if "action_losses" in args.include_for_metrics else None
            eval_set_kwargs["wm_losses"] = all_gen_losses if "wm_losses" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            eval_set_kwargs["l2_loss_imgs"] = all_l2_loss_imgs if "l2_loss_imgs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(
                    predictions=all_preds, 
                    label_ids=all_labels, 
                    pred_wm_indices=all_pred_wm_indices,
                    gt_wm_indices=all_pred_wm_gt_indices,
                    **eval_set_kwargs
                ),
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if isinstance(all_action_losses, list) and all_action_losses:
            metrics[f"{metric_key_prefix}_action_loss"] = np.concatenate(all_action_losses).mean().item()
        elif isinstance(all_action_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_action_loss"] = all_action_losses.mean().item()
        if isinstance(all_gen_losses, list) and all_gen_losses:
            metrics[f"{metric_key_prefix}_wm_loss"] = np.concatenate(all_gen_losses).mean().item()
        elif isinstance(all_gen_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_wm_loss"] = all_gen_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        if self.use_world_model:
            return self._prediction_step_with_world_model(inputs)
            # return self._save_eval_images(inputs)
        else:
            return self._prediction_step_directly(inputs)

    def _save_eval_images(self, inputs):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                # prepare world model input
                target = {}
                for key, value in inputs.items():
                    if key.startswith("observation.image") and "history" in key:
                        assert len(value.shape) == 5, "The observation image input should be 5 dims: (B T C H W)"
                        inputs[key] = value[:, :self.cur_n_obs_img_steps]
                        if key in self.pred_img_keys:
                            target[key.replace('history', 'pred')] = value[:, self.cur_n_obs_img_steps:self.cur_n_obs_img_steps + self.cur_n_pred_img_steps]

                target["observation.images.image0_pred"] = target["observation.images.image0_pred"][:, -1]
                pred_img = self.policy.vae.img_to_reconstructed_img(target["observation.images.image0_pred"], last_one=True)
                gt_img = target["observation.images.image0_pred"]

        if isinstance(pred_img, list):
            pred_img = torch.cat(pred_img, dim=0)

        gt_img = target[self.pred_img_keys[0].replace('history', 'pred')]
        gt_wm_indices = self.policy.vae.img_to_idxBl(gt_img.reshape(-1, *gt_img.shape[-3:]))
        gt_decode_img = self.policy.vae.idxBl_to_img(gt_wm_indices, same_shape=False, last_one=True)

        if self.save_pred_img:
            self.save_pred_img_idx = self._save_batch_images(
                images=pred_img, 
                output_dir=f"{self.args.output_dir}/pred/worker_{self.worker_idx}_rank_{self.local_rank_idx}", 
                start_idx=self.save_pred_img_idx,
                dataset_idx=inputs["dataset_idx"],
            )
        if self.save_gt_img:
            self.save_gt_img_idx = self._save_batch_images(
                images=gt_img,
                output_dir=f"{self.args.output_dir}/raw/worker_{self.worker_idx}_rank_{self.local_rank_idx}",
                start_idx=self.save_gt_img_idx,
                dataset_idx=inputs["dataset_idx"],
            )
        if self.save_rec_img:
            self.save_rec_img_idx = self._save_batch_images(
                images=gt_decode_img,
                output_dir=f"{self.args.output_dir}/rec/worker_{self.worker_idx}_rank_{self.local_rank_idx}",
                start_idx=self.save_rec_img_idx,
                dataset_idx=inputs["dataset_idx"],
            )
        if self.save_gen_comparison:
            wm_in_imgs = inputs["observation.images.image0_history"]
            wm_in_imgs = torch.cat((wm_in_imgs, gt_img.unsqueeze(1), gt_decode_img.unsqueeze(1)), dim=1)
            self.save_wm_in_img_idx = self._save_comparison_images(
                images=wm_in_imgs,
                output_dir=f"{self.args.output_dir}/wm_comparison/worker_{self.worker_idx}_rank_{self.local_rank_idx}",
                start_idx=self.save_wm_in_img_idx,
            )
        return (
            torch.tensor(0, device=inputs["action"].device),
            torch.tensor(0, device=inputs["action"].device),
            torch.tensor(0, device=inputs["action"].device),
            torch.tensor(0, device=inputs["action"].device),
            torch.tensor(0, device=inputs["action"].device),
            torch.tensor(0, device=inputs["action"].device),
            torch.tensor(0, device=inputs["action"].device),
            torch.tensor(0, device=inputs["action"].device),
        )

    def _prediction_step_directly(self, inputs):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                action_pred = self.policy.select_action(inputs)

        loss = F.mse_loss(inputs["action"], action_pred, reduction="none")

        if "action_is_pad" in inputs:
            action_is_pad = ~inputs["action_is_pad"].unsqueeze(-1).repeat(1, 1, action_pred.shape[-1])
            loss = loss * action_is_pad

        loss = loss.mean()
        action_pred = nested_detach(action_pred)

        pad_id = torch.tensor(-100, device=action_pred.device)
        action_with_pad = torch.where(action_is_pad, inputs["action"], pad_id)

        return (
            loss, action_pred, action_with_pad, 
            torch.tensor(0, device=action_pred.device), 
            torch.tensor(0, device=action_pred.device), 
            torch.tensor(0, device=action_pred.device), 
            torch.tensor(0, device=action_pred.device), 
            torch.tensor(0, device=action_pred.device), 
        )

    def _prediction_step_with_world_model(self, inputs):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                # prepare world model input
                target = {}
                for key, value in inputs.items():
                    if key.startswith("observation.image") and "history" in key:
                        assert len(value.shape) == 5, "The observation image input should be 5 dims: (B T C H W)"
                        inputs[key] = value[:, :self.cur_n_obs_img_steps]
                        if key in self.pred_img_keys:
                            target[key.replace('history', 'pred')] = value[:, self.cur_n_obs_img_steps:self.cur_n_obs_img_steps + self.cur_n_pred_img_steps]

                action_pred, pred_imgs, logits, pred_wm_indices = self.policy.select_action_with_world_model(
                    inputs, 
                    predict_action_only=False,
                    top_k=900,
                    top_p=0.95,
                    num_samples=1,
                    rng=torch.Generator(device=inputs["action"].device),
                )

        ref = self.policy.unnormalize_outputs({"action": inputs["action"][..., :7]})["action"]
        loss_action = F.mse_loss(action_pred, ref, reduction="mean")

        # TODO: only support one image key for now
        gt_img = target[self.pred_img_keys[0].replace('history', 'pred')]
        gt_wm_indices = self.policy.vae.img_to_idxBl(gt_img.reshape(-1, *gt_img.shape[-3:]))
        gt_decode_img = self.policy.vae.idxBl_to_img(gt_wm_indices, same_shape=False, last_one=True)
        # pred_imgs = gt_decode_img
        gt_wm_indices = torch.cat(gt_wm_indices, dim=1)
        gen_token_len = logits.shape[1]
        gt_wm_indices = gt_wm_indices[:, :gen_token_len]
        loss_pred_img = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt_wm_indices.reshape(-1), reduction="mean")

        # t1 = ((pred_imgs + 1) / 2).clamp(0, 1)
        # t2 = ((gt_img.squeeze(1) + 1) / 2).clamp(0, 1)
        l2_loss_imgs = torch.tensor(0, device=gt_wm_indices.device)
        # F.mse_loss(t1, t2, reduction="mean")

        if self.save_pred_img:
            self.save_pred_img_idx = self._save_batch_images(
                images=pred_imgs, 
                output_dir=f"{self.args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}/pred/worker_{self.worker_idx}_rank_{self.local_rank_idx}", 
                start_idx=self.save_pred_img_idx, 
                dataset_idx=inputs["dataset_idx"], 
            )
        if self.save_gt_img:
            self.save_gt_img_idx = self._save_batch_images(
                images=gt_img.squeeze(1),
                output_dir=f"{self.args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}/gt/worker_{self.worker_idx}_rank_{self.local_rank_idx}",
                start_idx=self.save_gt_img_idx, 
                dataset_idx=inputs["dataset_idx"], 
            )
        if self.save_rec_img:
            self.save_rec_img_idx = self._save_batch_images(
                images=gt_decode_img,
                output_dir=f"{self.args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}/rec/worker_{self.worker_idx}_rank_{self.local_rank_idx}",
                start_idx=self.save_rec_img_idx, 
                dataset_idx=inputs["dataset_idx"], 
            )
        if self.save_gen_comparison:
            wm_in_imgs = inputs["observation.images.image0_history"]
            wm_in_imgs = torch.cat((wm_in_imgs, gt_img, pred_imgs.unsqueeze(1)), dim=1)
            self.save_wm_in_img_idx = self._save_comparison_images(
                images=wm_in_imgs,
                output_dir=f"{self.args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}/wm_comparison/worker_{self.worker_idx}_rank_{self.local_rank_idx}",
                start_idx=self.save_wm_in_img_idx,
            )

        if self.eval_if:
            if_scores = self._eval_if(inputs["task"], inputs["observation.images.image0_history"], pred_imgs, gt_img.squeeze(1), inputs["dataset_idx"])
            self.save_if_scores(if_scores, f"{self.args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}/eval_mllm_outputs/worker_{self.worker_idx}_rank_{self.local_rank_idx}")

        if "action_is_pad" in inputs:
            action_is_pad = ~inputs["action_is_pad"].unsqueeze(-1).repeat(1, 1, action_pred.shape[-1])
            loss_action = loss_action * action_is_pad

        loss_action = loss_action.mean()
        action_pred = nested_detach(action_pred)

        pad_id = torch.tensor(-100, device=action_pred.device)
        action_with_pad = torch.where(action_is_pad[:, :, :7], ref, pad_id)

        loss = loss_action + loss_pred_img * self.args.gen_out_loss_ratio

        return (
            loss, loss_action, loss_pred_img, 
            action_pred, action_with_pad, 
            pred_wm_indices, gt_wm_indices, 
            l2_loss_imgs
        )

    def save_if_scores(self, if_scores, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(if_scores)):
            if_scores[i] = if_scores[i].replace("\n", " ")

        with open(os.path.join(output_dir, f"eval_mllm_outputs.txt"), "a") as f:
            f.write("\n".join(if_scores))

    def _eval_if(self, task, wm_in_imgs, pred_imgs, gt_img, dataset_idx):
        base64_wm_imgs = []
        for batch in wm_in_imgs:
            batch_base64_wm_imgs = []
            for img in batch:
                batch_base64_wm_imgs.append(self._tensor_to_base64(img))
            base64_wm_imgs.append(batch_base64_wm_imgs)

        base64_pred_imgs = []
        for img in pred_imgs:
            base64_pred_imgs.append(self._tensor_to_base64(img))

        base64_ref_imgs = []
        for img in gt_img:
            base64_ref_imgs.append(self._tensor_to_base64(img))

        scores = []

        for sample in zip(base64_wm_imgs, base64_pred_imgs, task, dataset_idx, base64_ref_imgs):
            hist_imgs = sample[0]
            pred_img = sample[1]
            lang = sample[2]
            idx = sample[3]
            ref_img = sample[4]

            response = None
            max_retries = 10

            import time
            for _ in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.eval_mllm_model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": 
                                [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{hist_imgs[0]}"
                                        }
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{hist_imgs[1]}"
                                        }
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{hist_imgs[2]}"
                                        }
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{hist_imgs[3]}"
                                        }
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{pred_img}"
                                        }
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{ref_img}"
                                        }
                                    },
                                    {
                                        "type": "text", 
                                        "text": self.build_prompt(lang),
                                    },
                                ]
                            }
                        ]
                    )
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(0.3)

            if response is None:
                scores.append(f"{idx}: 0 0")
            else:
                content = response.choices[0].message.content.strip()
                scores.append(f"{idx}: {content}")

        return scores

    def build_prompt(self, task_name: str) -> str:
        prompt = f"""
            You are a visual sequence evaluation model for robot activity prediction.

            You will be given the following inputs:

            - A task instruction: "{task_name}"
            - Four historical images (Image 1 to Image 4), representing a temporal sequence of the robot performing the task
            - One generated image (Image 5), which is a predicted next step in the sequence
            - One reference image (Image 5_GT), which is the ground-truth next image for comparison

            Evaluate Image 5 based on the following three criteria:

            1. **Scene Consistency Score (0 or 1)**:  
                Does Image 5 maintain spatial and environmental consistency with Images 1–4? This includes:
                - Background layout and geometry should remain stable across the sequence
                - Lighting, colors, and textures must remain coherent
                - If Image 5 is **blurry, low-resolution, or lacks structural detail**, assign a score of 0

            2. **Object Consistency Score (0 or 1)**:  
                Are key objects consistent in **presence, identity, appearance, and position** compared with Images 1–4?  
                - Blurry, missing, deformed, or hallucinated objects should lead to a score of 0
                - The robot itself should also be treated as a key object

            3. **Task Progress Following Score (0 or 1)**:  
                Based on the task instruction and the visual progression from Images 1–4, does Image 5 show a plausible and coherent next action?  
                Additionally, compare Image 5 to Image 5_GT:
                - If Image 5 significantly deviates from Image 5_GT in a way that hinders task progression, score 0

            Your evaluation should be **STRICT**.

            Return **exactly three numbers (0 or 1)** separated by spaces, followed by a **brief explanation**.

            Output format (first line must be just the 3 numbers):

            1 0 1, Reason: Image 5 maintains background structure, but the object being manipulated is blurred and appears in a different shape. However, the robot continues a plausible motion toward the goal based on the task instruction.

            Important: First return three numbers, separated by spaces, then a newline, including any explanations and inferences, but without any additional text.
        """.strip()

        return prompt

    def _tensor_to_base64(self, img_tensor):
        assert img_tensor.ndim == 3 and img_tensor.shape[0] == 3

        if img_tensor.min() < 0:
            img_tensor = (img_tensor + 1) / 2

        img_uint8 = (img_tensor.clamp(0, 1) * 255).byte()
        img = transforms.ToPILImage()(img_uint8.cpu())
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return img_base64

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.accelerator.unwrap_model(self.model)._save_pretrained(Path(output_dir))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        # if multiple adapters exist, they get saved in sub directories
        adapter_subdirs = (
            [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]
            if os.path.isdir(resume_from_checkpoint)
            else []
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            weights_only_kwarg = {"weights_only": True}
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
                        )
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    model = PreTrainedPolicy._load_as_safetensor(model, safe_weights_file, "cpu", False)
                    logger.info(f"\033[31mLoading model from {safe_weights_file} complete !!\033[0m")
                else:
                    raise NotImplementedError("Not implemented")
                # release memory
                # del state_dict
                # self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif _is_peft_model(model):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            # TODO: in the future support only specific min PEFT versions
            if (hasattr(model, "active_adapter") or hasattr(model, "active_adapters")) and hasattr(
                model, "load_adapter"
            ):
                if os.path.exists(resume_from_checkpoint):
                    # For BC for older PEFT versions
                    if hasattr(model, "active_adapters"):
                        active_adapters = model.active_adapters
                        if len(active_adapters) > 1:
                            logger.warning("Multiple active adapters detected will only consider the first adapter")
                        active_adapter = active_adapters[0]
                    else:
                        active_adapter = model.active_adapter

                    if adapter_subdirs:
                        for subdir_name in adapter_subdirs:
                            peft_id = os.path.join(resume_from_checkpoint, subdir_name)
                            model.load_adapter(peft_id, subdir_name, is_trainable=(subdir_name == active_adapter))
                        model.set_adapter(active_adapter)
                    else:
                        model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    def _get_train_sampler(self, train_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]:
        return LargeScaleWeightedRandomSampler(self.training_ds_sample_weights, len(train_dataset))

    def _get_eval_sampler(self, eval_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]:
        return LargeScaleWeightedRandomSampler(self.eval_ds_sample_weights, len(eval_dataset))

    def _save_batch_images(self, images, output_dir, start_idx=0, dataset_idx=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # images = torch.clamp(images, -1, 1)
        transform = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        images = transform(images)
        images = images.cpu().numpy()
        images = (images * 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f"ds-{dataset_idx[i].item()}_idx-{i + start_idx}.png")
            imageio.imwrite(image_path, image)
        return start_idx + len(images)
    
    def _save_comparison_images(self, images, output_dir, start_idx=0):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        batch_size, n_imgs, C, H, W = images.shape
        images = images.reshape(batch_size * n_imgs, C, H, W)
        save_image(images, f"{output_dir}/idx-{start_idx}.png", nrow=n_imgs, normalize=True, value_range=(-1, 1))
        start_idx += 1
        return start_idx


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        action_losses (`np.ndarray`, *optional*): Action loss values computed during evaluation.
        wm_losses (`np.ndarray`, *optional*): World model loss values computed during evaluation.
        pred_wm_indices (`np.ndarray`, *optional*): Predicted world model indices.
        gt_wm_indices (`np.ndarray`, *optional*): Ground truth world model indices.
        inputs (`np.ndarray`, *optional*): Input data passed to the model.
        losses (`np.ndarray`, *optional*): Loss values computed during evaluation.
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        action_losses: Union[np.ndarray, Tuple[np.ndarray]] = None,
        wm_losses: Union[np.ndarray, Tuple[np.ndarray]] = None,
        pred_wm_indices: Union[np.ndarray, Tuple[np.ndarray]] = None,
        gt_wm_indices: Union[np.ndarray, Tuple[np.ndarray]] = None,
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        losses: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        l2_loss_imgs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.action_losses = action_losses
        self.wm_losses = wm_losses
        self.pred_wm_indices = pred_wm_indices
        self.gt_wm_indices = gt_wm_indices
        self.inputs = inputs
        self.losses = losses
        self.l2_loss_imgs = l2_loss_imgs

        self.elements = (self.predictions, self.label_ids)
        if self.action_losses is not None:
            self.elements += (self.action_losses,)
        if self.wm_losses is not None:
            self.elements += (self.wm_losses,)
        if self.pred_wm_indices is not None:
            self.elements += (self.pred_wm_indices,)
        if self.gt_wm_indices is not None:
            self.elements += (self.gt_wm_indices,)
        if self.inputs is not None:
            self.elements += (self.inputs,)
        if self.losses is not None:
            self.elements += (self.losses,)
        if self.l2_loss_imgs is not None:
            self.elements += (self.l2_loss_imgs,)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.elements):
            raise IndexError("tuple index out of range")
        return self.elements[idx]