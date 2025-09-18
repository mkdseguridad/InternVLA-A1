import os
import logging
import argparse
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import time
import transformers
from transformers import set_seed, HfArgumentParser

from src.models.configuration_mwm import MWMConfig
from src.policies.mwm_policy import MWMPolicy
from src.utils.utils import (
    load_ckpt,
    clean_overrides,
    save_training_args, 
    set_policy_config,
    get_second_last_checkpoint,
)
from src.processors.data_processors.data_config import create_data_config
from src.processors.data_processors.data_loader import create_data, CollateFn
from src.processors.train_processors.compute_metrics import compute_metrics
from src.processors.train_processors.policy_trainer import PolicyTrainer, PI0TrainingArguments
from src.processors.train_processors.optimizer_scheduler import create_optimizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger()


def main(args, overrides):
    #########################################################
    # Set the policy config and training config
    #########################################################
    config = OmegaConf.load(Path(args.config_file))
    override_cfg = OmegaConf.from_dotlist(clean_overrides(overrides))
    config = OmegaConf.merge(config, override_cfg)
 
    policy_config = MWMConfig.from_pretrained(f"{config.policy.policy_config}")
    policy_config = set_policy_config(policy_config, config.policy)

    parser_training_args = HfArgumentParser((PI0TrainingArguments))
    training_args = OmegaConf.to_container(config.exp.training_args, resolve=True)
    training_args = parser_training_args.parse_dict(training_args)[0]
    # set output dir
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d-%H-%M')
    training_args.output_dir = f"outputs/{formatted_time}/{training_args.run_name}"
    config.exp.training_args.output_dir = training_args.output_dir

    #########################################################
    # Save training args
    #########################################################
    worker_idx = int(os.environ.get("MLP_ROLE_INDEX", 0))
    local_rank_idx = int(os.environ.get('LOCAL_RANK', -1))
    if worker_idx == 0 and local_rank_idx in [-1, 0]:
        save_training_args(training_args, policy_config, config)
        print(f"saved training args on worker {worker_idx}, local rank {local_rank_idx}") 

    #########################################################
    # Log on each process summary
    #########################################################
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)
    logger.info(f"Training config: {args}")
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    #########################################################
    # Create dataset
    #########################################################
    start_time = time.time()
    data_config = create_data_config(config.dataset, policy_config, config.exp)
    end_time = time.time()
    print(f"create_data_config time cost: {(end_time - start_time)/60.0} min")
    start_time = time.time()
    (
        training_dataset, 
        eval_dataset, 
        image_transforms, 
        training_ds_sample_weights, 
        eval_ds_sample_weights,
        cur_n_obs_img_steps, 
        cur_n_pred_img_steps
    ) = create_data(
        policy_config=policy_config, 
        dataset_config=data_config, 
        training_args=training_args, 
        stage=config.exp.stage,
        max_eval_samples=config.exp.max_eval_samples,
    )
    end_time = time.time()
    print(f"create_data time cost: {(end_time - start_time)/60.0} min")

    logger.info(f"Training dataset:\n{training_dataset}")
    logger.info(f"len(training_dataset): {len(training_dataset)}")
    if eval_dataset is not None:
        logger.info(f"len(eval_dataset): {len(eval_dataset)}")

    #########################################################
    # Create model
    #########################################################
    logger.info("Creating model")
    kwargs = {"config": policy_config}

    # if "stage3" in config.exp.stage:
    #     from lerobot.common.datasets.utils import dataset_to_policy_features
    #     from lerobot.configs.types import FeatureType
    #     features = dataset_to_policy_features(training_dataset.datasets[0]._dataset.meta.features)
    #     kwargs["output_features"] = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    #     kwargs["input_features"] = {key: ft for key, ft in features.items() if key not in kwargs["output_features"]}
    #     kwargs["dataset_stats"] = training_dataset.datasets[0]._dataset.meta.stats

    kwargs["pretrained_name_or_path"] = policy_config.pretrained_path
    kwargs["training_args"] = training_args
    if policy_config.pretrained_path and not args.debug:
        policy = MWMPolicy.from_pretrained(**kwargs)
        policy = load_ckpt(policy, config)
    else:
        policy = MWMPolicy(**kwargs)

    optimizer = create_optimizer(policy, training_args)

    #########################################################
    # Resume from checkpoint
    #########################################################   
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_second_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    #########################################################
    # Create trainer
    #########################################################
    trainer = PolicyTrainer(
        policy=policy,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None),
        data_collator=CollateFn(policy_config.max_state_dim, policy_config.max_action_dim),
        image_transforms=image_transforms,
        compute_metrics=compute_metrics,
        use_world_model=policy_config.use_world_model,
        cur_n_obs_img_steps=cur_n_obs_img_steps,
        cur_n_pred_img_steps=cur_n_pred_img_steps,
        training_ds_sample_weights=training_ds_sample_weights,
        eval_ds_sample_weights=eval_ds_sample_weights,
        save_pred_img=config.exp.save_pred_img,
        save_gt_img=config.exp.save_gt_img,
        save_rec_img=config.exp.save_rec_img,
        save_wm_comparison=config.exp.save_wm_comparison,
        max_eval_samples=config.exp.max_eval_samples,
        eval_url=config.exp.eval_url,
        eval_mllm_model_name=config.exp.eval_mllm_model_name,
    )

    #########################################################   
    # Training
    #########################################################
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        print(f"Resume training from checkpoint: {checkpoint}")

        if config.exp.eval_first:
            trainer.evaluate()
            return None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument('--debug', action='store_true', help='to enable debug mode')
    args, unknown = parser.parse_known_args()
    main(args, overrides=unknown)