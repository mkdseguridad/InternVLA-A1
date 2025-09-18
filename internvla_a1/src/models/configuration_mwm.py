# coding=utf-8
# Copyright 2024 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING, AutoConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from src.models.internvl.configuration_internvl_chat import InternVLChatConfig

logger = logging.get_logger(__name__)


class MWMConfig(PretrainedConfig):
    model_type = "mwm"
    sub_configs = {"und_expert_config": AutoConfig, "gen_expert_config": AutoConfig, "act_expert_config": AutoConfig}
    has_no_defaults_at_init = True

    def __init__(
        self,
        und_expert_config=None,
        gen_expert_config=None,
        act_expert_config=None,
        proj_width=1024,
        chunk_size=50,
        max_action_dim=32,
        max_state_dim=32,
        tokenizer_max_length=48,
        use_cache=True,
        use_world_model=True,
        attention_implementation="eager",
        resize_imgs_with_padding="(224, 224)",
        image_tokenizer_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # TODO: make it configurable
        self.use_world_model = True
        self.is_encoder_decoder = False

        self.proj_width = proj_width
        self.chunk_size = chunk_size

        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim

        self.tokenizer_max_length = tokenizer_max_length
        self.use_cache = use_cache
        self.attention_implementation = attention_implementation
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.image_tokenizer_path = image_tokenizer_path

        self.und_expert_config = und_expert_config
        if isinstance(self.und_expert_config, dict):
            self.und_expert_config = InternVLChatConfig(**und_expert_config)
        elif und_expert_config is None:
            self.und_expert_config = InternVLChatConfig()

        if self.use_world_model:    
            self.gen_expert_config = gen_expert_config
            if isinstance(self.gen_expert_config, dict):
                self.gen_expert_config = CONFIG_MAPPING[gen_expert_config["model_type"]](**gen_expert_config)
            elif gen_expert_config is None:
                self.gen_expert_config = CONFIG_MAPPING["qwen2"]()

        self.act_expert_config = act_expert_config
        if isinstance(self.act_expert_config, dict):
            self.act_expert_config = CONFIG_MAPPING["qwen2"](**act_expert_config)
        elif act_expert_config is None:
            self.act_expert_config = CONFIG_MAPPING["qwen2"]()        

    @property
    def ignore_index(self):
        warnings.warn(
            "The `ignore_index` attribute is deprecated and will be removed in v4.47.",
            FutureWarning,
        )
        return self._ignore_index

    @ignore_index.setter
    def ignore_index(self, value):
        self._ignore_index = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_ignore_index", None)
        return output
    
    def to_diff_dict(self):
        """Override to_diff_dict to avoid recursive_diff_dict issues with sub-configs"""
        try:
            return super().to_diff_dict()
        except KeyError as e:
            # Fallback to regular to_dict if diff calculation fails
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to compute diff dict due to {e}, falling back to full dict")
            return self.to_dict()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if pretrained_model_name_or_path.endswith(".json"):
            config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            config = super().from_pretrained(config_path, **kwargs)
        return config