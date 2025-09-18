# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
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

from typing import List, Optional, Union

import torch
import torch.version
from pytest import Cache
from torch import nn
from transformers import (
    Qwen2ForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

from src.models.internvl.modeling_internvl_chat import InternVLChatModel


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class InternVLWithExpertModel(PreTrainedModel):
    config_class = PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_flash_attn_2 = False
    _supports_sdpa = True

    def __init__(self, config: PretrainedConfig):
        super().__init__(config=config)
        self.config = config

        self.und_expert = InternVLChatModel(config=config.und_expert_config)
        self.act_expert = Qwen2ForCausalLM(config=config.act_expert_config)
        self.act_expert.model.embed_tokens = None                 # Remove unused embed_tokens
        self.act_expert.lm_head = None
        if hasattr(self.config, "gen_expert_config") and self.config.gen_expert_config is not None:
            self.gen_expert = Qwen2ForCausalLM(config=config.gen_expert_config)
            self.gen_expert.model.embed_tokens = None             # Remove unused embed_tokens
            self.gen_expert.lm_head = None

        self.is_causal = False

        self.num_att_heads = self.config.und_expert_config.llm_config.num_attention_heads
        self.num_key_value_heads = self.config.und_expert_config.llm_config.num_key_value_heads
        self.num_key_value_groups = self.num_att_heads // self.num_key_value_heads

        self.freeze_vision_encoder = True
        self.train_act_expert_only = False
        self.train_gen_expert_only = False

        self.module_to_bf16()

    def set_requires_grad(
        self,
        freeze_vision_encoder=False,
        train_act_expert_only=False,
        train_gen_expert_only=False,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_act_expert_only = train_act_expert_only
        self.train_gen_expert_only = train_gen_expert_only

        if freeze_vision_encoder:
            self.und_expert.vision_model.eval()
            for params in self.und_expert.vision_model.parameters():
                params.requires_grad = False

        if train_act_expert_only:
            self.und_expert.eval()
            for params in self.und_expert.parameters():
                params.requires_grad = False

        if hasattr(self.config, "gen_expert_config") and self.config.gen_expert_config is not None \
            and train_gen_expert_only:
            print("\033[93mTraining World Model Expert only\033[0m")
            self.und_expert.eval()
            self.act_expert.eval()
            for params in self.und_expert.parameters():
                params.requires_grad = False
            for params in self.act_expert.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.und_expert.vision_model.eval()

        if self.train_act_expert_only:
            self.und_expert.eval()

        if self.train_gen_expert_only:
            self.und_expert.eval()
            self.act_expert.eval()

    def module_to_bf16(self):
        self.und_expert = self.und_expert.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "gen_expert.model.layers",
            "act_expert.model.layers",
            "vision_model",
            "und_expert.mlp1",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        return self.und_expert.extract_feature(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.und_expert.language_model.model.embed_tokens(tokens)

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        cat_past_key_values: Optional[bool] = False,
        is_eval: Optional[bool] = False,
    ):
        if hasattr(self.config, "gen_expert_config") and self.config.gen_expert_config is not None:
            models = [self.und_expert.language_model.model, self.gen_expert.model, self.act_expert.model]
        else:
            models = [self.und_expert.language_model.model, self.act_expert.model]

        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.und_expert.config.llm_config.num_hidden_layers
        head_dim = self.und_expert.config.llm_config.hidden_size // self.und_expert.config.llm_config.num_attention_heads
        num_hidden_layers_und_expert = self.und_expert.config.llm_config.num_hidden_layers
        num_hidden_layers_gen_expert = self.gen_expert.config.num_hidden_layers
        num_hidden_layers_act_expert = self.act_expert.config.num_hidden_layers
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = models[i].layers[layer_idx]
                # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
                # hidden_states = hidden_states * normalizer
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=torch.bfloat16)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            # if use_cache and past_key_values is None:
            #     past_key_values = {}
            # import pdb;pdb.set_trace()
            # if use_cache:
            #     if fill_kv_cache:
            #         past_key_values[layer_idx] = {
            #             "key_states": key_states,
            #             "value_states": value_states,
            #         }
            #     else:
            #         # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
            #         # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
            #         # the max len, then we (for instance) double the cache size. This implementation already exists
            #         # in `transformers`. (molbap)
            #         key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
            #         value_states = torch.cat(
            #             [past_key_values[layer_idx]["value_states"], value_states], dim=1
            #         )
            #         # for world model, we need to store the key and value states
            #         if cat_past_key_values:
            #             past_key_values[layer_idx]["key_states"] = key_states
            #             past_key_values[layer_idx]["value_states"] = value_states

            if self.config.attention_implementation == "eager":
                att_output = self.eager_attention_forward(
                    attention_mask, batch_size, head_dim, query_states, key_states, value_states
                )
            elif self.config.attention_implementation == "sdpa":
                att_output = torch.nn.functional.scaled_dot_product_attention(
                    query=query_states.permute(0, 2, 1, 3),
                    key=key_states.permute(0, 2, 1, 3),
                    value=value_states.permute(0, 2, 1, 3),
                    attn_mask=attention_mask[:, None, :, :],
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=False,
                )
                att_output = att_output.permute(0, 2, 1, 3)
                att_output = att_output.reshape(batch_size, -1, self.num_key_value_heads * self.num_key_value_groups * head_dim)
            elif self.config.attention_implementation == "flex":
                raise NotImplementedError("Flex attention is not implemented (yet)")
            else:
                raise ValueError(f"Unsupported attention implementation: {self.config.attention_implementation}")
            # att_output = att_output.to(dtype=torch.bfloat16)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # TODO: first dropout (by default 0.0)

                    # first residual
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # TODO: second dropout (by default 0.0)

                    # second residual
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def forward_flow_matching(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        cat_past_key_values: Optional[bool] = False,
        is_eval: Optional[bool] = False,
    ):
        if hasattr(self.config, "gen_expert_config") and self.config.gen_expert_config is not None:
            models = [self.und_expert.language_model.model, self.gen_expert.model, self.act_expert.model]
        else:
            models = [self.und_expert.language_model.model, self.act_expert.model]

        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        # num_layers = self.und_expert.config.llm_config.num_hidden_layers
        # num_layers = self.act_expert.config.num_hidden_layers
        # head_dim = self.und_expert.config.llm_config.hidden_size // self.und_expert.config.llm_config.num_attention_heads
        head_dim = self.act_expert.config.hidden_size // self.act_expert.config.num_attention_heads
        num_hidden_layers_und_expert = self.und_expert.config.llm_config.num_hidden_layers
        num_hidden_layers_gen_expert = self.gen_expert.config.num_hidden_layers
        num_hidden_layers_act_expert = self.act_expert.config.num_hidden_layers

        for layer_idx in range(num_hidden_layers_und_expert):
            inputs_embeds_layer_idx = []
            if layer_idx < num_hidden_layers_und_expert - num_hidden_layers_act_expert:
                inputs_embeds_layer_idx.append(inputs_embeds[0])
                inputs_embeds_layer_idx.append(inputs_embeds[1])
                inputs_embeds_layer_idx.append(None)
            else:
                inputs_embeds_layer_idx.append(inputs_embeds[0])
                inputs_embeds_layer_idx.append(inputs_embeds[1])
                inputs_embeds_layer_idx.append(inputs_embeds[2])
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds_layer_idx):
                if hidden_states is None:
                    continue
                if i == len(inputs_embeds_layer_idx) - 1:
                    select_layer_idx = layer_idx - (num_hidden_layers_und_expert - num_hidden_layers_act_expert)
                else:
                    select_layer_idx = layer_idx
                layer = models[i].layers[select_layer_idx]
                # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
                # hidden_states = hidden_states * normalizer
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=torch.bfloat16)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            if layer_idx < num_hidden_layers_und_expert - num_hidden_layers_act_expert:
                query_states = apply_rope(query_states, position_ids[:, :query_states.shape[1]])
                key_states = apply_rope(key_states, position_ids[:, :query_states.shape[1]])
            else:
                query_states = apply_rope(query_states, position_ids)
                key_states = apply_rope(key_states, position_ids)

            # if use_cache and past_key_values is None:
            #     past_key_values = {}
            # # import pdb;pdb.set_trace()
            # if use_cache:
            #     if fill_kv_cache:
            #         past_key_values[layer_idx] = {
            #             "key_states": key_states,
            #             "value_states": value_states,
            #         }
            #     else:
            #         # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
            #         # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
            #         # the max len, then we (for instance) double the cache size. This implementation already exists
            #         # in `transformers`. (molbap)
            #         key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
            #         value_states = torch.cat(
            #             [past_key_values[layer_idx]["value_states"], value_states], dim=1
            #         )
            #         # for world model, we need to store the key and value states
            #         if cat_past_key_values:
            #             past_key_values[layer_idx]["key_states"] = key_states
            #             past_key_values[layer_idx]["value_states"] = value_states
            if layer_idx < num_hidden_layers_und_expert - num_hidden_layers_act_expert:
                attention_mask_layer_idx = attention_mask[:, :query_states.shape[1], :query_states.shape[1]]
            else:
                attention_mask_layer_idx = attention_mask
            if self.config.attention_implementation == "eager":
                att_output = self.eager_attention_forward(
                    attention_mask_layer_idx, batch_size, head_dim, query_states, key_states, value_states
                )
            elif self.config.attention_implementation == "sdpa":
                att_output = torch.nn.functional.scaled_dot_product_attention(
                    query=query_states.permute(0, 2, 1, 3),
                    key=key_states.permute(0, 2, 1, 3),
                    value=value_states.permute(0, 2, 1, 3),
                    attn_mask=attention_mask_layer_idx[:, None, :, :],
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=False,
                )
                att_output = att_output.permute(0, 2, 1, 3)
                att_output = att_output.reshape(batch_size, -1, self.num_key_value_heads * self.num_key_value_groups * head_dim)
            elif self.config.attention_implementation == "flex":
                raise NotImplementedError("Flex attention is not implemented (yet)")
            else:
                raise ValueError(f"Unsupported attention implementation: {self.config.attention_implementation}")
            # att_output = att_output.to(dtype=torch.bfloat16)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds_layer_idx):
                if hidden_states is None:
                    continue
                if i == len(inputs_embeds_layer_idx) - 1:
                    select_layer_idx = layer_idx - (num_hidden_layers_und_expert - num_hidden_layers_act_expert)
                else:
                    select_layer_idx = layer_idx
                layer = models[i].layers[select_layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # TODO: first dropout (by default 0.0)

                    # first residual
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # TODO: second dropout (by default 0.0)

                    # second residual
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end
                else:
                    outputs_embeds.append(None)

            if layer_idx < num_hidden_layers_und_expert - num_hidden_layers_act_expert:
                inputs_embeds[0] = outputs_embeds[0]
                inputs_embeds[1] = outputs_embeds[1]
            else:
                inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_att_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        if att_weights.dtype == torch.float32:
            big_neg = -2.3819763e38  # See gemma/modules.py
        elif att_weights.dtype == torch.bfloat16 or att_weights.dtype == torch.float16:
            big_neg = -1e9
        else:
            raise ValueError(f"Unsupported dtype: {att_weights.dtype}")

        masked_att_weights = att_weights.masked_fill(~attention_mask[:, None, :, :], big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
