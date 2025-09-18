from dataclasses import dataclass
from typing import List
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.internvl_with_expert import (
    InternVLWithExpertModel
)
from lerobot.common.policies.pi0.modeling_pi0 import (
    create_sinusoidal_pos_embedding,
    sample_beta,
    make_att_2d_masks,
)


class MWMFlowMatching(nn.Module):

    def __init__(self, config, vae=None, **kwargs):
        super().__init__()
        self.config = config

        self.internvl_with_expert = InternVLWithExpertModel(self.config)

        if self.config.use_world_model:
            self.vlm_expert_hidden_size = config.und_expert_config.hidden_size
            self.wm_expert_hidden_size = config.gen_expert_config.hidden_size

            # world model modules
            self.vocab_size = 64000
            self.vqvae_dim = 1024
            init_std = math.sqrt(1 / self.wm_expert_hidden_size / 3)

            self.with_extra_pred_tokens = self.config.with_extra_pred_tokens
            # if self.model_mode == "seer":
            if self.with_extra_pred_tokens:
                # TODO: Hack for 64 extra pred tokens
                self.extra_pred_tokens = nn.Parameter(
                    torch.zeros(1, 1, 64, self.wm_expert_hidden_size)
                )
                nn.init.trunc_normal_(self.extra_pred_tokens.data, mean=0, std=init_std)

            # spatial downsampling
            self.spatial_conv = SpatialDownsampling(
                self.wm_expert_hidden_size, 
                self.wm_expert_hidden_size, 
                config.gen_expert_config.spatial_conv_kernel_size, 
                config.gen_expert_config.spatial_conv_stride
            )

            # spatial upsampling (reverse of spatial downsampling)
            self.spatial_upconv = SpatialUpsampling(
                self.wm_expert_hidden_size, 
                self.wm_expert_hidden_size, 
                config.gen_expert_config.spatial_conv_kernel_size, 
                config.gen_expert_config.spatial_conv_stride
            )

            # wm_embedding
            self.wm_embeddings = nn.Embedding(self.vocab_size, self.wm_expert_hidden_size)
            self.wm_hist_pos_embs = nn.Parameter(torch.empty(1, 2048, self.wm_expert_hidden_size))
            nn.init.trunc_normal_(self.wm_hist_pos_embs.data, mean=0, std=init_std)

            # position embedding
            # (Optional) Classifier free guidance
            self.cfg_scale = 0.0
            if self.cfg_scale > 0:
                self.register_buffer('cfg_embedding', torch.empty(1, 1, self.vlm_expert_hidden_size))
                nn.init.trunc_normal_(self.cfg_embedding, mean=0, std=init_std)

            # projection
            self.wm_out_layer_norm = nn.LayerNorm(self.wm_expert_hidden_size)
            self.wm_out_proj = nn.Linear(self.config.proj_width, self.vocab_size)

        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        # replace the action_out_proj from a linear layer to a mlp
        # if not config.use_world_model:
        # self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        # else:
        self.action_out_proj = nn.Sequential(
            nn.Linear(self.config.proj_width, 4 * self.config.proj_width),
            nn.SiLU(),
            nn.Linear(4 * self.config.proj_width, self.config.max_action_dim),
        )

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        training_args = kwargs.get("training_args", None)

        self.set_requires_grad(training_args)

    def set_requires_grad(self, training_args=None):
        if training_args is None:
            self.freeze_vision_encoder = True
            self.train_act_expert_only = False
            self.train_gen_expert_only = False
            self.train_state_proj = True
        else:
            self.freeze_vision_encoder = training_args.freeze_vision_encoder
            self.train_act_expert_only = training_args.train_act_expert_only
            self.train_gen_expert_only = training_args.train_gen_expert_only
            self.train_state_proj = training_args.train_state_proj

        self.internvl_with_expert.set_requires_grad(
            freeze_vision_encoder=self.freeze_vision_encoder,
            train_act_expert_only=self.train_act_expert_only,
            train_gen_expert_only=self.train_gen_expert_only,
        )
        for params in self.state_proj.parameters():
            params.requires_grad = self.train_state_proj

        if self.train_gen_expert_only:
            freeze_modules = ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]
            for name, param in self.named_parameters():
                if any (x in name for x in freeze_modules):
                    param.requires_grad = False

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_und_inputs(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with internvit and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        lang_emb = self.internvl_with_expert.und_expert.language_model.get_input_embeddings()(lang_tokens).clone()

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img = img.to(dtype=torch.bfloat16)
            img_emb = self.internvl_with_expert.embed_image(img)
            # img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_act_inputs(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        # state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_gen_inputs(self, world_model_input_embs, world_model_input_masks):
        """
        Embeds the world model input embs by downsampling the spatial and temporal dimensions.

        Args:
            world_model_input_embs: (B, N_view, T, H, W, D)

        Returns:
            embs: (B, N_view * H*W//16, D)
            pad_masks: (B, N_view * H*W//16)
            att_masks: (B, N_view * H*W//16)
        """
        B, N_view, T, H, W, D = world_model_input_embs.shape
        # spatial downsampling 4x: -> B, N_view, T, H//4, W//4, D
        world_model_inputs = self.spatial_conv(world_model_input_embs)
        # temporal downsampling 3x: -> B, N_view, T/3, H*W/1/6, D
        # world_model_inputs = self.temporal_conv(world_model_input_embs)

        # B, N_view, T/3, H*W//16, D -> B, N_view, H*W//16, D
        # world_model_inputs = world_model_inputs.squeeze(-3)
        # flatten the N_view and T dimensions: -> B, N_view * H*W//16, D
        # world_model_inputs = world_model_inputs.flatten(start_dim=1, end_dim=2)
        world_model_inputs = world_model_inputs.reshape(B, -1, D)
        world_model_inputs += self.wm_hist_pos_embs[:, :world_model_inputs.shape[1]]

        embs = world_model_inputs
        # if self.model_mode == "seer":
        if self.with_extra_pred_tokens:
            pred_obs_tokens = self.extra_pred_tokens.expand(B, N_view, -1, -1)
            pred_obs_tokens = pred_obs_tokens.flatten(start_dim=1, end_dim=2)
            embs = torch.cat([embs, pred_obs_tokens], dim=1)

        # Create attention masks (full attention for world model tokens)
        att_masks = [0] * (embs.shape[1])
        att_masks = torch.tensor(att_masks, device=embs.device)
        att_masks = att_masks[None, :].expand(embs.shape[0], len(att_masks))

        # Create padding masks based on input masks
        seq_len = world_model_inputs.shape[1]
        pad_masks = world_model_input_masks[:, :, None].expand(
            B, N_view, seq_len // N_view
        )
        pad_masks = pad_masks.reshape(B, -1)
        #if self.model_mode == "seer":
        if self.with_extra_pred_tokens:
            pad_masks = torch.cat([
                pad_masks, torch.ones_like(pred_obs_tokens[:, :, 0], dtype=torch.bool)
            ], dim=1)

        return embs, pad_masks, att_masks

    def forward_with_world_model(
        self, 
        images: List[torch.Tensor], 
        img_masks: List[torch.Tensor], 
        lang_tokens: torch.Tensor, 
        lang_masks: torch.Tensor, 
        state: torch.Tensor, 
        world_model_input_embs: List[torch.Tensor], 
        world_model_input_masks: List[torch.Tensor],
        actions: torch.Tensor, 
        noise: torch.Tensor | None = None, 
        time: torch.Tensor | None = None,
    ) -> Tensor:
        if self.train_gen_expert_only:
            und_embs, und_pad_masks, und_att_masks = self.embed_und_inputs(
                images, img_masks, lang_tokens, lang_masks
            )
            gen_embs, gen_pad_masks, gen_att_masks = self.embed_gen_inputs(
                world_model_input_embs, world_model_input_masks
            )
            if self.cfg_scale > 0:
                batch_size = und_embs.shape[0]
                seq_len = und_embs.shape[1]
                mask_flags = (torch.rand(batch_size, device=und_embs.device) < self.cfg_scale).unsqueeze(1).unsqueeze(2)
                cfg_emb = self.cfg_embedding.expand(batch_size, seq_len, -1)
                und_embs = torch.where(mask_flags, cfg_emb, und_embs)

            pad_masks = torch.cat([und_pad_masks, gen_pad_masks], dim=1)
            att_masks = torch.cat([und_att_masks, gen_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            (_, gen_out, _), _ = self.internvl_with_expert.forward( # forward_flow_matching(   # forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[und_embs, gen_embs, None],
                use_cache=False,
                fill_kv_cache=False,
            )

            original_shape = world_model_input_embs.shape  # (B, N_view, T, H, W, D)
            upsampled_gen_out = self.upsample_gen_outputs(gen_out, original_shape)
            gen_logits = self.wm_out_proj(self.wm_out_layer_norm(upsampled_gen_out))

            return 0, gen_logits

        else:

            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)

            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            # compute the position and level embeddings for the world model output
            und_embs, und_pad_masks, und_att_masks = self.embed_und_inputs(
                images, img_masks, lang_tokens, lang_masks
            )
            gen_embs, gen_pad_masks, gen_att_masks = self.embed_gen_inputs(
                world_model_input_embs, world_model_input_masks
            )
            act_embs, act_pad_masks, act_att_masks = self.embed_act_inputs(
                state, x_t, time
            )

            # 2. prepare the mask and position ids
            pad_masks = torch.cat([und_pad_masks, gen_pad_masks, act_pad_masks], dim=1)
            att_masks = torch.cat([und_att_masks, gen_att_masks, act_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            # 3. compute the logits
            (_, gen_out, act_out), _ = self.internvl_with_expert.forward_flow_matching( # forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[und_embs, gen_embs, act_embs],
                use_cache=False,
                fill_kv_cache=False,
            )

            # 4. upsample the gen_out back to original spatiotemporal resolution and compute the logits
            original_shape = world_model_input_embs.shape  # (B, N_view, T, H, W, D)
            upsampled_gen_out = self.upsample_gen_outputs(gen_out, original_shape)
            gen_logits = self.wm_out_proj(self.wm_out_layer_norm(upsampled_gen_out))

            # 5. compute the loss for the action
            act_out = act_out[:, -self.config.chunk_size :]
            act_out = act_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(act_out)

            losses = F.mse_loss(u_t, v_t, reduction="none")

            return losses, gen_logits

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""

        # setup the target, if indices is not None, use the indices to get the target from the actions

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_und_inputs(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_act_inputs(
            state, x_t, time
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.internvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_und_inputs(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.internvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        return x_t

    def sample_actions_with_world_model(
        self, 
        images: List[torch.Tensor], 
        image_masks: List[torch.Tensor], 
        lang_tokens: torch.Tensor, 
        lang_masks: torch.Tensor, 
        state: torch.Tensor, 
        world_model_input_embs: torch.Tensor, 
        world_model_input_masks: torch.Tensor,
        predict_action_only: bool = True,
        noise = None,
    ) -> Tensor:
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        und_embs, und_pad_masks, und_att_masks = self.embed_und_inputs(
            images, image_masks, lang_tokens, lang_masks
        )
        gen_embs, gen_pad_masks, gen_att_masks = self.embed_gen_inputs(
            world_model_input_embs, world_model_input_masks
        )

        # 2. prepare the mask and position ids
        pad_masks = torch.cat([und_pad_masks, gen_pad_masks], dim=1)
        att_masks = torch.cat([und_att_masks, gen_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # 3. compute KV cache
        (_, gen_out, _), past_key_values = self.internvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[und_embs, gen_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        # 4. generate action
        num_hidden_layers_gen_expert = self.internvl_with_expert.gen_expert.config.num_hidden_layers
        num_hidden_layers_act_expert = self.internvl_with_expert.act_expert.config.num_hidden_layers
        selected_past_kv_layers = list(past_key_values.keys())[num_hidden_layers_gen_expert - num_hidden_layers_act_expert:num_hidden_layers_gen_expert]
        past_key_values = {idx:past_key_values[layer] for idx, layer in enumerate(selected_past_kv_layers)}
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        if predict_action_only:
            return x_t

        original_shape = world_model_input_embs.shape  # (B, N_view, T, H, W, D)
        upsampled_gen_out = self.upsample_gen_outputs(gen_out, original_shape)
        gen_logits = self.wm_out_proj(self.wm_out_layer_norm(upsampled_gen_out))
        gen_indices = gen_logits.argmax(dim=-1)

        return ActionOutput(actions=x_t, pred_img_indices=gen_indices)

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        act_embs, act_pad_masks, act_att_masks = self.embed_act_inputs(state, x_t, timestep)
        act_len = act_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].repeat(1, act_len, 1)

        act_att_2d_masks = make_att_2d_masks(act_pad_masks, act_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, act_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(act_pad_masks, dim=1) - 1

        inputs_embeds = [None, None, act_embs] if self.config.use_world_model else [None, act_embs]
        outputs_embeds, _ = self.internvl_with_expert.forward_flow_matching( # forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            is_eval=True,
        )
        suffix_out = outputs_embeds[-1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def upsample_gen_outputs(self, gen_out, original_shape):
        """
        Upsample the generated outputs back to original spatial resolution only.
        No temporal upsampling is performed.
        
        Args:
            gen_out: Generated outputs from the model, shape (B, seq_len, D)
            original_shape: Tuple of (B, N_view, T, H, W, D) - original input shape
        
        Returns:
            Upsampled outputs with shape (B, N_view, H, W, D) - spatially upsampled to original resolution
        """
        B, N_view, T, H, W, D = original_shape
        H_downsampled, W_downsampled = H // 4, W // 4

        # if self.model_mode == "seer": # seer mode
        if self.with_extra_pred_tokens: # seer mode
            gen_out = gen_out[:, -N_view*64:]
            # Reshape to spatial format
            gen_out = gen_out.reshape(B, N_view, H_downsampled, W_downsampled, D)
        else: # gr2 mode
            gen_out = gen_out.reshape(B, N_view, T, H_downsampled, W_downsampled, D)
            # Average over time dimension
            gen_out = gen_out.mean(dim=2)

        gen_out = self.spatial_upconv(gen_out)

        return gen_out

class SpatialDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=4, stride=4):
        super(SpatialDownsampling, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernal_size,
            stride=stride,
            padding=0
        )

    def forward(self, x):
        B, N_view, T, H, W, D = x.shape
        x = x.reshape(B * N_view * T, H, W, D)
        
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, N_view, T, -1, self.conv.out_channels)

        return x


class TemporalUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=4, output_padding=0):
        super(TemporalUpsampling, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=output_padding
        )

    def forward(self, x):
        # x shape: B, N_view, T_compressed, L, D
        B, N_view, T_compressed, L, D = x.shape
        x = x.reshape(B * N_view * L, T_compressed, D)
        x = x.permute(0, 2, 1)  # (B*N_view*L, D, T_compressed)
        x = self.conv_transpose(x)  # (B*N_view*L, D, T_original)
        x = x.permute(0, 2, 1)  # (B*N_view*L, T_original, D)
        T_original = x.shape[1]
        x = x.reshape(B, N_view, L, T_original, D)
        x = x.permute(0, 1, 3, 2, 4)  # (B, N_view, T_original, L, D)

        return x


class SpatialUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=4, output_padding=0):
        super(SpatialUpsampling, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=output_padding
        )

    def forward(self, x):
        # x shape: B, N_view, T, H_compressed, W_compressed, D
        B, N_view, H_compressed, W_compressed, D = x.shape
        x = x.reshape(B * N_view, H_compressed, W_compressed, D)
        
        x = x.permute(0, 3, 1, 2)  # (B*N_view, D, H_compressed, W_compressed)
        x = self.conv_transpose(x)  # (B*N_view, D, H_original, W_original)
        x = x.permute(0, 2, 3, 1)  # (B*N_view, H_original, W_original, D)
        
        H_original, W_original = x.shape[1], x.shape[2]
        x = x.reshape(B, N_view, H_original, W_original, D)

        return x


@dataclass
class ActionOutput:
    actions: torch.Tensor
    pred_img_indices: torch.Tensor | None = None