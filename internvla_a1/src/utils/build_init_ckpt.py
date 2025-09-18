from src.models.configuration_mwm import MWMConfig
from src.policies.mwm_policy import MWMPolicy

import safetensors


ckpt = safetensors.torch.load_file("/fs-computility/efm/shared/model_weights/InternVL3-1B/model.safetensors", device="cpu")


policy_config = MWMConfig.from_pretrained("./config/waic_vla.json")
policy = MWMPolicy(config=policy_config)


und_expert = policy.model.internvl_with_expert.und_expert
gen_expert = policy.model.internvl_with_expert.gen_expert
act_expert = policy.model.internvl_with_expert.act_expert


# # 获取 gen_expert 的 state_dict，这将作为我们填充权重的目标
small_ckpt_target = gen_expert.state_dict()

new_ckpt = {} # 这个字典可能您是想用于保存最终裁剪后的权重，这里先保留

print(f"开始拷贝/裁剪预训练权重到 gen_expert...")

# 遍历原始预训练模型的权重
for k, v in ckpt.items():
    # 过滤掉非 language_model 部分和 lm_head
    if not k.startswith("language_model") or "lm_head" in k:
        continue

    # 构造新专家中的对应key
    new_key_name = k.replace("language_model.", "")

    # 检查 new_key_name 是否存在于 small_ckpt_target 中
    # 有些层可能在小专家中不存在，或者命名方式略有不同
    if new_key_name not in small_ckpt_target:
        print(f"Warning: {new_key_name} not found in gen_expert's state_dict. Skipping.")
        continue

    # 获取小专家中对应参数的引用
    small_v_param = small_ckpt_target[new_key_name]

    # 确保维度一致性检查是在 tensor 上，而不是 parameter 对象上
    small_v_shape = small_v_param.shape
    v_shape = v.shape

    if small_v_shape == v_shape:
        # 维度完全一致，直接拷贝
        print(f"直接拷贝 {k} (shape: {v_shape})")
        small_v_param.copy_(v)
    else:
        # 维度不一致，进行裁剪
        print(f"裁剪 {k} 从 {v_shape} 到 {small_v_shape}")
        assert small_v_param.ndim == v.ndim, \
            f"维度数量不匹配：{new_key_name}, 原始:{v.ndim}, 目标:{small_v_param.ndim}"

        if small_v_param.ndim == 1:
            # 处理偏置 (bias) 和 LayerNorm 的 weight/bias (一维张量)
            assert small_v_shape[0] <= v_shape[0], \
                f"裁剪错误：目标维度 {small_v_shape[0]} 大于原始维度 {v_shape[0]} for {new_key_name}"
            small_v_param.copy_(v[:small_v_shape[0]])
        elif small_v_param.ndim == 2:
            # 处理线性层权重 (in_features, out_features) 或 (out_features, in_features)
            # 需要根据具体的层类型来判断是 (out, in) 还是 (in, out)
            # 对于 PyTorch 的 nn.Linear，权重是 (out_features, in_features)

            # 假设大部分权重是 (out_features, in_features)
            # 那么裁剪时需要裁剪 out_features 和 in_features
            target_out_features = small_v_shape[0]
            target_in_features = small_v_shape[1]

            original_out_features = v_shape[0]
            original_in_features = v_shape[1]

            assert target_out_features <= original_out_features and \
                   target_in_features <= original_in_features, \
                   f"裁剪错误：目标维度 {small_v_shape} 至少有一个大于原始维度 {v_shape} for {new_key_name}"

            # 执行裁剪
            # 这里是通用的裁剪逻辑，取左上角部分
            small_v_param.copy_(v[:target_out_features, :target_in_features])

            # 特殊处理 QKV 投影权重（如果您的模型是单一个大矩阵投影 QKV）
            # 例如 'self_attn.qkv_proj.weight'，通常是 (3*hidden_size, hidden_size)
            # 如果您的专家模型中 QKV 是分开的，则不需要特别处理，它会作为普通的线性层被裁剪
            # 如果是合在一起的，并且您希望裁剪后每个Q/K/V的头数或维度对应，需要更复杂的逻辑
            # 这里暂时假设是通用的线性层裁剪，即简单地裁剪到目标维度
        else:
            print(f"Warning: 无法处理 {new_key_name} 的维度数量 {v.ndim}，跳过拷贝。")
            continue
    new_ckpt[new_key_name] = small_v_param

print("权重拷贝/裁剪完成。")

abc1 = policy.model.internvl_with_expert.gen_expert.load_state_dict(new_ckpt.copy())
abc2 = policy.model.internvl_with_expert.act_expert.load_state_dict(new_ckpt.copy())

abc3 = policy.model.internvl_with_expert.und_expert.load_state_dict(ckpt)
safetensors.torch.save_model(policy, "./ckpt/init_model.safetensors")
print(1)





