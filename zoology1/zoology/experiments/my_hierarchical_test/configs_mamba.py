import uuid
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig

# 1. 实验设置
sweep_id = uuid.uuid4().hex[:6]
sweep_name = f"hierarchical-mamba-test-{sweep_id}"
VOCAB_SIZE = 8_192
L_MAX = 64
D_MODEL = 64
N_LAYERS = 2
LEARNING_RATE = 1e-3

# 2. 数据配置
data = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L_MAX, num_kv_pairs=16, num_examples=10000)],
    test_configs=[MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=L_MAX, num_kv_pairs=16, num_examples=1000)],
    batch_size=64,
    cache_dir="/workspace/zoology/cache"
)

# 3. 模型配置
models = []
model_factory_kwargs = {
    "state_mixer": ModuleConfig(name="zoology.mixers.mlp.GLU", kwargs={"hidden_mult": 4}),
    "vocab_size": VOCAB_SIZE,
    "max_position_embeddings": L_MAX,
    "d_model": D_MODEL,
    "n_layers": N_LAYERS,
}

# Mamba
base_mamba_config = ModuleConfig(
    name="mamba_ssm.Mamba",
    kwargs={"d_model": D_MODEL, "d_state": 16, "d_conv": 4, "expand": 2},
)

#循环创建您的分层模型
for levels in [1, 2, 3]:
    models.append(ModelConfig(
        name=f"Hierarchical-Mamba-L{levels}-d{D_MODEL}",
        block_type="MambaBlock",
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hierarchical.HierarchicalMixer",
            kwargs={"levels": levels, "mixer_config": base_mamba_config, "l_max": L_MAX}
        ),
        **model_factory_kwargs
    ))

# 基线: 标准注意力
models.append(ModelConfig(
    name=f"Baseline-Mamba-d{D_MODEL}",
    sequence_mixer=base_mamba_config,
    **model_factory_kwargs
))

# 4. 创建最终的训练配置列表
configs = []
for model in models:
    config = TrainConfig(
        model=model,
        data=data,
        learning_rate=LEARNING_RATE,
        max_epochs=64,
        run_id=f"{model.name}-lr{LEARNING_RATE:.1e}",
        logger=LoggerConfig(project_name="hierarchical_test", entity="xz3954-new-york-university"),
        sweep_id=sweep_name,
    )
    configs.append(config) 