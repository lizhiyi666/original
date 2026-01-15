from pathlib import Path
from omegaconf import DictConfig

from datamodule import DataModule
from add_thin.diffusion.model import AddThin
from add_thin.backbones.classifier import PointClassifier
from add_thin.distributions.intensities import MixtureIntensity
from tasks import DensityEstimation
from discrete_diffusion.diffusion_transformer import DiffusionTransformer

def instantiate_datamodule(config: DictConfig):
    return DataModule(
        config.root,
        config.name,
        batch_size=config.batch_size,
    )

def instantiate_model(config: DictConfig, datamodule) -> AddThin:
    classifier = PointClassifier(
        hidden_dims=config.temporal_hidden_dims,
        layer=config.temporal_classifier_layer,
    )
    intensity = MixtureIntensity(
        n_components=config.temporal_mix_components,
        embedding_size=config.temporal_hidden_dims,
        distribution="normal",
        time_segments=config.temporal_time_segments,
    )

    tpp_model = AddThin(
        classifier_model=classifier,
        intensity_model=intensity,
        max_time=datamodule.train_data.tmax.item(),
        steps=config.temporal_steps,
        hidden_dims=config.temporal_hidden_dims,
        emb_dim=config.temporal_hidden_dims,
        encoder_layer=config.temporal_encoder_layer,
        n_max=datamodule.n_max,
        kernel_size=config.temporal_kernel_size,
        num_condition_types=config.num_condition_types,
        time_segments=config.temporal_time_segments,
    )

    # [新增] 从配置中读取约束投影参数
    use_constraint_projection = getattr(config, 'use_constraint_projection', False)
    projection_tau = getattr(config, 'projection_tau', 0.0)
    projection_lambda = getattr(config, 'projection_lambda', 1.0)
    projection_alm_iters = getattr(config, 'projection_alm_iters', 10)  # 默认 10（论文建议）
    projection_eta = getattr(config, 'projection_eta', 0.2)  # [新增] 学习率 η，默认 0.2
    projection_mu = getattr(config, 'projection_mu', 1.0)  # [新增] 惩罚权重 μ，默认 1.0
    projection_frequency = getattr(config, 'projection_frequency', 10)  # [新增] 投影频率
    use_gumbel_softmax = getattr(config, "use_gumbel_softmax", True)
    gumbel_temperature = getattr(config, "gumbel_temperature", 1.0)
    projection_last_k_steps = getattr(config, "projection_last_k_steps", 60)

    discrete_diffusion =  DiffusionTransformer(
        diffusion_step=config.spatial_hidden_dims,
        alpha_init_type='alpha1',
        type_classes=datamodule.num_category,
        poi_classes=datamodule.num_poi,
        num_condition_types=config.num_condition_types,
        use_constraint_projection=use_constraint_projection,  # [新增]
        projection_tau=projection_tau,  # [新增]
        projection_lambda=projection_lambda,  # [新增]
        projection_alm_iters=projection_alm_iters,  # [新增]
        projection_eta=projection_eta,  # [新增] 学习率 η
        projection_mu=projection_mu,  # [新增] 惩罚权重 μ
        projection_frequency=projection_frequency,  # [新增]
        # [新增] 传入 Gumbel-Softmax 超参
        use_gumbel_softmax=use_gumbel_softmax,
        gumbel_temperature=gumbel_temperature,
        projection_last_k_steps=projection_last_k_steps,
    )
    return tpp_model, discrete_diffusion



def instantiate_task(config: DictConfig, tpp_model, discrete_diffusion, datamodule=None): # [修改] 增加 datamodule 参数
    # 尝试从 datamodule 获取 svd_components
    svd_components = getattr(datamodule, 'svd_components', None)
    
    # 获取 loss weight 配置 (需要在 config/train.yaml 中添加 task.po_loss_weight)
    po_loss_weight = getattr(config, 'po_loss_weight', 0.0) 

    return DensityEstimation(
        tpp_model,
        discrete_diffusion,
        config.learning_rate1,
        config.learning_rate2,
        config.weight_decay1,
        config.weight_decay2,
        svd_components=svd_components, # [新增]
        po_loss_weight=po_loss_weight  # [新增]
    )