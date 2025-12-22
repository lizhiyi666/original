import pytorch_lightning as pl
import torch
import torch.nn as nn
from pathlib import Path
import wandb
import os 

import numpy as np
import matplotlib.pyplot as plt



from datamodule import Batch
from add_thin.metrics import (
    MMD,
    lengths_distribution_wasserstein_distance,
)
from evaluations.statistical_metrics import Get_Statistical_Metrics
from partial_order_loss import PartialOrderLoss

class Tasks(pl.LightningModule):
    def __init__(
        self,
        tpp_model,
        discrete_diffusion,
        learning_rate1,
        learning_rate2,
        weight_decay1: float = 0.0,
        weight_decay2: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("tpp_model","discrete_diffusion"))

        self.weight_decay1 = weight_decay1
        self.weight_decay2 = weight_decay2
        self.learning_rate1 = learning_rate1
        self.learning_rate2 = learning_rate2
        self.tpp_model = tpp_model
        self.discrete_diffusion = discrete_diffusion
        self.classification_loss_func = nn.BCEWithLogitsLoss(reduction="none")

    @property
    def automatic_optimization(self) -> bool:
        return False     

    def classification_loss(self, x_n_int_x_0, x_n: Batch):
        """
        Compute BCE loss for the classification task.
        """
        x_n_int_x_0 = x_n_int_x_0.flatten()[x_n.mask.flatten()]
        target = x_n.kept.flatten()[x_n.mask.flatten()]
        loss = self.classification_loss_func(x_n_int_x_0, target.float())
        loss = (loss).sum() / len(x_n)
        return loss

    def intensity_loss(self, log_prob_x_0):
        """
        Compute the average (over batch) negative log-likelihood of the event sequences.
        """
        return -log_prob_x_0.mean()

    def get_loss(self, log_prob_x_0, x_n_int_x_0, x_n):
        """
        Compute the loss for the classification and intensity.
        """
        intensity = self.intensity_loss(log_prob_x_0) / self.tpp_model.n_max

        classification = (
            self.classification_loss(x_n_int_x_0, x_n) / self.tpp_model.n_max
        )
        loss = classification + intensity
        return loss, classification, intensity

    def step(self, batch, name):
        """
        Apply tpp_model to batch and compute loss.
        """
        # Forward pass
        x_n_int_x_0, log_prob_x_0, x_n = self.tpp_model.forward(batch)

        # Spatial loss
        spatial_loss = self.discrete_diffusion.training_losses(batch).mean()


        # Compute loss
        temporal_loss, classification, intensity = self.get_loss(
            log_prob_x_0, x_n_int_x_0, x_n
        )

        total_loss = spatial_loss + temporal_loss
        # Log loss
        self.log(
            f"{name}/total_loss",
            total_loss.detach().item(),
            batch_size=batch.batch_size,
        )
        self.log(
            f"{name}/spatial_loss",
            spatial_loss.detach().item(),
            batch_size=batch.batch_size,
        )

        self.log(
            f"{name}/temporal_loss",
            temporal_loss.detach().item(),
            batch_size=batch.batch_size,
        )

        self.log(
            f"{name}/log-likelihood",
            intensity.detach().item(),
            batch_size=batch.batch_size,
        )
        if classification is not None:
            self.log(
                f"{name}/BCE",
                classification.detach().item(),
                batch_size=batch.batch_size,
            )
        return temporal_loss,spatial_loss,total_loss

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.tpp_model.parameters(), lr=self.learning_rate1, weight_decay=self.weight_decay1,)
        
        optimizer2 = torch.optim.AdamW(self.discrete_diffusion.parameters(), lr=self.learning_rate2, weight_decay=self.weight_decay2)

        lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer1, factor=0.95, patience=1000, 
            #verbose=True
        )

        lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2, factor=0.95, patience=1000, 
            #verbose=True
        )
        return ({"optimizer": optimizer1,"lr_scheduler": {"scheduler": lr_scheduler1}},
                {"optimizer": optimizer2,"lr_scheduler": {"scheduler": lr_scheduler2}})


class DensityEstimation(Tasks):
    def __init__(
        self, tpp_model, discrete_diffusion, learning_rate1, learning_rate2, 
        weight_decay1, weight_decay2,
        svd_components=None, po_loss_weight=0.0 # [新增参数]
    ):
        super().__init__(
            tpp_model, discrete_diffusion, learning_rate1, learning_rate2, weight_decay1, weight_decay2
        )
         # [新增] 初始化偏序损失
        self.po_loss_fn = None
        self.po_loss_weight = po_loss_weight
        if svd_components is not None and po_loss_weight > 0:
            self.po_loss_fn = PartialOrderLoss(svd_components)

    def step(self, batch, name):
        # Forward pass
        x_n_int_x_0, log_prob_x_0, x_n = self.tpp_model.forward(batch)
        
        # Spatial loss (Original)
        spatial_loss,category_logits = self.discrete_diffusion.training_losses(batch)
        spatial_loss=spatial_loss.mean()
        
        current_epoch = self.current_epoch
        warmup_start = 50
        warmup_end = 100
        
        effective_weight = 0.0
        if self.po_loss_weight > 0:
            if current_epoch < warmup_start:
                effective_weight = 0.0
            elif current_epoch < warmup_end:
                # 线性增长
                ratio = (current_epoch - warmup_start) / (warmup_end - warmup_start)
                effective_weight = self.po_loss_weight * ratio
            else:
                effective_weight = self.po_loss_weight

        # 计算 PO Loss
        po_loss = torch.tensor(0.0, device=self.device)
        if self.po_loss_weight > 0 and self.po_loss_fn is not None and batch.po_matrix is not None:
            # 1. 获取 Logits
            #category_logits = self.discrete_diffusion.get_category_logits(batch)
            
            # 2. 计算 Loss (传入 po_matrix 而不是 po_encoding)
            # 确保 po_matrix 在同一设备上
            target_matrix = batch.po_matrix.to(self.device)
            
            po_loss = self.po_loss_fn(
                logits=category_logits,
                target_matrix=target_matrix,
                mask=batch.mask
            )
            
            # 使用动态权重
            spatial_loss = spatial_loss + effective_weight * po_loss
            
            # 记录实际使用的权重，方便观察
            self.log(f"{name}/po_loss_weight", effective_weight, batch_size=batch.batch_size)

        # Compute loss
        temporal_loss, classification, intensity = self.get_loss(
            log_prob_x_0, x_n_int_x_0, x_n
        )

        total_loss = spatial_loss + temporal_loss
       
        return temporal_loss, spatial_loss, total_loss

    def training_step(self, batch, batch_idx):
        loss_temporal, loss_spatial, loss_all = self.step(batch, "train")

        opt1, opt2 = self.optimizers()

        opt1.zero_grad()
        self.manual_backward(loss_temporal)
        opt1.step()

        opt2.zero_grad()
        self.manual_backward(loss_spatial)
        opt2.step()

        sch1, sch2 = self.lr_schedulers()
        sch1.step(loss_temporal)
        sch2.step(loss_spatial)

        self.log_dict({"t_loss": loss_temporal, "s_loss": loss_spatial}, prog_bar=True)


    def test_step(self, batch, batch_idx):
        pass

