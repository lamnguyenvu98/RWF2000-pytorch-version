from typing import Any, Optional, Union
from lightning.pytorch import Callback
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
import numpy as np
import torchmetrics

from neptune.types import File

class ComputeLoss:
    def __init__(self):
        self.all_losses = []
    
    def __call__(self, loss_value: Union[torch.Tensor, float]) -> None:
        if isinstance(loss_value, torch.Tensor):
            loss_value: float = loss_value.detach().item()
        self.all_losses.append(loss_value)
    
    def compute(self) -> None:
        return np.mean(self.all_losses)
    
    def reset(self) -> None:
        self.all_losses.clear()

class ModelMetricsCallback(Callback):
    def __init__(self, num_classes=2, task="multiclass"):
        super().__init__()
        self.train_metrics        = torchmetrics.Accuracy(num_classes=num_classes, task=task)
        self.val_metrics          = torchmetrics.Accuracy(num_classes=num_classes, task=task)
        self.val_cfm              = torchmetrics.ConfusionMatrix(task=task, num_classes=num_classes)
        self.train_loss_metrics   = ComputeLoss()
        self.val_loss_metrics     = ComputeLoss()
            
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        pred_y: torch.Tensor = outputs["predict_y"]
        gt_y: torch.Tensor = outputs["gt_y"]
        train_batch_loss = outputs["loss"]
        train_batch_accuracy = self.train_metrics(pred_y.softmax(dim=-1), gt_y)
        self.train_loss_metrics(train_batch_loss)

        # Log train batch result:
        pl_module.log("train_batch_loss", train_batch_loss, prog_bar=True, logger=False)
        pl_module.log("train_batch_accuracy", train_batch_accuracy, prog_bar=True, logger=False)        
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        train_epoch_accuracy = self.train_metrics.compute()
        train_epoch_loss = self.train_loss_metrics.compute()
        pl_module.log("train_epoch_loss", train_epoch_loss, prog_bar=False, logger=True)
        pl_module.log("train_epoch_accuracy", train_epoch_accuracy, prog_bar=False, logger=True)
        
        self.train_metrics.reset()
        self.train_loss_metrics.reset()
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        pred_y: torch.Tensor = outputs["predict_y"]
        gt_y: torch.Tensor = outputs["gt_y"]
        val_batch_loss = outputs["loss"]
        val_batch_accuracy = self.val_metrics(pred_y.softmax(dim=-1), gt_y)
        self.val_loss_metrics(val_batch_loss)
        self.val_cfm.update(pred_y.softmax(dim=-1), gt_y)

        # Log train batch result:
        pl_module.log("val_batch_loss", val_batch_loss, prog_bar=True, logger=False)
        pl_module.log("val_batch_accuracy", val_batch_accuracy, prog_bar=True, logger=False) 
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        val_epoch_loss = self.val_loss_metrics.compute()
        val_epoch_accuracy = self.val_metrics.compute()        
        fig, ax = self.val_cfm.plot(labels=["Fight", "NonFight"])
        
        pl_module.log("val_epoch_loss", val_epoch_loss, prog_bar=False, logger=True)
        pl_module.log("val_epoch_accuracy", val_epoch_accuracy, prog_bar=False, logger=True)
        
        if pl_module.logger is not None:
            pl_module.logger.experiment['CFM/ConfusionMatrix_{:02d}'.format(pl_module.current_epoch)].upload(File.as_image(fig))
        
        self.val_loss_metrics.reset()
        self.val_metrics.reset()
        self.val_cfm.reset()
        