import torchmetrics
import seaborn as sns
import torch
import numpy as np
from typing import Union

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

class ConfusionMatrix():
  def __init__(self, num_classes: int = 2, task: str = "multiclass"):
    self.num_classes = num_classes
    self.task = task
    self.cfm = torchmetrics.ConfusionMatrix(task=task, num_classes=num_classes)

  def __call__(self, y_preds, y_true):
    result: torch.Tensor = self.cfm(y_preds, y_true)
    self.result = result.detach().cpu().numpy()

  def _plot(self):
    sns.heatmap(self.result, annot=True, fmt='g')


