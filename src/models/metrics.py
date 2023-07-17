import torchmetrics
import seaborn as sns
import torch

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


