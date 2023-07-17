import torchmetrics
import seaborn as sns
import torch

class ConfusionMatrix():
  def __init__(self, num_classes: int = 2):
    self.num_classes = num_classes
    self.result = None

  def __call__(self, y_preds, y_true):
    result: torch.Tensor = torchmetrics.functional.confusion_matrix(y_preds, 
                                                                    y_true, 
                                                                    num_classes=self.num_classes,
                                                                    task="multiclass")
    self.result = result.detach().cpu().numpy()

  def _plot(self):
    sns.heatmap(self.result, annot=True, fmt='g')


