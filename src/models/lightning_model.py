from typing import Tuple
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch import LightningModule
from neptune.types import File
from src.models.fgn_model import FlowGatedNetwork
from src.models.metrics import ConfusionMatrix
import matplotlib.pyplot as plt

class FGN(LightningModule):
    def __init__(self,
                 learning_rate: float = 0.001, 
                 momentum: float = 0.9, 
                 weight_decay: float = 1e-6, 
                 step_size: int = 10, 
                 gamma: float = 0.7) -> None:
        super(FGN, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.save_hyperparameters()
        self.example_input_array  = torch.randn((1, 5, 64, 224, 224))
        
        self.learning_rate        = learning_rate
        self.momentum             = momentum
        self.weight_decay         = weight_decay
        self.step_size            = step_size
        self.gamma                = gamma
        
        self.loss_function        = nn.CrossEntropyLoss()
        self.train_metrics        = torchmetrics.Accuracy(num_classes=2, task="multiclass")
        self.val_metrics          = torchmetrics.Accuracy(num_classes=2, task="multiclass")
        self.test_metric_acc      = torchmetrics.Accuracy(num_classes=2, task="multiclass")
        self._precision           = torchmetrics.Precision(num_classes=2, task="multiclass", ignore_index=1)
        self.recall               = torchmetrics.Recall(num_classes=2, task="multiclass", ignore_index=1)
        self.val_cfm              = ConfusionMatrix()
        
        self.model                = FlowGatedNetwork()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.model(x)

    def on_train_epoch_start(self):
       self.optimizers().param_groups[0]['lr'] = self.lr_schedulers().get_last_lr()[0]

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        batch_loss = self.loss_function(preds, y)
        acc = self.train_metrics(preds.softmax(dim=-1), y)
        lr = self.optimizers().param_groups[0]['lr']
        self.log("batch_acc", acc, prog_bar=True, logger=False)
        self.log("lr", lr, prog_bar=True, logger=False)
        return {'loss': batch_loss * self.trainer.accumulate_grad_batches}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mean_acc = self.train_metrics.compute()
        lr = self.optimizers().param_groups[0]['lr']
        self.log("train_acc", mean_acc)
        self.log("train_loss", avg_loss)
        self.log("lr", lr, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        batch_loss = self.loss_function(preds, y)
        acc = self.val_metrics(preds.softmax(dim=-1), y)
        self.log("val_b_loss", batch_loss, prog_bar=True, logger=False)
        self.log("val_b_acc", acc, prog_bar=True, logger=False)
        return {'batch_val_loss': batch_loss, 'gt': y, 'pred': preds.softmax(dim=-1).argmax(dim=-1)}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        mean_acc = self.val_metrics.compute()
        self.log("val_loss", loss)
        self.log("val_acc", mean_acc)
        
        # Draw Confusion Matrix
        y_true = torch.cat([x['gt'] for x in outputs])
        y_preds = torch.cat([x['pred'] for x in outputs])
        fig = plt.figure()
        self.val_cfm(y_preds, y_true)
        self.val_cfm._plot()
        self.logger.experiment['CFM/ConfusionMatrix_{}'.format(self.current_epoch)].upload(File.as_image(fig))
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        batch_loss = self.loss_function(preds, y)
        acc = self.test_metric_acc(preds.softmax(dim=-1), y)
        return {'gt': y, 'pred': preds.softmax(dim=-1).argmax(dim=-1)}

    def test_epoch_end(self, outputs):
        cfm = ConfusionMatrix()
        y_true = torch.cat([x['gt'] for x in outputs])
        y_preds = torch.cat([x['pred'] for x in outputs])
        result = cfm(y_preds, y_true)
        cfm._plot()
        plt.show()
        mean_acc = self.test_metric_acc.compute()
        self.log('test_acc', mean_acc, logger=False)
        self.test_metric_acc.reset()
        return {'test_acc': mean_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9,15], gamma=self.gamma)
        return [optimizer], [scheduler]

if __name__ == '__main__':
  ckp_path = 'model_dir/best.ckpt'
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dummy_input = torch.randn((1, 5, 64, 224, 224))
  model = FGN().to(device)
  trained_ckp = torch.load(ckp_path, map_location='cpu')['state_dict']
  # model_ckp = model.state_dict()
  # for k, v in model_ckp.items():
  #   model_ckp[k] = trained_ckp['model.' + k]
  model.load_state_dict(trained_ckp)
  model.eval()
  out = model(dummy_input)
  print(out.shape)
  
  
  
