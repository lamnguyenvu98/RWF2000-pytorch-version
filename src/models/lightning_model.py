import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from src.models.fgn_model import FlowGatedNetwork

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
        # self._precision           = torchmetrics.Precision(num_classes=2, task="multiclass", ignore_index=1)
        # self.recall               = torchmetrics.Recall(num_classes=2, task="multiclass", ignore_index=1)
        
        self.model                = torch.compile(FlowGatedNetwork())

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        batch_loss = self.loss_function(preds, y)
        loss = batch_loss * self.trainer.accumulate_grad_batches
        outputs = {'loss': loss, "predict_y": preds, "gt_y": y}
        return outputs

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        loss = self.loss_function(preds, y)
        outputs = {'loss': loss, 'predict_y': preds, 'gt_y': y}
        return outputs


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
