from turtle import forward
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from neptune.new.types import File
import matplotlib.pyplot as plt

class ConfusionMatrix():
  def __init__(self, num_classes=2):
    self.num_classes=num_classes
    self.result = None
  
  def __call__(self, y_preds, y_true):
    result = torchmetrics.functional.confusion_matrix(y_preds, y_true, num_classes=self.num_classes)
    self.result = result.detach().cpu().numpy()
  
  def _plot(self):
    import seaborn as sns
    sns.heatmap(self.result, annot=True, fmt='g')

class AttentionMechanism(nn.Module):
  def __init__(self):
     super().__init__()
     self.alpha =  nn.Parameter(torch.zeros(1))
     self.softmax = nn.Softmax(dim=-1)
  
  def forward(self, x):
      query = x.view(x.size(0), x.size(1), -1)
      key = query.permute(0, 2, 1)
      energy = torch.bmm(query, key)
      energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
      attention = self.softmax(energy_new)
      value = x.view(x.size(0), x.size(1), -1)
      out = torch.bmm(attention, value)
      out = out.view_as(x)
      out = self.alpha * out + x
      return out

# Conv 3d Block
# Weight initialize: kaiming normal
class Conv3d_Block(nn.Module):
  def __init__(self, in_channels, out_channels, pool_size, activation='relu'):
    super(Conv3d_Block, self).__init__()

    acts_fn = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid()
    }

    self.activation = acts_fn.get(activation, nn.ReLU())
    
    self.Conv3DBlock = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding="same", bias=False),
        nn.BatchNorm3d(out_channels),
        self.activation,
        nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding="same", bias=False),
        nn.BatchNorm3d(out_channels),
        self.activation,
        nn.MaxPool3d(pool_size)
      )  
      
  def forward(self, x):
    return self.Conv3DBlock(x)
  
      
class FlowGatedNetwork(nn.Module):
  def __init__(self):
    super(FlowGatedNetwork, self).__init__()
    self.RGB_Network = nn.Sequential(
            Conv3d_Block(3, 16, pool_size=(1, 2, 2), activation='relu'),
            Conv3d_Block(16, 16, pool_size=(1, 2, 2), activation='relu'),
            Conv3d_Block(16, 32, pool_size=(1, 2, 2), activation='relu'),
            Conv3d_Block(32, 32, pool_size=(1, 2, 2), activation='relu'),
        )

    self.OptFlow_Network = nn.Sequential(
            Conv3d_Block(2, 16, pool_size=(1, 2, 2), activation='sigmoid'),
            Conv3d_Block(16, 16, pool_size=(1, 2, 2), activation='sigmoid'),
            Conv3d_Block(16, 32, pool_size=(1, 2, 2), activation='sigmoid'),
            Conv3d_Block(32, 32, pool_size=(1, 2, 2), activation='sigmoid'),
        )

    self.MaxPool = nn.MaxPool3d((8, 1, 1))

    self.Merging = nn.Sequential(
            Conv3d_Block(32, 64, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(64, 64, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(64, 128, pool_size=(2, 2, 2), activation='relu')
        )

    self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )    

  def forward(self, x):
      rgb = self.RGB_Network(x[:, :3, ...])
      opt = self.OptFlow_Network(x[:, 3:, ...])
      x = torch.mul(rgb, opt)
      x = self.MaxPool(x)
      x = self.Merging(x)
      x = self.classifier(x)
      return x

class TrainingModel(LightningModule):
    def __init__(self,
                 learning_rate: float = 0.001, 
                 momentum: float = 0.9, 
                 weight_decay: float = 1e-6, 
                 step_size: int = 10, 
                 gamma: float = 0.7):
        super(TrainingModel, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.save_hyperparameters()
        self.example_input_array  = torch.randn((1, 5, 64, 224, 224))
        
        self.learning_rate        = learning_rate
        self.momentum             = momentum
        self.weight_decay         = weight_decay
        self.step_size            = step_size
        self.gamma                = gamma
        
        self.loss_function        = nn.CrossEntropyLoss()
        self.train_metrics        = torchmetrics.Accuracy()
        self.val_metrics          = torchmetrics.Accuracy()
        self.test_metric_acc      = torchmetrics.Accuracy()
        self.val_cfm              = ConfusionMatrix()
        
        self.model = FlowGatedNetwork()
        self.model.apply(self.init_weights)

    def forward(self, x):
      return self.model(x)

    def init_weights(self, m):
      if isinstance(m, nn.Conv3d):
          nn.init.kaiming_normal_(m.weight)

    # def on_train_epoch_start(self):
    #    self.optimizers().param_groups[0]['lr'] = self.lr_schedulers().get_last_lr()[0]

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
        if self.current_epoch % 2 == 0:
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
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5, min_lr=1e-9)
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
            'monitor': 'val_acc' 
        }
        return [optimizer], [scheduler]

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dummy_input = torch.randn((1, 5, 64, 224, 224))
  model = FlowGatedNetwork()
  model.eval()
  model(dummy_input)