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
    
    # Initialize kernel (he_normal)    
    # self.Conv3DBlock.apply(self.init_weights)
  
  def forward(self, x):
    return self.Conv3DBlock(x)
  
  # def init_weights(self, m):
  #   if isinstance(m, nn.Conv3d):
  #     nn.init.kaiming_normal_(m.weight)

# Layers that combine 2 output of RGB_Flow_Layer into one 
# (using element wise matrix multiplication)
class Fusion(nn.Module):
  def __init__(self):
    super(Fusion, self).__init__()
    return None
  def forward(self, x):
    rgb, opt = x
    x = torch.mul(rgb, opt)
    return x

class AttentionMechanism(nn.Module):
  def __init__(self):
     super().__init__()
     self.alpha =  nn.Parameter(torch.zeros(1))
     self.softmax = nn.Softmax(dim=-1)
  
  def forward(self, x):
      x_vectorize = x.view(x.size(0), x.size(1), -1)
      F = torch.bmm(x_vectorize, x_vectorize.permute(0, 2, 1))
      scores = self.softmax(F)
      value = torch.bmm(scores, x_vectorize)
      value = value.view_as(x)
      x = self.alpha * value + x
      return x
      

class FlowGatedNetworkV2(nn.Module):
  def __init__(self):
    super(FlowGatedNetworkV2, self).__init__()
        
    self.RGB_Network = nn.Sequential(
            Conv3d_Block(3, 16, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(16, 16, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(16, 32, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(32, 32, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(32, 32, pool_size=(2, 2, 2), activation='relu')
        )

    self.OptFlow_Network = nn.Sequential(
            Conv3d_Block(2, 16, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(16, 16, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(16, 32, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(32, 32, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(32, 32, pool_size=(2, 2, 2), activation='relu')
        )

    self.conv1 = nn.Sequential(
            Conv3d_Block(64, 128, pool_size=(2, 1, 1), activation='relu'),
            Conv3d_Block(128, 64, pool_size=(1, 1, 1), activation='relu')
    )

    self.attention = AttentionMechanism()

    self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
    )
    
    self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
    )
    
    self.maxpool2d = nn.MaxPool2d((4, 4))

    self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )    
    
  def forward(self, x):
      rgb = self.RGB_Network(x[:, :3, ...])
      opt = self.OptFlow_Network(x[:, 3:, ...])
      x = torch.concat([rgb, opt], dim=1)
      x = self.conv1(x)
      x = x.squeeze(dim=2)
      x = self.attention(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.maxpool2d(x)
      x = self.FC(x)
      print(x.shape)
      return x

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
            Conv3d_Block(2, 16, pool_size=(1, 2, 2), activation='relu'),
            Conv3d_Block(16, 16, pool_size=(1, 2, 2), activation='relu'),
            Conv3d_Block(16, 32, pool_size=(1, 2, 2), activation='relu'),
            Conv3d_Block(32, 32, pool_size=(1, 2, 2), activation='relu'),
        )

    self.Fusion = Fusion()

    self.MaxPool = nn.MaxPool3d((8, 1, 1))

    self.Merging = nn.Sequential(
            Conv3d_Block(32, 64, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(64, 64, pool_size=(2, 2, 2), activation='relu'),
            Conv3d_Block(64, 128, pool_size=(2, 2, 2), activation='relu')
        )

    self.FC = nn.Sequential(
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
      x = self.Fusion([rgb, opt])
      x = self.MaxPool(x)
      x = self.Merging(x)
      x = self.FC(x)
      return x

class TrainingModel(LightningModule):
    def __init__(self, lr: float = 0.001, 
                 momentum: float = 0.9, 
                 weight_decay: float = 1e-6, 
                 step_size: int = 10, 
                 gamma: float = 0.7):
        super(TrainingModel, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.lr                   = lr
        self.momentum             = momentum
        self.weight_decay         = weight_decay
        self.step_size            = step_size
        self.gamma                = gamma
        
        self.loss_function        = nn.CrossEntropyLoss()
        self.train_metrics        = torchmetrics.Accuracy()
        self.val_metrics          = torchmetrics.Accuracy()
        self.test_metric_acc      = torchmetrics.Accuracy()
        self.val_cfm              = ConfusionMatrix()
        
        self.save_hyperparameters()
        
        self.example_input_array  = torch.randn((1, 5, 64, 224, 224))

        self.model = FlowGatedNetworkV2()
        self.model.apply(self.init_weights)

    def forward(self, x):
      return self.model(x)

    def init_weights(self, m):
      if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight)

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        batch_loss = self.loss_function(preds, y)
        acc = self.train_metrics(preds.softmax(dim=-1), y)
        self.log("batch_acc", acc, prog_bar=True, logger=False)
        return {'loss': batch_loss * self.trainer.accumulate_grad_batches}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mean_acc = self.train_metrics.compute()
        self.train_metrics.reset()
        lr = self.optimizers().param_groups[0]['lr']
        self.log("train_acc", mean_acc)
        self.log("train_loss", avg_loss)
        self.log("lr", lr)
        # self.logger.experiment.add_scalar("Loss/Train", loss, self.current_epoch)
        # self.logger.experiment.add_scalar("Acc/Train", mean_acc, self.current_epoch)
        # self.logger.experiment.add_scalar("LearningRate", lr, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        batch_loss = self.loss_function(preds, y)
        acc = self.val_metrics(preds.softmax(dim=-1), y)
        self.log("val_b_loss", batch_loss, prog_bar=True, logger=False)
        self.log("val_b_acc", acc, prog_bar=True, logger=False)
        return {'batch_val_loss': batch_loss, 'gt': y, 'pred': preds.softmax(dim=-1).argmax(dim=-1)}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        mean_acc = self.val_metrics.compute() 
        self.val_metrics.reset()
        self.log("val_loss", loss)
        self.log("val_acc", mean_acc)
        if self.current_epoch % 2 == 0:
            y_true = torch.cat([x['gt'] for x in outputs])
            y_preds = torch.cat([x['pred'] for x in outputs])
            fig = plt.figure()
            self.val_cfm(y_preds, y_true)
            self.val_cfm._plot()
            self.logger.experiment['ConfusionMatrix_E{}'.format(self.current_epoch)].upload(File.as_image(fig))
        # self.logger.experiment.add_scalar("Loss/Val", loss, self.current_epoch)
        # self.logger.experiment.add_scalar("Acc/Val", mean_acc, self.current_epoch)

    def test_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        batch_loss = self.loss_function(preds, y)
        acc = self.test_metric_acc(preds.softmax(dim=-1), y)
        return {'gt': y, 'pred': preds.softmax(dim=-1).argmax(dim=-1)}

    def test_epoch_end(self, outputs):
        cfm = ConfusionMatrix()
        y_true = torch.cat([x['gt'] for x in outputs])
        y_preds = torch.cat([x['pred'] for x in outputs])
        result = cfm(y_preds, y_true)
        mean_acc = self.test_metric_acc.compute()
        self.test_metric_acc.reset()
        self.log('test_acc', mean_acc)
        return {'test_acc': mean_acc, 'cfm_object': cfm}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dummy_input = torch.randn((1, 5, 64, 224, 224)).to(device)
  model = FlowGatedNetworkV2().to(device)
  # model(dummy_input)
  train_model = TrainingModel()
  train_model(dummy_input)