import torch
import torch.nn as nn

# Conv 3d Block
# Weight initialize: kaiming normal
class Conv3d_Block(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, pool_size: tuple = (1, 2, 2), activation: str = 'relu') -> torch.Tensor:
    super(Conv3d_Block, self).__init__()

    acts_fn = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid()
    }

    self.activation = acts_fn.get(activation, nn.ReLU())

    self.Conv3DBlock = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        self.activation,
        nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        self.activation,
        nn.MaxPool3d(pool_size)
      )

    self.Conv3DBlock.apply(self.init_weights)

  def forward(self, x):
    return self.Conv3DBlock(x)

  def init_weights(self, m):
    if isinstance(m, nn.Conv3d):
      nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm3d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()


class FlowGatedNetwork(nn.Module):
  def __init__(self) -> None:
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
            Conv3d_Block(32, 32, pool_size=(1, 2, 2), activation='sigmoid'),
        )

    self.MaxPool = nn.MaxPool3d((8, 1, 1))

    # self.attention_mechanism = AttentionMechanism()

    self.Merging = nn.Sequential(
            Conv3d_Block(32, 64, pool_size=(2, 1, 1), activation='relu'),
            Conv3d_Block(64, 64, pool_size=(2, 1, 1), activation='relu'),
            Conv3d_Block(64, 128, pool_size=(2, 1, 1), activation='relu'),
        )

    self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    self.fc = nn.Linear(128, 2)

    # self.classifier = nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(128, 128),
    #         nn.ReLU(),
    #         nn.Dropout(0.2),
    #         nn.Linear(128, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, 2),
    #     )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      rgb = self.RGB_Network(x[:, :3, ...])
      opt = self.OptFlow_Network(x[:, 3:, ...])
      x = torch.mul(rgb, opt)
      x = self.MaxPool(x)
      x = self.Merging(x)
      # x = x.squeeze(2)
      x = self.global_avg_pool(x)
      # print(x.shape)
      # x = self.classifier(x)
      x = x.flatten(start_dim=1)
      x = self.fc(x)
      return x.view(x.size(0), -1)

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dummy_input = torch.randn((1, 5, 64, 224, 224))
  model = FlowGatedNetwork().to(device)
  out = model(dummy_input.to(device))
  print(out.shape)
