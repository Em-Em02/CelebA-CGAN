import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_SIZE = 256

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.linear1 = nn.Linear(LATENT_SIZE, 16384)
    self.conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
    self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=2)
    self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=2)
    self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=2)
    self.conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)


    self.conv_refine = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    self.bn1 = nn.BatchNorm2d(num_features=512)
    self.bn2 = nn.BatchNorm2d(num_features=256)
    self.bn3 = nn.BatchNorm2d(num_features=128)
    self.bn4 = nn.BatchNorm2d(num_features=64)

  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = torch.unflatten(x, 1, (1024, 4, 4))
    x = F.leaky_relu(self.bn1(self.conv1(x)))
    x = F.leaky_relu(self.bn2(self.conv2(x)))
    x = F.leaky_relu(self.bn3(self.conv3(x)))
    x = F.leaky_relu(self.bn4(self.conv4(x)))
    x = F.relu(self.conv5(x))
    x = torch.tanh(self.conv_refine(x))
    return x
  
def generator_loss(pred_synth):
  t_synth = torch.ones_like(pred_synth)
  return F.binary_cross_entropy(pred_synth, t_synth)