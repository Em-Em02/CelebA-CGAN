import torch
import torch.nn as nn
import torch.nn.functional as F

LABEL_SMOOTHING = 0.0

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=0)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.linear1 = nn.Linear(256*6*6, 1)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x))
    x = F.leaky_relu(self.conv2(x))
    x = F.leaky_relu(self.conv3(x))
    x = F.leaky_relu(self.conv4(x))
    x = F.leaky_relu(self.conv5(x))
    x = torch.flatten(x, 1)
    x = self.linear1(x)
    x = torch.sigmoid(x)
    return x
  
def discriminator_loss(pred_synth, pred_real):
  t_synth = torch.zeros_like(pred_synth)+LABEL_SMOOTHING
  t_real = torch.ones_like(pred_real)-LABEL_SMOOTHING
  return F.binary_cross_entropy(pred_synth, t_synth) + F.binary_cross_entropy(pred_real, t_real)