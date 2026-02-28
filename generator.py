import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_SIZE = 256
ATTRIBUTE_SIZE = 40

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.linear1 = nn.Linear(LATENT_SIZE+ATTRIBUTE_SIZE, 16384)
    self.conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=2)
    self.conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=2)
    self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=2)
    self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3 , kernel_size=4, stride=2, padding=1)


    self.conv_refine = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

    self.bn1 = nn.BatchNorm2d(num_features=128)
    self.bn2 = nn.BatchNorm2d(num_features=64)
    self.bn3 = nn.BatchNorm2d(num_features=32)

  def forward(self, x, cond):
    x_cond = torch.cat((x, cond), dim=1)
    x = F.relu(self.linear1(x_cond))
    x = torch.unflatten(x, 1, (256, 8, 8))
    x = F.leaky_relu(self.bn1(self.conv1(x)))
    x = F.leaky_relu(self.bn2(self.conv2(x)))
    x = F.leaky_relu(self.bn3(self.conv3(x)))
    x = F.relu(self.conv4(x))
    x = torch.tanh(self.conv_refine(x))
    return x
  
def generator_loss(pred_synth):
  t_synth = torch.ones_like(pred_synth)
  return F.binary_cross_entropy_with_logits(pred_synth, t_synth)