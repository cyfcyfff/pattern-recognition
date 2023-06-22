import lenet
from lenet import device
import torch

net = lenet.LeNet().to(device)
print(net.parameters())