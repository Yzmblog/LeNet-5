from LeNet import LeNet
import torch

net = LeNet()
test = torch.randn((64, 1, 28, 28))
out = net(test)
print(out.size())