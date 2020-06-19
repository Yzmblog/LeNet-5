import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(6),
                                  nn.MaxPool2d(kernel_size=2, stride=2),

                                  nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(16),
                                  nn.MaxPool2d(kernel_size=2, stride=2),

                                  nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=0),
                                  nn.BatchNorm2d(64))

        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(576, 84)
        self.fc2 = nn.Linear(84, 40)

    def forward(self, input):
        batch_size = input.size(0)
        out = self.net(input)
        out = out.view(batch_size, -1)
        out = self.drop(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out