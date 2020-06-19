import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from LeNet import LeNet
import argparse
import tensorboardX as tX
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='LeNet-5')
parser.add_argument('--logdir', default='log', help='log directory')
parser.add_argument('--num-epochs', type=int, default=60, help='number of training epochs')

args = parser.parse_args()

writer = tX.SummaryWriter(log_dir=args.logdir, comment='LeNet-5')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))])
transform1 = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))])

data_train = torchvision.datasets.MNIST(root='./data', transform=transform, download=True, train=True)
data_test = torchvision.datasets.MNIST(root='./data', transform= transform, download=True, train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=64, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=64, shuffle=True)


# for i, data in enumerate(data_loader_train):
#     inputs, label = data
#     plt.imshow(inputs[0].reshape(28, 28))
#     plt.show()

def train(model, optimizer, criterion, epochToPrint, loss_to_draw):
    model.train()
    correct = 0
    for idx, data in enumerate(data_loader_train):
        inputs, label = data
        inputs = inputs.to(device)
        label = label.to(device)
        out = model(inputs)
        loss = criterion(out, label)
        loss_to_draw.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = out.max(1, keepdim=True)[1]  ##find the index of the max value
        correct += pred.eq(label.view_as(pred)).sum().item()
    print('epoch: {} || loss: {}'.format(epochToPrint, loss.item()))
    print('Accuracy in train set is {}%'.format(100. * correct / len(data_loader_train.dataset)))

def test(model):
    correct = 0
    # num = 0
    with torch.no_grad():
        for idx, data in enumerate(data_loader_test):
            inputs, label = data
            inputs = inputs.to(device)
            label = label.to(device)
            out = model(inputs)
            pred = out.max(1, keepdim=True)[1] ##find the index of the max value
            correct += pred.eq(label.view_as(pred)).sum().item()
            # if (num <= 5):
            #     error_idx = (pred.eq(label.view_as(pred)) == 0).nonzero()
            #     fig, ax = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
            #     ax = ax.flatten()
            #     for n in error_idx:
            #         i = 0
            #         img = inputs[n[0]].reshape(28, 28)
            #         ax[i].set_title(pred[n[0]][0].item())
            #         img = img.cpu()
            #         ax[i].imshow(img)
            #         i += 1
            #     plt.show()
    print('Accuracy in test set is {}%'.format(100. * correct / len(data_loader_test.dataset)))

def main():
    model = LeNet().to(device)
    optim = torch.optim.Adam(model.parameters())
    criterion = F.cross_entropy
    loss_to_draw = []
    for i in range(args.num_epochs):
        train(model, optim, criterion, i, loss_to_draw)
        test(model)
    x = np.arange(1, 60000, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_to_draw)
    plt.show()




if __name__ == '__main__':
    main()

