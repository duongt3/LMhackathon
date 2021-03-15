import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # # skip_connection
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=1)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=1)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet10(img_channels=3, num_classes=10):
    return ResNet(block, [1, 1, 1, 1], img_channels, num_classes)


def ResNet12(img_channels=3, num_classes=10):
    return ResNet(block, [1, 1, 2, 1], img_channels, num_classes)


def ResNet14(img_channels=3, num_classes=10):
    return ResNet(block, [1, 2, 2, 1], img_channels, num_classes)


def ResNet16(img_channels=3, num_classes=10):
    return ResNet(block, [1, 2, 2, 2], img_channels, num_classes)


def ResNet18(img_channels=3, num_classes=10):
    return ResNet(block, [2, 2, 2, 2], img_channels, num_classes)

def ResNet20(img_channels=3, num_classes=10):
    return ResNet(block, [2, 3, 2, 2], img_channels, num_classes)

def ResNet22(img_channels=3, num_classes=10):
    return ResNet(block, [2, 3, 3, 2], img_channels, num_classes)

def ResNet24(img_channels=3, num_classes=10):
    return ResNet(block, [2, 3, 3, 3], img_channels, num_classes)

def ResNet26(img_channels=3, num_classes=10):
    return ResNet(block, [3, 3, 3, 3], img_channels, num_classes)

def ResNet34(img_channels=3, num_classes=10):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

def ResNet50(img_channels=3, num_classes=10):
    return ResNet(block, [6, 6, 6, 6], img_channels, num_classes)

def ResNet98(img_channels=3, num_classes=10):
    return ResNet(block, [12, 12, 12, 12], img_channels, num_classes)

def test():
    net = ResNet18()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
    print(y.shape)

def getModelWeights():
    model = ResNet98(img_channels=1, num_classes=10)
    model.load_state_dict(torch.load("../models/ResNet98_noskip.pt"))
    model.eval()

    weights = {}
    # Print model's state_dict
    print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # for name, param in model.named_parameters():
    #     if "weight" in name:
    #         weights[name] = param

    # pd.DataFrame.from_dict(weights)
    # print(weights)

    weights = []
    for name, param in model.named_parameters():
        if "weight" in name:
            weights.append(param.detach().numpy().flatten())
            # for weight in param.detach().numpy():
            #     weights.append(weight.flatten())

    flatlist = []
    for layer in weights:
        for weight in layer:
            flatlist.append(weight)

    df = pd.DataFrame(flatlist)
    sns.kdeplot(data=flatlist, shade=True)
    plt.xlim([-5, 5])
    plt.xlabel("Weight Values")
    plt.title("98 Layer CNN without Skip Connection")
    plt.show()
    print(df)


def main():
    # Retrieve MNIST data set and split into train/test
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

    batch_size = 100

    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=mnist_test,
        batch_size=batch_size,
        shuffle=True)

    # Define Model: MNIST has 1 channel and 10 classes
    model = ResNet98(img_channels=1, num_classes=10)
    model.to('cuda:0')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    no_epochs = 5
    total_train = 0
    correct_train = 0
    # Training
    for epoch in range(no_epochs):
        total_train_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(x.float().to('cuda:0'))
            if epoch == no_epochs - 1:  # Find the training accuracy for the last epoch
                train_pred = torch.argmax(pred.data, dim=1).cpu().numpy()
                total_train += x.data.size()[0]
                correct_train += np.sum(train_pred == target.data.numpy())
            loss = criterion(pred, target.long().to('cuda:0'))
            loss.backward()
            optimizer.step()
        print('Epoch: ' + str(epoch + 1) + '/' + str(no_epochs) + ', Train Loss: ' + str(loss.item()))
    train_accuracy = (correct_train / total_train) * 100
    print('Training Accuracy: ' + str(train_accuracy))

    # Testing
    total_test = 0
    correct_test = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        out = model(x.float().to('cuda:0'))
        loss = criterion(out, target.long().to('cuda:0'))
        pred_batch = torch.argmax(out.data, dim=1).cpu().numpy()
        total_test += x.data.size()[0]
        correct_test += np.sum(pred_batch == target.data.numpy())

    test_accuracy = (correct_test / total_test) * 100
    print('Test Accuracy' + str(test_accuracy))

    total_params = sum(p.numel() for p in model.parameters())

    model_name = '../models/ResNet98_noskip.pt'
    torch.save(model.state_dict(), str(model_name))

    outputs = [total_params, test_accuracy, train_accuracy]
    f = open('../outputs/ResNetOutputs_noskip.npy', 'a')
    np.savetxt(f, [outputs], fmt='%d %10.5f %10.5f', delimiter=' ')
    f.close()

if __name__ == "__main__":
    #main()
    getModelWeights()
