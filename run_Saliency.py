import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class Net(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc_step = pl.metrics.Accuracy()
        self.train_acc_epoch = pl.metrics.Accuracy()
        self.test_acc_step = pl.metrics.Accuracy()
        self.test_acc_epoch = pl.metrics.Accuracy()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.drop_out = torch.nn.Dropout(p=0.5)
        self.lin1 = torch.nn.Linear(7 * 7 * 64, 1000)
        self.lin2 = torch.nn.Linear(1000, 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.drop_out(x)
        x = self.relu(self.lin1(x))
        x = self.softmax(self.lin2(x))
        return x


def imshow(img, transpose=True):
    img = img  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')
    net = Net(batch_size=60000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    USE_PRETRAINED_MODEL = True

    if USE_PRETRAINED_MODEL:
        print("Using existing trained model")
        net.load_state_dict(torch.load('../models/MNIST.pt'))
    else:
        for epoch in range(5):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), '../models/MNIST_Saliency.pt')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    ind = 1

    input = images[ind].unsqueeze(0)
    input.requires_grad = True

    net.eval()

    def attribute_image_features(algorithm, input, **kwargs):
        net.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                  target=labels[ind],
                                                  **kwargs
                                                  )

        return tensor_attributions

    saliency = Saliency(net)
    nt = NoiseTunnel(saliency)
    grads = attribute_image_features(nt, input, nt_type='smoothgrad',
                                     n_samples=5, stdevs=0.2)
    grads = np.transpose(grads.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    ig = IntegratedGradients(net)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    ig = IntegratedGradients(net)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq', stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # dl = DeepLift(net)
    # attr_dl = attribute_image_features(dl, input, baselines=input * 0)
    # attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    print('Original Image')
    print('Predicted:', classes[predicted[ind]],
          ' Probability:', torch.max(F.softmax(outputs, 1)).item())

    original_image = np.transpose(images[ind].cpu().detach().numpy(), (1, 2, 0))

    _ = viz.visualize_image_attr(None, original_image,
                                 method="original_image", title="Original Image")

    _ = viz.visualize_image_attr(grads, original_image, method="heat_map", sign="absolute_value",
                                 show_colorbar=True, title="Overlayed Gradient Magnitudes with SmoothGrad")
    #
    _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="all",
                                 show_colorbar=True, title="Overlayed Integrated Gradients")

    _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value",
                                 outlier_perc=10, show_colorbar=True,
                                 title="Overlayed Integrated Gradients \n with SmoothGrad Squared")

    # _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map", sign="all", show_colorbar=True,
    #                              title="Overlayed DeepLift")


if __name__ == '__main__':
    main()
