import os
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

class ConvNet(pl.LightningModule):
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc_epoch(logits, y)
        self.log('train_acc_step', self.train_acc_step(logits, y), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.test_acc_epoch(logits, y)
        self.log_dict({'train_acc': self.train_acc_epoch.compute(), 'test_acc': self.test_acc_epoch.compute()},
                      prog_bar=True, on_step=False, on_epoch=True)

    def training_epoch_end(self, outs):
        self.train_acc_epoch.reset()

    def validation_epoch_end(self, outs):
        self.test_acc_epoch.reset()

    def train_dataloader(self):
        # transforms
        # prepare transforms standard to MNIST
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        # data
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(mnist_train, batch_size=self.batch_size, num_workers=2, shuffle=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(mnist_test, batch_size=self.batch_size, num_workers=2, shuffle=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_progress_bar_dict(self):
        # don't show the version number
        # This just stops the version number from printing out on the progress bar. Not necessary to run the model.
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


def main():
    USE_PRETRAINED_MODEL = False
    no_epochs = 10
    vanilla_batch = 60000  # This is the entire size of the training set.
    model = ConvNet(
        batch_size=vanilla_batch)  # Default batch size will get overwritten by the auto_scale_batch_size method

    if USE_PRETRAINED_MODEL:
        print("Using existing trained model")
        model.load_state_dict(torch.load('../models/MNIST_2.pt'))
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=no_epochs, auto_scale_batch_size='power',
                             check_val_every_n_epoch=1)  # Auto scale arguments can be: [None, 'power', 'binsearch']
        trainer.tune(model)  # This does the auto scaling
        trainer.fit(model)  # This does the training and validation
        # trainer.test(model)    # Runs the test data (if you had one)
        torch.save(model.state_dict(), '../models/MNIST1.pt')

if __name__ == '__main__':
    main()
    # To see the things you logged in tensorboard run the following command in the directory of the file
    # tensorboard --logdir=./lightning_logs --port=6006
