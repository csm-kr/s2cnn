# pylint: disable=E1101,R,C
import numpy as np
import torch.nn as nn
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
import gzip
import pickle
import numpy as np
from torch.autograd import Variable
import argparse
from examples.sun360.sun360_dataset import SUN360Dataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 5e-3


class S2ConvNet_original(nn.Module):

    def __init__(self):
        super(S2ConvNet_original, self).__init__()

        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 112
        b_l1 = 10
        b_l2 = 6

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.conv1 = S2Convolution(
            nfeature_in=3,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3)

        self.out_layer = nn.Linear(f2, f_output)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = so3_integrate(x)
        x = self.out_layer(x)

        return x


def main():
    root = "D:\data\SUN360_panoramas_1024x512"
    train_dataset = SUN360Dataset(root, split='train')
    test_dataset = SUN360Dataset(root, split='test')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=4,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=4,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=2)

    classifier = S2ConvNet_original()
    classifier.to(DEVICE)
    print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE).squeeze(-1)   # [0, 0, 0, 0] size 가 batch 여야 cross entropy 사용가능

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE,
                loss.item()), end="")
        print("")

        correct = 0
        total = 0
        for images, labels in test_loader:

            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).squeeze(-1)

                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()

        print('Test Accuracy: {0}'.format(100 * correct / total))


if __name__ == '__main__':
    main()
