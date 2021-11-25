from utils.data_loader import data_process
from nodes.node import node
# from server.server import server
from model.model import CNNMnist, MLP, CNNCifar, CNNFashion_Mnist
from model.resnet import resnet18
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from copy import deepcopy
import torch.multiprocessing as mp
from utils.model_utils import save_checkpoint, load_checkpoint
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameter
dataset = 'Fashion_Mnist'
num_nodes = 1
global_rounds = 100
local_epochs = 1
local_steps = 1
batch_size = 4

if __name__ == '__main__':
    # split data
    dt = data_process(dataset)
    trainset,testset = dt.train_dataset, dt.test_dataset
    net = CNNFashion_Mnist()
    net.to(device)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    def test():
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
            100 * correct / total))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(global_rounds):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        test()
    print('Finished Training')
