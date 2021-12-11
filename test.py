import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


from fedbase.baselines import central, fedavg, ditto, cfl, local
from fedbase.model.model import CNNCifar, CNNMnist
from fedbase.nodes.node import node
import torch.optim as optim
import torch.nn as nn
import numpy as np

# np.random.seed(0)

fedavg.fedavg('./data/','mnist', 64, 10, CNNMnist, nn.CrossEntropyLoss, optim.SGD, 2, 30)
# local.local('./data/','mnist', 64, 30, CNNMnist, nn.CrossEntropyLoss, optim.Adam, 100, 3)
# ditto.ditto('./data/','mnist', 64, 30, CNNMnist, nn.CrossEntropyLoss, 'SGD', 100, 3, 0.95)
# cfl.cfl('./data/','mnist', 64, 3, 30, CNNMnist, nn.CrossEntropyLoss, 'SGD', 100, 3)