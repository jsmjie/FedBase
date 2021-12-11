import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


from fedbase.baselines import central, fedavg, ditto, cfl, local
from fedbase.model.model import CNNCifar, CNNMnist, CNNFashion_Mnist
from fedbase.nodes.node import node
import torch.optim as optim
import torch.nn as nn
import numpy as np

# np.random.seed(0)

# central.central('fashion_mnist', 64, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 2)
# central.central('cifar10', 64, CNNCifar, nn.CrossEntropyLoss, optim.SGD, 2)
# fedavg.fedavg('fashion_mnist', 64, 20, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 30, split_para = 0.1, split_method = 'dirichlet')
# local.local('fashion_mnist', 64, 30, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 3, split_para = 0.1, split_method = 'dirichlet')
# ditto.ditto('fashion_mnist', 64, 30, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 3, 0.95, split_para = 0.1, split_method = 'dirichlet')
cfl.cfl('fashion_mnist', 64, 3, 30, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 30, split_para = 0.1, split_method = 'dirichlet')
# cfl.cfl('fashion_mnist', 64, 3, 30, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 30, split_para = 3, split_method = 'class')