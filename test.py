import os
from fedbase.baselines import central, fedavg, ditto, cfl, local
from fedbase.model.model import CNNCifar, CNNMnist, CNNFashion_Mnist
from fedbase.nodes.node import node
import torch.optim as optim
import torch.nn as nn
import numpy as np
from multiprocessing import Process, Pool
from functools import partial

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# np.random.seed(0)
# central.central('fashion_mnist', 64, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 2)
# central.central('cifar10', 64, CNNCifar, nn.CrossEntropyLoss, optim.SGD, 2)
# fedavg.fedavg('fashion_mnist', 64, 20, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 30, split_para = 0.1, split_method = 'dirichlet')
# fedavg.fedavg('cifar10', 64, 20, CNNCifar, nn.CrossEntropyLoss, optim.SGD, 1, 30, **{'split_para': 0.1, 'split_method': 'dirichlet'})
# local.local('fashion_mnist', 64, 30, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 3, split_para = 0.1, split_method = 'dirichlet')
# ditto.ditto('fashion_mnist', 64, 30, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 3, 0.95, split_para = 0.1, split_method = 'dirichlet')
# cfl.cfl('fashion_mnist', 64, 3, 30, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 30, split_para = 0.1, split_method = 'dirichlet')
# cfl.cfl('fashion_mnist', 64, 3, 30, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, 1, 30, split_para = 3, split_method = 'class')
global_rounds = 1
num_nodes = 20
local_steps = 10

def main(seeds):
    np.random.seed(seeds)
    for j,k in [(2, 'class'), (0.1, 'dirichlet'), (0.5, 'dirichlet'), (1, 'dirichlet')]:
        central.central('fashion_mnist', 64, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, global_rounds)
        central.central('cifar10', 64, CNNCifar, nn.CrossEntropyLoss, optim.SGD, global_rounds)
        fedavg.fedavg('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
        fedavg.fedavg('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.SGD, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
        # local.local('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
        # local.local('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.SGD, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
        # ditto.ditto('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, global_rounds, local_steps, 0.95, **{'split_para':j, 'split_method':k})
        # ditto.ditto('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.SGD, global_rounds, local_steps, 0.95, **{'split_para':j, 'split_method':k})
        # for K in [3,5,10]:  
        #     cfl.cfl('fashion_mnist', 64, K, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.SGD, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
        #     cfl.cfl('cifar10', 64, K, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.SGD, global_rounds, local_steps, **{'split_para':j, 'split_method':k})

# multiprocessing
import time
if __name__ == '__main__':
    start = time.perf_counter()
    # with Pool(2) as p:
    #     p.map(main, [0,1])
    main(0)
    main(1)
    print(time.perf_counter()-start, "seconds")