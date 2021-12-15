import os
from fedbase.baselines import central, fedavg, ditto, cfl, local
from fedbase.model.model import CNNCifar, CNNMnist, CNNFashion_Mnist
from fedbase.nodes.node import node
from fedbase.utils.utils import unpack_args
import torch.optim as optim
import torch.nn as nn
import numpy as np
import multiprocessing as mp
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

global_rounds = 1
num_nodes = 20
local_steps = 10
multi_processes = 4
seeds = 2

@unpack_args
def main(seeds, j, k):
    np.random.seed(seeds)
    # for j,k in [(2, 'class'), (0.1, 'dirichlet'), (0.5, 'dirichlet'), (1, 'dirichlet')]:
    central.central('fashion_mnist', 64, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds)
    central.central('cifar10', 64, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds)
    fedavg.fedavg('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    fedavg.fedavg('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    # local.local('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    # local.local('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    # ditto.ditto('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, 0.95, **{'split_para':j, 'split_method':k})
    # ditto.ditto('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, 0.95, **{'split_para':j, 'split_method':k})
    # for K in [3,5,10]:  
    #     cfl.cfl('fashion_mnist', 64, K, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    #     cfl.cfl('cifar10', 64, K, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})

# multiprocessing
if __name__ == '__main__':
    start = time.perf_counter()
    mp.set_start_method('spawn')
    with mp.Pool(multi_processes) as p:
        p.map(main, [(i, j, k) for i in range(seeds) for j,k in zip([2, 0.1, 0.5, 1], ['class', 'dirichlet', 'dirichlet', 'dirichlet'])])
    print(time.perf_counter()-start, "seconds")