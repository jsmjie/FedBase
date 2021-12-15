import os
from fedbase.baselines import central, fedavg, ditto, cfl, local, fedavg_finetune
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
    central.run('fashion_mnist', 64, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds)
    central.run('cifar10', 64, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds)
    fedavg.run('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    fedavg.run('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    fedavg_finetune.run('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, 30, **{'split_para':j, 'split_method':k})
    fedavg_finetune.run('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, 30,  **{'split_para':j, 'split_method':k})
    local.run('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    local.run('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    ditto.run('fashion_mnist', 64, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, 0.95, **{'split_para':j, 'split_method':k})
    ditto.run('cifar10', 64, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, 0.95, **{'split_para':j, 'split_method':k})
 
@unpack_args
def main1(seeds, j, k, K):
    np.random.seed(seeds)
    cfl.run('fashion_mnist', 64, K, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})
    cfl.run('cifar10', 64, K, num_nodes, CNNCifar, nn.CrossEntropyLoss, optim.Adam, global_rounds, local_steps, **{'split_para':j, 'split_method':k})

# multiprocessing
if __name__ == '__main__':
    start = time.perf_counter()
    mp.set_start_method('spawn')
    with mp.Pool(multi_processes) as p:
        p.map(main, [(i, j, k) for i in range(27, 27+seeds) for j,k in zip([2, 0.1, 0.5, 1], ['class', 'dirichlet', 'dirichlet', 'dirichlet'])])
        # p.map(main1, [(i, j, k, K) for i in range(17, 17+seeds) for j,k in zip([2, 0.1, 0.5, 1], ['class', 'dirichlet', 'dirichlet', 'dirichlet']) for K in [3,5,10]])
    print(time.perf_counter()-start, "seconds")