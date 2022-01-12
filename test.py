import os
from fedbase.baselines import central, fedavg, ditto, cfl, local, fedavg_finetune
from fedbase.model.model import CNNCifar, CNNMnist, CNNFashion_Mnist
from fedbase.nodes.node import node
from fedbase.utils.utils import unpack_args
from fedbase.utils.data_loader import data_process
import torch.optim as optim
import torch.nn as nn
from functools import partial
import numpy as np
import multiprocessing as mp
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
global_rounds = 1
num_nodes = 200
local_steps = 10
optimizer = partial(optim.SGD,lr=0.001,momentum=0.9)


@unpack_args
def main0(seeds, dataset_splited, model):
    np.random.seed(seeds)
    fedavg.run(dataset_splited, 64, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    fedavg_finetune.run(dataset_splited, 64, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 10)
    # local.run(dataset_splited, 64, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # ditto.run(dataset_splited, 64, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95)

@unpack_args
def main1(seeds, dataset_splited, model):
    np.random.seed(seeds)
    # fedavg.run(dataset_splited, 64, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # fedavg_finetune.run(dataset_splited, 64, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 10)
    local.run(dataset_splited, 64, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    ditto.run(dataset_splited, 64, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95)

@unpack_args
def main2(seeds, dataset_splited, model, K):
    np.random.seed(seeds)
    cfl.run(dataset_splited, 64, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)

@unpack_args
def main3(seeds, dataset_splited, K):
    np.random.seed(seeds)
    central.run(dataset_splited, 64, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds)
    
# multiprocessing
if __name__ == '__main__':
    multi_processes = 4
    seeds = 1
    # Run
    # main1((1,0.5,'dirichlet',3))
    start = time.perf_counter()
    mp.set_start_method('spawn')
    with mp.Pool(multi_processes) as p:
        # client_wise
        # p.map(main0, [(i, data_process(dataset).split_dataset(num_nodes, j, k), model) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) for j, k in zip([2, 0.1, 0.5, 1], ['class', 'dirichlet', 'dirichlet', 'dirichlet'])])
        # p.map(main1, [(i, j, k, K) for i in range(17, 17+seeds) for j,k in zip([2, 0.1, 0.5, 1], ['class', 'dirichlet', 'dirichlet', 'dirichlet']) for K in [3,5,10]])
        # group_wise
        p.map(main0, [(i, data_process(dataset).split_dataset_groupwise(n0,j0,k0,n1,j1,k1), model) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) \
            for n0,n1 in zip([5],[40]) for j0, k0 in zip([5, 0.1], ['class', 'dirichlet']) for j1, k1 in zip([10], ['dirichlet'])])
    print(time.perf_counter()-start, "seconds")