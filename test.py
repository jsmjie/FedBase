import os
from fedbase.baselines import central, fedavg, ditto, wecfl, local, fedavg_finetune, fedprox, ifca
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
global_rounds = 100
num_nodes = 200
local_steps = 10
batch_size = 64
optimizer = partial(optim.SGD,lr=0.001, momentum=0.9)


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
    wecfl.run(dataset_splited, 64, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)

@unpack_args
def main3(seeds, dataset_splited, K):
    np.random.seed(seeds)
    central.run(dataset_splited, 64, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds)

@unpack_args
def main4(seed, dataset_splited, model):
    np.random.seed(seed)
    # fedavg.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # fedavg_finetune.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 10)
    fedprox.run(dataset_splited, batch_size, num_nodes, model,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.1)

@unpack_args
def main5(seed, dataset_splited, model, K):
    np.random.seed(seed)
    ifca.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    
# multiprocessing
if __name__ == '__main__':
    # data_process('cifar10').split_dataset(200,2,'class')
    # for i in range(3):
    #     data_process('cifar10').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet')
    # data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',40,5,'class')
    # data_process('cifar10').split_dataset(200,0.1,'dirichlet')
    # data_process('cifar10').split_dataset_groupwise(3,0.1,'dirichlet',10,10,'dirichlet')
    # ditto.run(data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',10,5,'class'), 16, 10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, 3, 10, 0.95)
    # fedprox.run(data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',10,5,'class'), 16, 10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, 3, 10, 1)
    # fedavg.run(data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',10,5,'class'), 16, 10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, 3, 10)
    # wecfl.run(data_process('cifar10').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 10, 'dirichlet'), 64, 10, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # wecfl.run(data_process('cifar10').split_dataset_groupwise(5, 3, 'class', 40, 2, 'class'), 64, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # ifca.run(data_process('fashion_mnist').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 5, 'dirichlet'), 64, 10, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    fedprox.run(data_process('cifar10').split_dataset_groupwise(10,6,'class',20,5,'class'), 64, num_nodes, CNNCifar,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 1)
    print(a)
    multi_processes = 4
    seeds = 1
    # Run
    # main1((1,0.5,'dirichlet',3))
    start = time.perf_counter()
    mp.set_start_method('spawn')
    with mp.Pool(multi_processes) as p:
        # group_wise
        p.map(main4, [(i, data_process(dataset).split_dataset_groupwise(n0,j0,k0,n1,j1,k1), model) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) \
            for n0,n1 in zip([5, 10],[40, 20]) for j0, k0, j1, k1 in zip([6, 0.1], ['class', 'dirichlet'], [5, 10], ['class', 'dirichlet'])])
        # p.map(main5, [(i, data_process(da
        # 5, 10], [40, 20]) for j0, k0, j1, k1 in zip([6, 0.1], ['class', 'dirichlet'], [5, 10], ['class', 'dirichlet'])])
        # # client_wise
        # p.map(main0, [(i, data_process(dataset).split_dataset(num_nodes, j, k), model) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) for j, k in zip([2, 0.1], ['class', 'dirichlet'])])
        # p.map(main1, [(i, data_process(dataset).split_dataset(num_nodes, j, k), model, K) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) for j, k in zip([2, 0.1], ['class', 'dirichlet']) for K in [3,5,10]])
    print(time.perf_counter()-start, "seconds")

    