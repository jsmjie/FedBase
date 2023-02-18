import os
from fedbase.baselines import *
from fedbase.model.model import *
from fedbase.nodes.node import node
from fedbase.utils.utils import unpack_args
from fedbase.utils.data_loader import data_process
import torch
import torch.optim as optim
import torch.nn as nn
from functools import partial
import numpy as np
import multiprocessing as mp
import time
import torchvision.models as models

os.chdir(os.path.dirname(os.path.abspath(__file__)))
global_rounds = 100
num_nodes = 200
local_steps = 10
batch_size = 32
# optimizer = partial(optim.SGD,lr=0.001, momentum=0.9)
optimizer = partial(optim.SGD,lr=0.001)
# device = torch.device('cuda:2')
device = torch.device('cuda')

@unpack_args
def main0(seeds, dataset_splited, model):
    np.random.seed(seeds)
    central.run(dataset_splited, batch_size, model, nn.CrossEntropyLoss, optimizer, global_rounds, device = device)

@unpack_args
def main1(seeds, dataset_splited, model):
    np.random.seed(seeds)
    fedavg.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # fedavg_finetune.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 10, device = device)
    # local.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # ditto.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95, device = device)
    fedprox.run(dataset_splited, batch_size, num_nodes, model,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.1, device = device)

@unpack_args
def main2(seeds, dataset_splited, model, K):
    np.random.seed(seeds)
    fedavg_ensemble.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, K, device = device)
    fedprox_ensemble.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, K, device = device)
    # wecfl.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # fesem.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # ifca.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # wecfl.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95, device = device)
    # fesem.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95, device = device)
    # ifca.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95, device = device)
    
# multiprocessing
if __name__ == '__main__':
    np.random.seed(1)
    # data_process('cifar10').split_dataset(200,2,'class')
    # for i in range(3):
    #     data_process('cifar10').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet')
    # for i in range(1):
    #     np.random.seed(i)
    # data_process('fashion_mnist').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet', plot_show=True)
    # data_process('fashion_mnist').split_dataset_groupwise(10,3,'class',20,2,'class', plot_show=True)
    # data_process('medmnist_pathmnist').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet', plot_show=True)
    # data_process('medmnist_octmnist').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet', plot_show=True)
    # data_process('medmnist_tissuemnist').split_dataset_groupwise(10,3,'class',20,2,'class', plot_show=True)
    # data_process('fashion_mnist').split_dataset(200,0.1,'dirichlet', plot_show= True)
    # data_process('fashion_mnist').split_dataset(200,2,'class', plot_show= True)
    # print(a)
    # # data_process('fashion_mnist').split_dataset_groupwise(10,3,'class',20, 2,'class', plot_show=True)
    # # data_process('fashion_mnist').split_dataset(18,0.1,'dirichlet', plot_show= True)
    # # data_process('cifar10').split_dataset(200,2,'class', plot_show= True)
    # # data_process('medmnist_octmnist').split_dataset(200,2,'class', plot_show= True)
    # print(a)
    # data_process('medmnist_pathmnist').split_dataset(200,2,'class', plot_show= True)
    # ditto.run(data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',10,5,'class'), 16, 10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, 3, 10, 0.95)
    # fedprox.run(data_process('fashion_mnist').split_dataset_groupwise(10,6,'class',20,5,'class'), batch_size, num_nodes, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 1)
    # fedprox_ensemble.run(data_process('fashion_mnist').split_dataset_groupwise(10,6,'class',20,5,'class'), 16,10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 1, 3)
    # fedavg.run(data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',10,5,'class'), 16, 10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, 3, 10)
    # fedavg_ensemble.run(data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',10,5,'class'), 16,10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, 2, 10, 3)
    # wecfl.run(data_process('medmnist_pathmnist').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet'), batch_size, 10, num_nodes, CNNPath, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, reg=0)
    # wecfl.run(data_process('medmnist_tissuemnist').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet'), batch_size, 10, num_nodes, CNNTissue, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, reg=0)
    # wecfl.run(data_process('cifar10').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 10, 'dirichlet'), batch_size, 10, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, reg=0)
    # wecfl.run(data_process('cifar10').split_dataset_groupwise(5, 3, 'class', 40, 2, 'class'), batch_size, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # cfl_res.run(data_process('cifar10').split_dataset_groupwise(5, 3, 'class', 40, 2, 'class'), batch_size, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    ifca_res.run(data_process('cifar10').split_dataset(200, 2, 'class'), batch_size, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, 3, global_rounds, local_steps)
    # wecfl_con.run(data_process('fashion_mnist').split_dataset(200, 2, 'class'), batch_size, 5, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # wecfl.run(data_process('cifar10').split_dataset(200, 0.1, 'dirichlet'), batch_size, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # wecfl_res.run(data_process('cifar10').split_dataset_groupwise(5, 3, 'class', 40, 2, 'class'), batch_size, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, 2, global_rounds, local_steps)
    # ifca.run(data_process('fashion_mnist').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 5, 'dirichlet'), batch_size, 10, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # ifca.run(data_process('cifar10').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 5, 'dirichlet'), batch_size, 10, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # ifca.run(data_process('medmnist_octmnist').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 5, 'dirichlet', plot_show= True), batch_size, 10, num_nodes, oct_net, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    print(a)
    multi_processes = 2
    seeds = 1
    # Run
    start = time.perf_counter()
    mp.set_start_method('spawn')
    with mp.Pool(multi_processes) as p:
        # group_wise
        # p.map(main4, [(i, data_process(dataset).split_dataset_groupwise(n0,j0,k0,n1,j1,k1), model) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) \
        #     for n0,n1 in zip([5, 10],[40, 20]) for j0, k0, j1, k1 in zip([6, 0.1], ['class', 'dirichlet'], [5, 10], ['class', 'dirichlet'])])
        # p.map(main5, [(i, data_process(dataset).split_dataset_groupwise(n0,j0,k0,n1,j1,k1), model, K) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) \
        # for K,n0,n1 in zip([5, 10], [5, 10],[40, 20]) for j0, k0, j1, k1 in zip([6, 0.1], ['class', 'dirichlet'], [5, 10], ['class', 'dirichlet'])])
        # client_wise
        # p.map(main1, [(i, data_process(dataset).split_dataset(num_nodes, j, k), model) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) for j, k in zip([2, 0.1], ['class', 'dirichlet'])])
        p.map(main2, [(i, data_process(dataset).split_dataset(num_nodes, j, k), model, K) for i in range(27, 27+seeds) for dataset, model in zip(['medmnist_octmnist'],[oct_net]) for j, k in zip([2, 0.1], ['class', 'dirichlet']) for K in [3,5,10]])
        p.close()
    print(time.perf_counter()-start, "seconds")