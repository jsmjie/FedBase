from utils.data_loader import data_process, log
from nodes.node import node
from server.server import server
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
import numpy as np
import pickle
import datetime as d



os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(666)

# hyperparameter
dataset_name = 'Fashion_Mnist'
num_nodes = 50
global_rounds = 100
local_epochs = 3
# local_steps = 1
batch_size = 4
K = 10
# print(global_rounds)

if __name__ == '__main__':
    # split data
    dt = data_process(dataset_name)
    # train_splited,test_splited = dt.split_dataset(num_nodes, 0.1, method='dirichlet')
    train_splited,test_splited = dt.split_dataset(num_nodes, 2, method='class')
    # initiate model, server, nodes
    # model = MLP(784, 30, 10)
    global_model =  CNNFashion_Mnist()
    server = server()
    server.assign_model(global_model, device)

    nodes = [node() for i in range(num_nodes)]
    local_models = [CNNFashion_Mnist() for i in range(num_nodes)]
    local_loss = [nn.CrossEntropyLoss() for i in range(num_nodes)]

    for i in range(num_nodes):
        # instance
        # nodes[i] = node()
        nodes[i].id = i
        # data
        # print(len(train_splited[i]), len(test_splited[i]))
        nodes[i].assign_train(DataLoader(train_splited[i],
                            batch_size=batch_size, shuffle=True))
        nodes[i].assign_test(DataLoader(test_splited[i],batch_size=batch_size, shuffle=False))
        # model
        nodes[i].assign_model(local_models[i], device)
        # objective
        nodes[i].assign_objective(local_loss[i])
        # optim
        # nodes[i].assign_optim(optim.Adam(nodes[i].model.parameters()))
        nodes[i].assign_optim(optim.SGD(nodes[i].model.parameters(), lr=0.001, momentum=0.9))

    # initialize parameters to nodes
    server.distribution(nodes, list(range(num_nodes)))
    
    # train!
    for i in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (i))
        # local update
        for j in range(num_nodes):
            nodes[j].local_update(local_epochs, device)
            nodes[j].local_test(device)

        # server clustering
        server.weighted_clustering(nodes, list(range(num_nodes)), K)
        # print(a)
        
        # server aggregation and distribution
        for i in range(K):
            # print( [j for j in list(range(num_nodes)) if nodes[j].label==i])
            server.aggregation(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==i], device)
            server.distribution(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==i])
        server.acc(nodes, list(range(num_nodes)))

    # plot
    # server.clustering_plot()
    # save
    log(os.path.basename(__file__)[:-3], nodes, server)