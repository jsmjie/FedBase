import os
from fedbase.baselines import *
from fedbase.model.model import *
from fedbase.nodes.node import node
from fedbase.server.server import server_class
from fedbase.utils.tools import unpack_args
from fedbase.utils.data_loader import data_process
import torch
import torch.optim as optim
import torch.nn as nn
from functools import partial
import numpy as np
import multiprocessing as mp
import time
import torchvision.models as models
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.abspath(__file__))) # set the current path as the working directory
global_rounds = 1
num_nodes = 200
local_steps = 10
batch_size = 32
# optimizer = partial(optim.SGD,lr=0.001, momentum=0.9)
optimizer = partial(optim.SGD,lr=0.001)
# device = torch.device('cuda:2')
device = torch.device('cuda')  # Use GPU if available

# dataset_splited = data_process('cifar10').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 10, 'dirichlet')
dataset_splited = data_process('cifar10').split_dataset_groupwise(10, 3, 'class', 20, 2, 'class')
train_splited, test_splited, split_para = dataset_splited
model = CNNCifar
objective = nn.CrossEntropyLoss
K = 5

model_g = fedavg.run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, global_rounds, local_steps, device, log_file = True)
# model_local = local.run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, global_rounds, local_steps, device, log_file = True)
# model_cluster = ifca.run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, device)
model_cluster, assign = wecfl.run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, device = device)
# print(assign)

# test
 # initialize
server = server_class(device)
server.assign_model(model())
server.model_g = model()

nodes = [node(i, device) for i in range(num_nodes)]
for i in range(num_nodes):
    # data
    # print(len(train_splited[i]), len(test_splited[i]))
    nodes[i].assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
    nodes[i].assign_test(DataLoader(test_splited[i], batch_size=batch_size, shuffle=False))
    # objective
    nodes[i].assign_objective(objective())
    # model
    nodes[i].assign_model(model_g)

weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
# test accuracy
for i in range(num_nodes):
        nodes[i].local_test(model_res = model_cluster[[j for j in range(K) if i in assign[j]][0]])
server.acc(nodes, weight_list)


