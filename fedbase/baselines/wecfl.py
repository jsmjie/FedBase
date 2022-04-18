from fedbase.utils.data_loader import data_process, log
from fedbase.nodes.node import node
from fedbase.server.server import server_class
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from fedbase.model.model import CNNCifar, CNNMnist
import os
import sys
import inspect
from functools import partial

def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, reg = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # dt = data_process(dataset)
    # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited
    server = server_class(device)
    server.assign_model(model())

    nodes = [node(i, device) for i in range(num_nodes)]
    local_models = [model() for i in range(num_nodes)]
    local_loss = [objective() for i in range(num_nodes)]

    for i in range(num_nodes):
        # data
        # print(len(train_splited[i]), len(test_splited[i]))
        nodes[i].assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
        nodes[i].assign_test(DataLoader(test_splited[i],batch_size=batch_size, shuffle=False))
        # model
        nodes[i].assign_model(local_models[i])
        # objective
        nodes[i].assign_objective(local_loss[i])
        # optim
        nodes[i].assign_optim(optimizer(nodes[i].model.parameters()))

    # initialize parameters to nodes
    server.distribute(nodes, list(range(num_nodes)))

    # train!
    for i in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (i))
        # local update
        for j in range(num_nodes):
            if not reg:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
            else:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedprox, reg_model = server.aggregate(nodes, list(range(num_nodes))), lam= reg))
        # server clustering
        server.weighted_clustering(nodes, list(range(num_nodes)), K)
        # server aggregation and distribution by cluster
        for i in range(K):
            server.aggregate(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==i])
            server.distribute(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==i])
        # test accuracy
        for j in range(num_nodes):
            nodes[j].local_test()
        server.acc(nodes, list(range(num_nodes)))
    
    # log
    log(os.path.basename(__file__)[:-3] + '_' + str(K) + '_' + str(reg) + '_' + split_para, nodes, server)