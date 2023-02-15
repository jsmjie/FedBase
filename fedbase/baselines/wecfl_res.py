from fedbase.utils.data_loader import data_process, log
from fedbase.utils.visualize import dimension_reduction
from fedbase.nodes.node import node
from fedbase.server.server import server_class
from fedbase.baselines import local
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from fedbase.model.model import CNNCifar, CNNMnist
import os
import sys
import inspect
from functools import partial
import numpy as np

def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, warmup_rounds, global_rounds, local_steps, reg = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    train_splited, test_splited, split_para = dataset_splited
    # warmup
    local_models_warmup = local.run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, warmup_rounds, local_steps, device = device, log_file=False)

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
        # model
        nodes[i].assign_model(local_models_warmup[i])
        nodes[i].model_g = model()
        nodes[i].model_g.to(device)
        # objective
        nodes[i].assign_objective(objective())
        # optim
        nodes[i].assign_optim({'local': optimizer(nodes[i].model.parameters()),\
                'global': optimizer(nodes[i].model_g.parameters()),\
                    'all': optimizer(list(nodes[i].model.parameters())+list(nodes[i].model_g.parameters()))})
    
    del train_splited, test_splited

    # initialize K cluster model
    cluster_models = [model() for i in range(K)]

    # initialize clustering and distribute
    server.weighted_clustering(nodes, list(range(num_nodes)), K)
    for i in range(K):
        server.aggregate(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==i])
        server.distribute(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==i])
        cluster_models[i].load_state_dict(server.model.state_dict())

    # train!
    for i in range(global_rounds - warmup_rounds):
        print('-------------------Global round %d start-------------------' % (i))
        # local update
        # for j in range(num_nodes):
        #     nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_res, optimizer = nodes[j].optim['global'],\
        #         model_opt = nodes[j].model_g, model_fix = cluster_models[[l for l in range(K) if j in assignment[l]][0]]))
        #     nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_res, optimizer = nodes[j].optim['local'], \
        #         model_opt = nodes[j].model, model_fix = nodes[j].model_g))

        for j in range(num_nodes):
            nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_res, optimizer = nodes[j].optim['all'], \
                model_opt = nodes[j].model, model_fix = nodes[j].model_g))

        # server clustering
        server.weighted_clustering(nodes, list(range(num_nodes)), K)

        # server aggregation and distribution by cluster
        for k in range(K):
            server.aggregate(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==k])
            server.distribute(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==k])
            cluster_models[k].load_state_dict(server.model.state_dict())
        
        # aggregate model_g
        server.aggregate_model_g(nodes, list(range(num_nodes)))
        server.distribute_model_g(nodes, list(range(num_nodes)))
        
        # test accuracy
        for j in range(num_nodes):
            nodes[j].local_test()
        server.acc(nodes, list(range(num_nodes)))
    
    # log
    if not reg:
        log(os.path.basename(__file__)[:-3] + '_' + str(K)  + '_' + split_para, nodes, server)
    else:
        log(os.path.basename(__file__)[:-3] + '_' + str(K) + '_' + str(reg) + '_' + split_para, nodes, server)

    return cluster_models
    