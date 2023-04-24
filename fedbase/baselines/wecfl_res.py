from fedbase.utils.data_loader import data_process, log
from fedbase.utils.visualize import dimension_reduction
from fedbase.utils.tools import add_
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
        nodes[i].assign_optim({'local_0': optimizer(nodes[i].model.parameters()),\
                'local_1': optimizer(nodes[i].model_g.parameters()),\
                    'all': optimizer(list(nodes[i].model.parameters())+list(nodes[i].model_g.parameters()))})
    
    del train_splited, test_splited

    # initialize K cluster model
    cluster_models = [model() for i in range(K)]

    # initialize clustering and distribute
    server.weighted_clustering(nodes, list(range(num_nodes)), K)
    for j in range(K):
        assign_ls = [i for i in list(range(num_nodes)) if nodes[i].label==j]
        weight_ls = [nodes[i].data_size/sum([nodes[i].data_size for i in assign_ls]) for i in assign_ls]
        model_k = server.aggregate([nodes[i].model for i in assign_ls], weight_ls)
        server.distribute([nodes[i].model for i in assign_ls], model_k)
        cluster_models[j].load_state_dict(model_k)

    # train!
    for i in range(global_rounds - warmup_rounds):
        print('-------------------Global round %d start-------------------' % (i))
        # update model_g
        for j in range(num_nodes):
            nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_res, optimizer = nodes[j].optim['local_1'],\
                model_opt = nodes[j].model_g, model_fix = nodes[j].model))

        # aggregate and distribute model_g
        weight_all = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
        server.model.load_state_dict(server.aggregate([nodes[i].model_g for i in range(num_nodes)], weight_all))
        server.distribute([nodes[i].model_g for i in range(num_nodes)])

        # update model
        for j in range(num_nodes):        
            nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_res, optimizer = nodes[j].optim['local_0'], \
                model_opt = nodes[j].model, model_fix = nodes[j].model_g))

        # server clustering
        server.weighted_clustering(nodes, list(range(num_nodes)), K)

        # server aggregation and distribution by cluster
        for j in range(K):
            assign_ls = [i for i in list(range(num_nodes)) if nodes[i].label==j]
            weight_ls = [nodes[i].data_size/sum([nodes[i].data_size for i in assign_ls]) for i in assign_ls]
            model_k = server.aggregate([nodes[i].model for i in assign_ls], weight_ls)
            server.distribute([nodes[i].model for i in assign_ls], model_k)
            cluster_models[j].load_state_dict(model_k)
        
        # test accuracy
        for j in range(num_nodes):
             nodes[j].local_test(model_res = nodes[j].model_g)
        server.acc(nodes, weight_list)
    
    # log
    log(os.path.basename(__file__)[:-3] + add_(K) + add_(reg) + add_(split_para), nodes, server)

    return cluster_models
    