from fedbase.utils.data_loader import data_process, log
from fedbase.nodes.node import node
from fedbase.utils.tools import add_
from fedbase.server.server import server_class
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from fedbase.model.model import CNNCifar, CNNMnist
import os
import sys
import inspect
from functools import partial

def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, \
    reg_lam = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), finetune=False, finetune_steps = None):
    # dt = data_process(dataset)
    # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited
    server = server_class(device)
    server.assign_model(model())

    nodes = [node(i, device) for i in range(num_nodes)]
    # local_models = [model() for i in range(num_nodes)]
    # local_loss = [objective() for i in range(num_nodes)]

    for i in range(num_nodes):
        # data
        # print(len(train_splited[i]), len(test_splited[i]))
        nodes[i].assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
        nodes[i].assign_test(DataLoader(test_splited[i], batch_size=batch_size, shuffle=False))
        # model
        nodes[i].assign_model(model())
        # objective
        nodes[i].assign_objective(objective())
        # optim
        nodes[i].assign_optim(optimizer(nodes[i].model.parameters()))
    
    del train_splited, test_splited

    # initialize parameters to nodes
    server.distribute([nodes[i].model for i in range(num_nodes)])
    weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]

    # initialize K cluster model
    cluster_models = [model() for i in range(K)]

    # train!
    for t in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (t))
        # local update
        for j in range(num_nodes):
            if not reg_lam or t == 0:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
            else:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedprox, reg_model = cluster_models[nodes[j].label], reg_lam= reg_lam))
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
            nodes[j].local_test()
        server.acc(nodes, weight_list)
    
    if not finetune:
        assign = [[i for i in range(num_nodes) if nodes[i].label == k] for k in range(K)]
        # log
        log(os.path.basename(__file__)[:-3] + add_(K) + add_(reg_lam) + add_(split_para), nodes, server)
        return cluster_models, assign
    else:
        if not finetune_steps:
            finetune_steps = local_steps
        # fine tune
        for j in range(num_nodes):
            if not reg_lam:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
            else:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedprox, reg_model = cluster_models[nodes[j].label], reg_lam= reg_lam))
            nodes[j].local_test()
        server.acc(nodes, weight_list)
        # log
        log(os.path.basename(__file__)[:-3] + add_('finetune') + add_(K) + add_(reg_lam) + add_(split_para), nodes, server)
        return [nodes[i].model for i in range(num_nodes)]