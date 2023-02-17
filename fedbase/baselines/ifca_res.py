from fedbase.utils.data_loader import data_process, log
from fedbase.nodes.node import node
from fedbase.server.server import server_class
from fedbase.baselines import fedavg
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from fedbase.model.model import CNNCifar, CNNMnist
import os
import sys
import inspect
from functools import partial

def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, warmup_rounds, global_rounds, local_steps, reg = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    train_splited, test_splited, split_para = dataset_splited
    # warm up
    model_g = fedavg.run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, warmup_rounds, local_steps, device, log_file = False)

    # initialize
    server = server_class(device)
    server.assign_model(model())

    server.model_g = model()
    server.model_g.load_state_dict(model_g.state_dict())
    server.model_g.to(device)

    nodes = [node(i, device) for i in range(num_nodes)]

    for i in range(num_nodes):
        # data
        # print(len(train_splited[i]), len(test_splited[i]))
        nodes[i].assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
        nodes[i].assign_test(DataLoader(test_splited[i], batch_size=batch_size, shuffle=False))
        # objective
        nodes[i].assign_objective(objective())

        nodes[i].model_g = model()
        nodes[i].model_g.load_state_dict(model_g.state_dict())
        nodes[i].model_g.to(device)

    del train_splited, test_splited

    # initialize K cluster model
    cluster_models = [model() for i in range(K)]

    # train!
    for i in range(global_rounds - warmup_rounds):
        print('-------------------Global round %d start-------------------' % (i))
        # assign client to cluster
        assignment = [[] for _ in range(K)]
        for i in range(num_nodes):
            m = 0
            for k in range(1, K):
                # if i <=5:
                #     print(nodes[i].local_train_acc(cluster_models[m]), nodes[i].local_train_acc(cluster_models[k]))
                # if nodes[i].local_train_loss(cluster_models[m])>=nodes[i].local_train_loss(cluster_models[k]):
                if nodes[i].local_train_acc(cluster_models[m])<=nodes[i].local_train_acc(cluster_models[k]):
                    m = k
            assignment[m].append(i)
            nodes[i].assign_model(cluster_models[m])
            nodes[i].assign_optim({'local': optimizer(nodes[i].model.parameters()),\
                'global': optimizer(nodes[i].model_g.parameters()),\
                    'all': optimizer(list(nodes[i].model.parameters())+list(nodes[i].model_g.parameters()))})

        # print(server.clustering)
        server.clustering['label'].append(assignment)
        print(assignment)
        print([len(assignment[i]) for i in range(len(assignment))])

        # local update
        # for j in range(num_nodes):
        #     nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_res, optimizer = nodes[j].optim['local'], \
        #         model_opt = nodes[j].model, model_fix = nodes[j].model_g))
        
        for j in range(num_nodes):
            nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_res, optimizer = nodes[j].optim['all'], \
                model_opt = nodes[j].model, model_fix = nodes[j].model_g))

        # server aggregation and distribution by cluster
        for k in range(K):
            if len(assignment[k])>0:
                server.aggregate(nodes, assignment[k])
                server.distribute(nodes, assignment[k])
                cluster_models[k].load_state_dict(server.model.state_dict())
        
        # for j in range(num_nodes):
        #     nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_res, optimizer = nodes[j].optim['global'],\
        #         model_opt = nodes[j].model_g, model_fix = cluster_models[[l for l in range(K) if j in assignment[l]][0]]))
            
        # aggregate model_g
        server.aggregate_model_g(nodes, list(range(num_nodes)))
        server.distribute_model_g(nodes, list(range(num_nodes)))

        # test accuracy
        for i in range(num_nodes):
            nodes[i].local_test()
        server.acc(nodes, list(range(num_nodes)))

    # log
    if reg:
        log(os.path.basename(__file__)[:-3] + '_' + str(K) + '_' + str(reg) + '_' + split_para, nodes, server)
    else:
        log(os.path.basename(__file__)[:-3] + '_' + str(K) + '_' + split_para, nodes, server)

    return cluster_models