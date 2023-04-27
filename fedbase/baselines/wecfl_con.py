from fedbase.utils.data_loader import data_process, log
from fedbase.utils.visualize import dimension_reduction
from fedbase.utils.tools import add_
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
import numpy as np

def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, warmup_rounds, tmp, mu, base, reg_lam = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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

    # initialize K cluster model
    cluster_models = [model().to(device) for i in range(K)]

    # train!
    # b_list = []
    # uu_list = []
    weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
    for i in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (i))

        # local update
        server.model.load_state_dict(server.aggregate([nodes[i].model for i in range(num_nodes)], weight_list))
        for j in range(num_nodes):
            if i == 0:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
            elif i < warmup_rounds:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_con, \
                    model_sim = cluster_models[nodes[j].label], model_all = cluster_models, tmp = tmp, mu = mu, base = None\
                        , reg_lam = reg_lam, reg_model = server.model))
            else:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_con, \
                    model_sim = cluster_models[nodes[j].label], model_all = cluster_models, tmp = tmp, mu = mu, base = base\
                        , reg_lam = reg_lam, reg_model = server.model))
                
        # # tsne or pca plot
        # # if i == global_rounds-1:
        # if i == i :
        #     cluster_data = torch.nn.utils.parameters_to_vector(nodes[0].model.parameters()).cpu().detach().numpy()[-1000:]
        #     for i in range(1, num_nodes):
        #         cluster_data = np.concatenate((cluster_data, torch.nn.utils.parameters_to_vector(nodes[i].model.parameters()).cpu().detach().numpy()[-1000:]), axis = 0)
        #     cluster_data = np.reshape(cluster_data, (num_nodes,int(len(cluster_data)/num_nodes)))
        #     cluster_label = server.clustering['label'][-1]
        #     # cluster_label = np.repeat(range(10),20)
        #     dimension_reduction(cluster_data, cluster_label, method= 'tsne')
        # plot B
        # print(server.calculate_B(nodes, range(20)))
        # B_list, u_list = server.calculate_B(nodes, range(20))
        # b_list.append(max(B_list))
        # uu_list.append(max(u_list))
        # print(b_list, uu_list)
        # for k in range(num_nodes):
        #     nodes[k].grads = []
        
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
    
    # log
    log(os.path.basename(__file__)[:-3] + add_(K) + add_(base) + add_(tmp) + add_(mu) + add_(reg_lam) + add_(split_para), nodes, server)

    return cluster_models
    