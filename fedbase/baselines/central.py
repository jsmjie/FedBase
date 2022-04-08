from fedbase.utils.data_loader import data_process, log
from fedbase.nodes.node import node
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os


def run(dataset, batch_size, model, objective, optimizer, global_rounds, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    dt = dataset
    trainset,testset = dt.train_dataset, dt.test_dataset
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

    nodes0 = node(0, device)
    nodes0.assign_train(trainloader)
    nodes0.assign_test(testloader)
    nodes0.assign_model(model())
    nodes0.assign_objective(objective())
    nodes0.assign_optim(optimizer(nodes0.model.parameters()))

    print('-------------------start-------------------')
    for i in range(global_rounds):
        nodes0.local_update_epochs(1)
        nodes0.local_test(device)

    # log
    log(os.path.basename(__file__)[:-3]+ '_' + str(dataset), [nodes0], server={})