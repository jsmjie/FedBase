import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ChainDataset, ConcatDataset
import torch
from torch._utils import _accumulate
import matplotlib.pyplot as plt
import matplotlib as mpl
from fedbase.utils.utils import get_targets
from fedbase.utils import femnist
import os
import pickle
import datetime as d
import math
import pandas as pd
from pathlib import Path
from collections import Counter

class data_process:
    def __init__(self, dataset_name):
        dir ='./data/'
        self.dataset_name = dataset_name
        if dataset_name == 'mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            self.train_dataset = datasets.MNIST(
                dir+dataset_name, train=True, download=True, transform=apply_transform)
            self.test_dataset = datasets.MNIST(
                dir+dataset_name, train=False, download=True, transform=apply_transform)
        elif dataset_name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.train_dataset = datasets.CIFAR10(
                dir+dataset_name, train=True, download=True, transform=transform)
            self.test_dataset = datasets.CIFAR10(
                dir+dataset_name, train=False, download=True, transform=transform)
        elif dataset_name == 'femnist':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = femnist.FEMNIST(dir+dataset_name, train=True, download=False,
                                            transform=apply_transform)
            test_dataset = femnist.FEMNIST(dir+dataset_name, train=False, download=False,
                                           transform=apply_transform)
        elif dataset_name == 'fashion_mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
            self.train_dataset = datasets.FashionMNIST(
                dir+dataset_name, train=True, download=True, transform=apply_transform)
            self.test_dataset = datasets.FashionMNIST(
                dir+dataset_name, train=False, download=True, transform=apply_transform)

        # show image
        # batch_size = 4
        # trainloader = DataLoader(self.train_dataset, batch_size=batch_size,
        #                                   shuffle=True, num_workers=2)
        # classes = ('plane', 'car', 'bird', 'cat',
        #    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # def imshow(img):
        #     img = img / 2 + 0.5     # unnormalize
        #     npimg = img.numpy()
        #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #     plt.show()

        # # get some random training images
        # dataiter = iter(trainloader)
        # images, labels = dataiter.next()

        # # show images
        # imshow(torchvision.utils.make_grid(images))
        # # print labels
        # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    def split_dataset(self, num_nodes, alpha, method='dirichlet', train_dataset = None, test_dataset = None):
        train_dataset = self.train_dataset if train_dataset is None else train_dataset
        test_dataset = self.test_dataset if test_dataset is None else test_dataset
        train_targets, test_targets = get_targets(train_dataset), get_targets(test_dataset)
        if num_nodes == 1:
            return train_dataset, test_dataset
        else:
            if method == 'iid':
                train_lens_list = [int(len(train_dataset)/num_nodes) for i in range(num_nodes)]
                test_lens_list = [int(len(test_dataset)/num_nodes) for i in range(num_nodes)]
                train_splited, test_splited = random_split(Subset(train_dataset, torch.arange(sum(train_lens_list))), train_lens_list), random_split(Subset(test_dataset, torch.arange(sum(test_lens_list))), test_lens_list)
                # plot
                labels = torch.unique(train_targets)
                self.plot_split(labels, train_splited)
                return train_splited, test_splited
            else:
                labels, train_label_size = torch.unique(train_targets, return_counts=True)
                _, test_label_size = torch.unique(test_targets, return_counts=True)
                # print(train_label_size, test_label_size)
                l_train = train_label_size.reshape(
                    len(labels), 1).repeat(1, num_nodes)
                l_test = test_label_size.reshape(
                    len(labels), 1).repeat(1, num_nodes)
                # print(l_test)
                if method == 'dirichlet':
                    p = torch.tensor(np.round(np.random.dirichlet(np.repeat(alpha, num_nodes), len(labels)), round(math.log(len(test_dataset)/len(labels),10))))
                    # print(torch.sum(p,axis=1))
                elif method == 'class':
                    p = np.zeros((len(labels), 1))
                    J = np.random.choice(len(labels), alpha, replace=False)
                    p[J] = 1
                    for k in range(1, num_nodes):
                        x = np.zeros((len(labels), 1))
                        J = np.random.choice(len(labels), alpha, replace=False)
                        x[J] = 1
                        p = np.concatenate((p, x), axis=1)
                    p = p / np.repeat((p.sum(axis=1)+10**-10).reshape(len(labels), 1), num_nodes, axis=1)
                # print(p.sum(axis=1),p)
                train_size = torch.round(l_train*p).int()
                test_size = torch.round(l_test*p).int()
                train_splited = []
                test_splited = []
                train_label_index = []
                test_label_index = []
                for j in range(len(labels)):
                    train_label_index.append([(train_targets== labels[j]).nonzero(as_tuple=True)[
                                             0][offset-length:offset] for offset, length in zip(_accumulate(train_size[j, :]), train_size[j, :])])
                    test_label_index.append([(test_targets== labels[j]).nonzero(as_tuple=True)[
                                            0][offset-length:offset] for offset, length in zip(_accumulate(test_size[j, :]), test_size[j, :])])
                # how to deal with 0?
                for i in range(num_nodes):
                    if len(ConcatDataset([Subset(test_dataset, test_label_index[j][i]) for j in range(len(labels))]))>0:
                        train_splited.append(ConcatDataset(
                            [Subset(train_dataset, train_label_index[j][i]) for j in range(len(labels))]))
                        test_splited.append(ConcatDataset(
                            [Subset(test_dataset, test_label_index[j][i]) for j in range(len(labels))]))
                if len(test_splited)<num_nodes:
                    random_index = np.random.choice(range(len(test_splited)), num_nodes-len(test_splited), replace=False)
                    train_splited = train_splited + [train_splited[i] for i in range(len(train_splited)) if i in random_index]           
                    test_splited = test_splited + [test_splited[i] for i in range(len(test_splited)) if i in random_index]  

                self.plot_split(labels, train_splited)    
                return train_splited, test_splited, self.dataset_name +'_'+ str(num_nodes)+'_'+ str(alpha)+'_'+ str(method)
        
    def split_dataset_groupwise(self, num0, alpha0, method0, num1, alpha1, method1, train_dataset = None, test_dataset = None):
        train_dataset = self.train_dataset if train_dataset is None else train_dataset
        test_dataset = self.test_dataset if test_dataset is None else test_dataset
        train_targets = get_targets(train_dataset)
        train_splited = []
        test_splited = []
        train_splited_0, test_splited_0, _ = self.split_dataset(num0, alpha0, method0)
        for i in range(num0):
            train_tmp, test_tmp, _ = self.split_dataset(num1, alpha1, method1, train_dataset=train_splited_0[i], test_dataset=test_splited_0[i])
            train_splited += train_tmp
            test_splited += test_tmp
        #plot
        labels = torch.unique(train_targets)
        self.plot_split(labels, train_splited)
        return train_splited, test_splited, self.dataset_name +'_'+ str(num0)+'_'+ str(alpha0)+'_'+ str(method0)+'_'+ str(num1)+'_'+ str(alpha1)+'_'+ str(method1)

    def plot_split(self, labels, train_splited):
        return None
        train_size = []
        for x in train_splited:
            tmp = []
            train_classes = [int(label) for _, label in x]
            for i in range(len(labels)):
                tmp.append(train_classes.count(labels[i]))
            # train_targets = get_targets(x)
            # labels, train_label_size = torch.unique(train_targets, return_counts=True)
            train_size.append(tmp)
        train_size = torch.tensor(train_size).T
        # plot
        for i in range(len(labels)):
            plt.barh(range(len(train_splited)), train_size[i, :], left=torch.sum(
                train_size[:i], 0), label=str(int(labels[i])))
        # plt.title("Data distribution of dataset")
        plt.legend()
        mpl.rcParams['figure.dpi'] = 300
        plt.show()

def log(file_name, nodes, server):
    local_file = './log/' + file_name + "_" + d.datetime.now().strftime("%m%d_%H%M%S")+'_'+str(np.random.choice(10**3)) + ".pkl"
    log = {}
    log['node'] = {}
    for i in range(len(nodes)):
        log['node'][str(i)] = nodes[i].accuracy
    try:
        log['server'] = server.accuracy
        log['clustering'] = server.clustering
    except:
        print('No server')
    # pd.to_pickle(log, local_file)
    Path(local_file).parent.mkdir(parents=True, exist_ok=True)
    with  open(local_file, 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # read
    if os.path.exists(local_file):
        with open(local_file, 'rb') as f:
            log = pickle.load(f)
            print(log)

# dt = data_process('mnist')
# # dt.split_dataset(50, 2, method='class')
# dt.split_dataset(10, 0.1)


# def data_split(dataset, num_nodes, type='iid'):
#     if type == 'iid':
#         lens_list = [int(len(dataset)/num_nodes) for i in range(num_nodes-1)]
#         lens_list.append(len(dataset)-sum(lens_list))
#         return random_split(dataset, lens_list)
# def data_split_noniid(dataset, num_nodes, num_category):


# def data_split(dataset, num, type='iid'):
#     num_items = int(len(dataset)/num)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label



