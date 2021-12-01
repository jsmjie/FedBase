# from nodes.node import node
from copy import deepcopy
import torch
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.plotting import parallel_coordinates

class server_class():
    def __init__(self):
        self.accuracy = []
        self.clustering = []

    def assign_model(self, model, device):
        self.model = model
        self.model.to(device)

    def aggregate(self, nodes, idlist, device, weight_type='data_size'):
        # aggregated_weights = deepcopy(self.model.state_dict())
        aggregated_weights = self.model.state_dict()
        for j in aggregated_weights.keys():
            aggregated_weights[j] = torch.zeros(aggregated_weights[j].shape).to(device)
        # print(aggregated_weights[j])
        global_accuracy = 0
        sum_size = sum([nodes[i].data_size for i in idlist])
        for i in idlist:
            if weight_type == 'equal':
                weight = 1/len(idlist)
            elif weight_type == 'data_size':
                weight = nodes[i].data_size/sum_size
            for j in nodes[i].model.state_dict().keys():
                aggregated_weights[j] += nodes[i].model.state_dict()[j]*weight
            # print(aggregated_weights[j])
            global_accuracy += weight*nodes[i].accuracy[-1]
        print('Accuracy is %.2f %%' % (100*global_accuracy))
        # self.accuracy.append(global_accuracy)
        self.model.load_state_dict(aggregated_weights)
        # print('after_aggregation',self.model.state_dict()[list(self.model.state_dict().keys())[-1]])
    
    def acc(self, nodes, idlist, weight_type='data_size'):
        global_accuracy = 0
        sum_size = sum([nodes[i].data_size for i in idlist])
        for i in idlist:
            if weight_type == 'equal':
                weight = 1/len(idlist)
            elif weight_type == 'data_size':
                weight = nodes[i].data_size/sum_size
            global_accuracy += weight*nodes[i].accuracy[-1]
        print('GLOBAL Accuracy is %.2f %%' % (100*global_accuracy))
        self.accuracy.append(global_accuracy)
               
    def distribute(self, nodes, idlist):
        for i in idlist:
            nodes[i].model.load_state_dict(self.model.state_dict())

    def test(self, test_loader, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy on the %d test cases: %.2f %%' % (total, 100*correct / total))

    def model_similarity(model_repr_1, model_repr_2, repr='output'):
        if repr == 'output':
            self.similarity = (log(model_repr_1)-log(model_repr_2)).sum(axis=1).abs()

    def weighted_clustering(self, nodes, idlist, K, weight_type='data_size'):
        weight = []
        X = []
        sum_size = sum([nodes[i].data_size for i in idlist])
        # print(list(nodes[0].model.state_dict().keys()))
        for i in idlist:
            if weight_type == 'equal':
                weight.append(1/len(idlist))
            elif weight_type == 'data_size':
                weight.append(nodes[i].data_size/sum_size)
            X.append(np.array(torch.flatten(nodes[i].model.state_dict()[list(nodes[i].model.state_dict().keys())[-2]]).cpu()))
        # print(X, np.array(X).shape)
        kmeans = KMeans(n_clusters=K).fit(np.asarray(X), sample_weight= weight)
        labels = kmeans.labels_
        print(labels)
        for i in idlist:
            nodes[i].label = labels[i]
        self.clustering.append(labels)
    
    def clustering_plot(self):
        # print(self.clustering)
        # self.clustering =[[1,1,2,2,3,3],[1,1,1,2,2,2],[1, 1, 1, 2, 2, 2],[1, 1, 1, 2, 2, 2]]
        col = [str(i) for i in range(len(self.clustering))]+['id']
        self.clustering.append(list(range(len(self.clustering[0]))))
        data= pd.DataFrame(np.array(self.clustering).T,columns= col)
        for i in data.columns:
            data[i]=data[i].apply(lambda x: str(x))
        # Make the plot
        parallel_coordinates(data, 'id', colormap=plt.get_cmap("Set2"))
        plt.show()
