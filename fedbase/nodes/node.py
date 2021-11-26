from copy import deepcopy
import torch
from fedbase.utils.model_utils import save_checkpoint, load_checkpoint
from fedbase.model.model import CNNMnist, MLP

class node():
    def __init__(self):
        self.accuracy = []
        pass

    def id(self,id):
        self.id = id

    def assign_train(self, data):
        self.train = data
        self.data_size = len(data)
    
    def assign_test(self,data):
        self.test = data

    def assign_model(self, model, device):
        self.model = model
        self.model.to(device)

    def assign_objective(self, objective):
        self.objective = objective

    def assign_optim(self, optim):
        self.optim = optim

    def local_update(self, local_epochs, device):
        # local_steps may be better!!
        # if self.id == 3:
        #     print('before_local_update',self.model.state_dict()[list(self.model.state_dict().keys())[-1]])
        # running_loss = 0
        for j in range(local_epochs):
            for k, (inputs, labels) in enumerate(self.train):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                self.model.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                # optim
                self.loss = self.objective(outputs, labels)
                self.loss.backward()
                self.optim.step()
                # print
                # running_loss += loss.item()
                # if (k+1) % 100 == 0:    # print every 100 mini-batches
                #     print('[%d %d] node_%d loss: %.3f' %
                #           (j, k+1, self.id, running_loss/20))
                #     running_loss = 0
        # if self.id == 3:
        #     print('after_local_update',self.model.state_dict()[list(self.model.state_dict().keys())[-1]])
    def ditto_local_update(self, local_epochs, device, server, lam):
        for j in range(local_epochs):
            for k, (inputs, labels) in enumerate(self.train):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                self.model.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                # optim
                reg = 0
                for p,q in zip(self.model.parameters(),server.model.parameters()):
                    reg += torch.norm((p-q),2)
                self.loss = self.objective(outputs, labels) + lam*reg/2
                self.loss.backward()
                self.optim.step()

    def local_test(self, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of Device %d on the %d test cases: %.2f %%' % (self.id, total, 100*correct / total))
        self.accuracy.append(correct / total)

    def model_representation(self, test_set, repr='output'):
        self.model_repr = self.model(test_set)/len(test_set)
        return self.model_repr

    def log(self):
        pass


class nodes_control(node):
    def __init__(self, id_list):
        pass

    def assign_one_model(self, model):
        pass
