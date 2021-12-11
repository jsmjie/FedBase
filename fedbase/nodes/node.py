from copy import deepcopy
import torch
from fedbase.utils.model_utils import save_checkpoint, load_checkpoint
from fedbase.model.model import CNNMnist, MLP

class node():
    def __init__(self,id):
        self.id = id
        self.accuracy = []
        self.step = 0
        # self.train_steps = 0

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
    
    def local_update_steps(self, local_steps, device):
        # print(len(self.train), self.step)
        if len(self.train) - self.step > local_steps:
            for k, (inputs, labels) in enumerate(self.train):
                if k < self.step or k >= self.step + local_steps:
                    continue
                self.train_single_step(inputs, labels, device)
            self.step = self.step + local_steps
        else:
            for k, (inputs, labels) in enumerate(self.train):
                if k < self.step:
                    continue
                self.train_single_step(inputs, labels, device)
            for j in range((local_steps-len(self.train)+self.step)//len(self.train)):
                for k, (inputs, labels) in enumerate(self.train):
                    self.train_single_step(inputs, labels, device)
            for k, (inputs, labels) in enumerate(self.train):
                if k >=(local_steps-len(self.train)+self.step)%len(self.train):
                    continue
                self.train_single_step(inputs, labels, device)
            self.step = (local_steps-len(self.train)+self.step)%len(self.train)
        # print(len(self.train), self.step)
        # print(self.train_steps)

    def train_single_step(self, inputs, labels, device):
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
        # self.train_steps+=1

    def local_update_epochs(self, local_epochs, device):
        # local_steps may be better!!
        running_loss = 0
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
                running_loss += self.loss.item()
                if (k+1) % 100 == 0:    # print every 100 mini-batches
                    print('[%d %d] node_%d loss: %.3f' %
                          (j, k+1, self.id, running_loss/20))
                    running_loss = 0

    def local_update_ditto(self, local_steps, device, server, lam):
        # print(len(self.train), self.step)
        if len(self.train) - self.step > local_steps:
            for k, (inputs, labels) in enumerate(self.train):
                if k < self.step or k >= self.step + local_steps:
                    continue
                self.train_single_step_ditto(inputs, labels, device, server, lam)
            self.step = self.step + local_steps
        else:
            for k, (inputs, labels) in enumerate(self.train):
                if k < self.step:
                    continue
                self.train_single_step_ditto(inputs, labels, device, server, lam)
            for j in range((local_steps-len(self.train)+self.step)//len(self.train)):
                for k, (inputs, labels) in enumerate(self.train):
                    self.train_single_step_ditto(inputs, labels, device, server, lam)
            for k, (inputs, labels) in enumerate(self.train):
                if k >=(local_steps-len(self.train)+self.step)%len(self.train):
                    continue
                self.train_single_step_ditto(inputs, labels, device, server, lam)
            self.step = (local_steps-len(self.train)+self.step)%len(self.train)

    def train_single_step_ditto(self, inputs, labels, device, server, lam):
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
