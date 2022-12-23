from copy import deepcopy
import torch
from torch import linalg as LA
from fedbase.utils.model_utils import save_checkpoint, load_checkpoint
from fedbase.model.model import CNNMnist, MLP
from sklearn.metrics import accuracy_score, f1_score
from fedbase.utils.utils import unpack_args
from functools import partial
from statistics import mode

class node():
    def __init__(self, id, device):
        self.id = id
        self.test_metrics = []
        self.step = 0
        self.device = device
        self.grads = []
        # self.train_steps = 0

    def assign_train(self, data):
        self.train = data
        self.data_size = len(data)
    
    def assign_test(self,data):
        self.test = data

    def assign_model(self, model):
        self.model = model
        self.model.to(self.device)
        try:
            self.model = torch.compile(self.model)
        except:
            pass

    def assign_objective(self, objective):
        self.objective = objective

    def assign_optim(self, optim):
        self.optim = optim
    
    def local_update_steps(self, local_steps, train_single_step_func):
        # print(len(self.train), self.step)
        if len(self.train) - self.step > local_steps:
            for k, (inputs, labels) in enumerate(self.train):
                if k < self.step or k >= self.step + local_steps:
                    continue
                train_single_step_func(inputs, labels)
            self.step = self.step + local_steps
        else:
            for k, (inputs, labels) in enumerate(self.train):
                if k < self.step:
                    continue
                train_single_step_func(inputs, labels)
            for j in range((local_steps-len(self.train)+self.step)//len(self.train)):
                for k, (inputs, labels) in enumerate(self.train):
                    train_single_step_func(inputs, labels)
            for k, (inputs, labels) in enumerate(self.train):
                if k >=(local_steps-len(self.train)+self.step)%len(self.train):
                    continue
                train_single_step_func(inputs, labels)
            self.step = (local_steps-len(self.train)+self.step)%len(self.train)
        # torch.cuda.empty_cache()

    def train_single_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)
        # print(labels)
        # zero the parameter gradients
        self.model.zero_grad()
        # forward + backward + optimize
        outputs = self.model(inputs)
        # optim
        self.loss = self.objective(outputs, labels)
        self.loss.backward()
        # calculate accumulate gradients
        grads = torch.tensor([])
        for index, param in enumerate(self.model.parameters()):
            # param.grad = torch.tensor(grads[index])
            grads= torch.cat((grads, torch.flatten(param.grad).cpu()),0)
        self.grads.append(grads)
        
        self.optim.step()
        # self.train_steps+=1

    # for fedprox and ditto
    def train_single_step_fedprox(self, inputs, labels, reg_model, lam):
        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)
        # zero the parameter gradients
        self.model.zero_grad()
        # forward + backward + optimize
        outputs = self.model(inputs)
        # optim
        reg = 0
        for p,q in zip(self.model.parameters(), reg_model.parameters()):
            # reg += torch.square(LA.vector_norm((p-q),2))
            reg += torch.square(torch.norm((p-q),2))
        self.loss = self.objective(outputs, labels) + lam*reg/2
        # print(self.objective(outputs, labels))
        self.loss.backward()
        self.optim.step()
        # print('after', self.objective(self.model(inputs), labels))

    def local_update_epochs(self, local_epochs):
        # local_steps may be better!!
        running_loss = 0
        for j in range(local_epochs):
            for k, (inputs, labels) in enumerate(self.train):
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
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
        # torch.cuda.empty_cache()

    # for IFCA
    def local_train_loss(self, model):
        model.to(self.device)
        train_loss = 0
        for k, (inputs, labels) in enumerate(self.train):
            inputs = inputs.to(self.device)
            labels = torch.flatten(labels)
            labels = labels.to(self.device, dtype = torch.long)
            # forward
            outputs = model(inputs)
            train_loss += self.objective(outputs, labels)
            if k>=2:
                break
        # torch.cuda.empty_cache()
        return train_loss/(k+1)

    def local_test(self):
        predict_ts = torch.empty(0).to(self.device)
        label_ts = torch.empty(0).to(self.device)
        with torch.no_grad():
            for data in self.test:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                outputs = self.model(inputs)
                # print(outputs.data.dtype)
                _, predicted = torch.max(outputs.data, 1)
                predict_ts = torch.cat([predict_ts, predicted], 0)
                label_ts = torch.cat([label_ts, labels], 0)
        acc = accuracy_score(label_ts.cpu(), predict_ts.cpu())
        macro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='macro')
        # micro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='micro')
        # print('Accuracy, Macro F1, Micro F1 of Device %d on the %d test cases: %.2f %%, %.2f, %.2f' % (self.id, len(label_ts), acc*100, macro_f1, micro_f1))
        print('Accuracy, Macro F1 of Device %d on the %d test cases: %.2f %%, %.2f %%' % (self.id, len(label_ts), acc*100, macro_f1*100))
        self.test_metrics.append([acc, macro_f1])
        # torch.cuda.empty_cache()

    def local_ensemble_test(self, model_list, voting = 'soft'):
        predict_ts = torch.empty(0).to(self.device)
        label_ts = torch.empty(0).to(self.device)
        with torch.no_grad():
            for data in self.test:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                out_hard = []
                if voting == 'soft':
                    out = torch.zeros(self.model(inputs).data.shape).to(self.device)
                    for model in model_list:
                        outputs = model(inputs)
                        out = out + outputs.data/len(model_list)
                        _, predicted = torch.max(out, 1)
                elif voting == 'hard':
                    out_hard = []
                    for model in model_list:
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        out_hard.append(predicted)       
                    predicted = torch.tensor([mode([out_hard[i][j] for i in range(len(out_hard))]) for j in range(len(out_hard[0]))]).to(self.device)

                predict_ts = torch.cat([predict_ts, predicted], 0)
                label_ts = torch.cat([label_ts, labels], 0)
        acc = accuracy_score(label_ts.cpu(), predict_ts.cpu())
        macro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='macro')
        # micro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='micro')
        # print('Accuracy, Macro F1, Micro F1 of Device %d on the %d test cases: %.2f %%, %.2f, %.2f' % (self.id, len(label_ts), acc*100, macro_f1, micro_f1))
        print('Accuracy, Macro F1 of Device %d on the %d test cases: %.2f %%, %.2f %%' % (self.id, len(label_ts), acc*100, macro_f1*100))
        self.test_metrics.append([acc, macro_f1])


#     def model_representation(self, test_set, repr='output'):
#         self.model_repr = self.model(test_set)/len(test_set)
#         return self.model_repr

#     def log(self):
#         pass


# class nodes_control(node):
#     def __init__(self, id_list):
#         pass

#     def assign_one_model(self, model):
#         pass
