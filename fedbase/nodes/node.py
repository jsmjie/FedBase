import torch
from torch import linalg as LA
from fedbase.utils.model_utils import save_checkpoint, load_checkpoint
from fedbase.model.model import CNNMnist, MLP
from sklearn.metrics import accuracy_score, f1_score
from fedbase.utils.tools import unpack_args
from functools import partial
from statistics import mode
import torch.nn.functional as F

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
        self.data_size = len(data.dataset)
    
    def assign_test(self,data):
        self.test = data

    def assign_model(self, model):
        try:
            self.model.load_state_dict(model.state_dict())
        except:
            self.model = model
        self.model.to(self.device)
        # try:
        #     self.model = torch.compile(self.model)
        # except:
        #     pass

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

    def local_update_epochs(self, local_epochs, train_single_step_func):
        # local_steps may be better!!
        running_loss = 0
        for j in range(local_epochs):
            for k, (inputs, labels) in enumerate(self.train):
                train_single_step_func(inputs, labels)
        # torch.cuda.empty_cache()

    def train_single_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)
        # print(labels)
        # zero the parameter gradients
        # self.model.zero_grad(set_to_none=True)
        self.optim.zero_grad()
        # forward + backward + optimize
        outputs = self.model(inputs)
        # optim
        loss = self.objective(outputs, F.one_hot(labels, outputs.shape[1]).float())
        loss.backward()

        # calculate accumulate gradients
        # grads = torch.tensor([])
        # for index, param in enumerate(self.model.parameters()):
        #     # param.grad = torch.tensor(grads[index])
        #     grads= torch.cat((grads, torch.flatten(param.grad).cpu()),0)
        # self.grads.append(grads)

        self.optim.step()
        # self.train_steps+=1

    # for fedprox and ditto
    def train_single_step_fedprox(self, inputs, labels, reg_model, lam):
        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)
        # zero the parameter gradients
        # self.model.zero_grad(set_to_none=True)
        self.optim.zero_grad()
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
    
    def train_single_step_res(self, inputs, labels, optimizer, model_opt, model_fix):
        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)
        # zero the parameter gradients
        # model_opt.zero_grad(set_to_none=True)
        # model_fix.zero_grad(set_to_none=True)
        optimizer.zero_grad()
        # model_2.zero_grad(set_to_none=True)
        # forward + backward + optimize
        # loss 1
        # m = torch.nn.LogSoftmax(dim=1)
        # ls = torch.nn.NLLLoss()
        # outputs = (m(model_opt(inputs)) + m(model_fix(inputs)))/2
        # loss = ls(outputs, labels)

        # loss 2
        # reg = 0
        # for p,q in zip(model_opt.parameters(), model_fix.parameters()):
        #     reg += torch.norm((p-q),2)

        outputs = model_opt(inputs) + model_fix(inputs) 
        loss = self.objective(outputs, labels)

        # loss 3
        # reg = 0
        # for p,q in zip(model_opt.parameters(), model_fix.parameters()):
        #     reg += torch.square(torch.norm((p-q),2))
        # # print(reg)
        # outputs = model_opt(inputs)
        # loss = self.objective(outputs, labels) + 0.001*reg
        # # optim
        # outputs = torch.norm(model_opt(inputs) - model_fix(inputs), p = 2)
        # self.loss = outputs
        loss.backward()        
        optimizer.step()

    def train_single_step_con(self, inputs, labels, model_sim, model_all, tmp, mu, base = 'representation', reg_lam = None, reg_model = None):
        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)
        # zero the parameter gradients
        # model_opt.zero_grad(set_to_none=True)
        self.optim.zero_grad()
        # forward + backward + optimize

        # contrastive loss
        output_con_dn = 0
        if base == 'representation':
            # for i in model_all:
            #     i.to(self.device)
            #     output_con_dn += torch.exp(F.cosine_similarity(self.intermediate_output(inputs, self.model, self.model.conv2, 'conv2')\
            #          , self.intermediate_output(inputs, i, i.conv2, 'conv2'), dim = -1)/tmp)
            # output_con_n = torch.exp(F.cosine_similarity(self.intermediate_output(inputs, self.model, self.model.conv2, 'conv2')\
            #      , self.intermediate_output(inputs, model_sim, model_sim.conv2, 'conv2'), dim = -1)/tmp)
            for i in model_all:
                output_con_dn += torch.exp(F.cosine_similarity(self.model(inputs), i(inputs))/tmp)
            output_con_n = torch.exp(F.cosine_similarity(self.model(inputs), model_sim(inputs))/tmp)
            con_loss = torch.mean(-torch.log(output_con_n/output_con_dn))

        elif base == 'parameter':
            negative = [torch.cat(tuple([torch.flatten(i.state_dict()[k]) for k in i.state_dict().keys() if 'fc' in k]),0) for i in model_all]
            positive = torch.cat(tuple([torch.flatten(self.model.state_dict()[k]) for k in self.model.state_dict().keys() if 'fc' in k]),0)

            for i in range(len(model_all)):
                tmp = torch.exp(F.cosine_similarity(positive, negative[i], dim=0)/tmp)
                output_con_dn += tmp
                if i == self.label:
                    output_con_n = tmp
            con_loss = -torch.log(output_con_n/output_con_dn) 
        else:
            con_loss = 0
        
        # knowledge sharing
        if reg_lam:
            reg = torch.square(torch.norm(torch.cat(tuple([torch.flatten(self.model.state_dict()[k] - reg_model.state_dict()[k])\
                 for k in self.model.state_dict().keys() if 'fc' not in k]),0),2))
        else:
            reg, reg_lam = 0, 0
                
        loss = self.objective(self.model(inputs), labels) + con_loss * mu + reg_lam * reg
        # if self.id == 0:
        #     # print(self.model.state_dict()['fc1.bias'])
        #     # print(self.label)
        #     # for i in range(len(model_all)):
        #     #     print(i, model_all[i].state_dict()['fc1.bias'])
        #     print(self.intermediate_output(inputs, self.model, self.model.conv2, 'conv2').shape)
        #     # print(self.intermediate_output(inputs, model_sim, model_sim.conv2, 'conv2'))
            # print(output_con_n, output_con_dn)
        
        loss.backward()        
        self.optim.step()

    def intermediate_output(self, inputs, model, model_layer, layer_name):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        model_layer.register_forward_hook(get_activation(layer_name))
        out = model(inputs)
        return torch.flatten(activation[layer_name],1)

    # for IFCA
    def local_train_loss(self, model):
        model.to(self.device)
        train_loss = 0
        i = 0
        with torch.no_grad():
            for data in self.train:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                # forward
                outputs = model(inputs)
                train_loss += self.objective(outputs, labels)
                i+=1
                if i>=10:
                    break
        # return train_loss/len(self.train)
        return train_loss/i
    
    def local_train_acc(self, model):
        model.to(self.device)
        predict_ts = torch.empty(0).to(self.device)
        label_ts = torch.empty(0).to(self.device)
        i = 0
        with torch.no_grad():
            for data in self.train:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predict_ts = torch.cat([predict_ts, predicted], 0)
                label_ts = torch.cat([label_ts, labels], 0)
                i+=1
                if i>=10:
                    break
        acc = accuracy_score(label_ts.cpu(), predict_ts.cpu())
        return acc

    def local_test(self, model_res = None):
        predict_ts = torch.empty(0).to(self.device)
        label_ts = torch.empty(0).to(self.device)
        with torch.no_grad():
            for data in self.test:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                if model_res:
                    outputs = model_res(inputs) + self.model(inputs)
                else:
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
