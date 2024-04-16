import os
# -*- coding:utf-8 -*-
import math
import json
import matplotlib.pyplot as plt
from utils import get_model, extract_feature
import torch.nn as nn
import torch
import scipy.io
import copy
import numpy as np
from numpy.linalg import inv,eig, det
from data_utils import ImageDataset
import random
import torch.optim as optim
from torchvision import datasets

def add_model(dst_model, src_model, dst_no_data, src_no_data):
    if dst_model is None:
        result = copy.deepcopy(src_model)
        return result
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data*src_no_data + dict_params2[name1].data*dst_no_data)
    return dst_model

def scale_model(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model

def aggregate_models(models, weights):
    """aggregate models based on weights
    params:
        models: model updates from clients
        weights: weights for each model, e.g. by data sizes or cosine distance of features
    """
    
    if models == []:
        return None
    model = add_model(None, models[0], 0, weights[0])
    total_no_data = weights[0]
    for i in range(1, len(models)):
        model = add_model(model, models[i], total_no_data, weights[i])
        model = scale_model(model, 1.0 / (total_no_data+weights[i]))
        total_no_data = total_no_data + weights[i]
    return model


class Server():
    def __init__(self, clients, data, device, project_dir, model_name, num_of_clients, lr, drop_rate, stride, multiple_scale):
        self.project_dir = project_dir
        self.data = data
        self.device = device
        self.model_name = model_name
        self.clients = clients
        
        self.client_list = self.data.client_list
        self.num_of_clients = num_of_clients
        self.lr = lr
        self.multiple_scale = multiple_scale
        self.drop_rate = drop_rate
        self.stride = stride

        self.multiple_scale = []
        for s in multiple_scale.split(','):
            self.multiple_scale.append(math.sqrt(float(s)))

        self.full_model = get_model(750, drop_rate, stride).to(device)
        
        self.full_model.classifier.classifier = nn.Sequential()
        self.federated_model=self.full_model
        self.federated_model.eval()
        self.train_loss = []


    def train(self, epoch, cdw, use_cuda):
        models = []
        loss = []
        cos_distance_weights = []
        data_sizes = []
        class_sizes = [652,541,694,241,576,558]
        current_client_list = self.client_list
        t = 150
        lambda_number = 0.25
        for i in current_client_list:
            #here spoch is round
            if epoch < t:
              self.clients[i].train(self.federated_model, use_cuda)
            
            else:
              beta = round(lambda_number * (epoch/500), 2)
              print("beta:", beta)
              self.clients[i].train_ISDAloss(self.federated_model, use_cuda,beta)
            cos_distance_weights.append(self.clients[i].get_cos_distance_weight())
            loss.append(self.clients[i].get_train_loss())
            models.append(self.clients[i].get_model())
            data_sizes.append(self.clients[i].get_data_sizes())

        if epoch==0:
            self.L0 = torch.Tensor(loss) 

        avg_loss = sum(loss) / self.num_of_clients

        print("==============================")
        print("number of clients used:", len(models))
        print('Train Epoch: {}, AVG Train Loss among clients of lost epoch: {:.6f}'.format(epoch, avg_loss))
        print()
        
        self.train_loss.append(avg_loss)
        
        
        #get covsum
        E = np.eye(512)
        cov_norm = []
        covs = []
        
        for dataset in current_client_list:
          import pickle
          try:
            with open('./server/covariance_matrix.pickle'+dataset,'rb') as f:
              a_dict4 = pickle.load(f)
              f.close
              
              a = a_dict4['covariance_matrix']
              covs.append(a)
              print(dataset)
              print("take the eig max of the matrix")
              a1,a2 =np.linalg.eig(a)
              a1 = a1.real[0]
              print("1 eig max:",a1)
              cov_norm.append(a1)
              
              
          except EOFError: 
            print ('covariance matrix'+dataset+'in aggreate skip over')
            pass
            
          
        if epoch < t:
          agg_weights = data_sizes
        else:
          agg_weights = cov_norm
        #get cov_avg
        cov_avg = np.zeros((512,512))
        for i in range(len(covs)):
           cov_avg = cov_avg + (covs[i]*(agg_weights[i]/sum(agg_weights)))
        
        
        #save avg cov
        import pickle
        a_dict2 = {'covariance_matrix_avg':cov_avg}
        file = open('./server/covariance_matrix_avg.pickle','wb')
        pickle.dump(a_dict2,file)
        file.close
        
        weights = agg_weights
        print("weights of clients", current_client_list,weights)
        if cdw:
            print("cos distance weights:", cos_distance_weights)
            weights = cos_distance_weights

        self.federated_model = aggregate_models(models, weights)
        

    def draw_curve(self):
        plt.figure()
        x_epoch = list(range(len(self.train_loss)))
        plt.plot(x_epoch, self.train_loss, 'bo-', label='train')
        plt.legend()
        dir_name = os.path.join(self.project_dir, 'model', self.model_name)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        plt.savefig(os.path.join(dir_name, 'train.png'))
        plt.close('all')
        
    def test(self, use_cuda):
        print("="*10)
        print("Start Tesing!")
        print("="*10)
        print('We use the scale: %s'%self.multiple_scale)
        dataset = self.data.datasets[0]
        self.federated_model = self.federated_model.eval()
        if use_cuda:
            self.federated_model = self.federated_model.cuda()
        
        with torch.no_grad():
            gallery_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['gallery'], self.multiple_scale)
            query_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['query'], self.multiple_scale)

        result = {
            'gallery_f': gallery_feature.numpy(),
            'gallery_label': self.data.gallery_meta[dataset]['labels'],
            'gallery_cam': self.data.gallery_meta[dataset]['cameras'],
            'query_f': query_feature.numpy(),
            'query_label': self.data.query_meta[dataset]['labels'],
            'query_cam': self.data.query_meta[dataset]['cameras']}

        scipy.io.savemat(os.path.join(self.project_dir,
                    'model',
                    self.model_name,
                    'pytorch_result.mat'),
                    result)
                    
        print(self.model_name)
        print(dataset)

        os.system('python evaluate.py --result_dir {} --dataset {}'.format(os.path.join(self.project_dir, 'model', self.model_name), dataset))
              
              
    def knowledge_distillation(self, regularization):
        MSEloss = nn.MSELoss().to(self.device)
        optimizer = optim.SGD(self.federated_model.parameters(), lr=self.lr*0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
        self.federated_model.train()

        for _, (x, target) in enumerate(self.data.kd_loader): 
            x, target = x.to(self.device), target.to(self.device)
            # target=target.long()
            optimizer.zero_grad()
            soft_target = torch.Tensor([[0]*512]*len(x)).to(self.device)
        
            for i in self.client_list:
                i_label = (self.clients[i].generate_soft_label(x, regularization))
                soft_target += i_label
            soft_target /= len(self.client_list)
        
            output = self.federated_model(x)
            
            loss = MSEloss(output, soft_target)
            loss.backward()
            optimizer.step()
            print("train_loss_fine_tuning", loss.data)