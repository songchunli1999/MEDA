import time
import torch
import os
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import torchvision
from utils import get_optimizer, get_model
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
from optimization import Optimization
class Client():
    def __init__(self, cid, data, device, project_dir, model_name, local_epoch, lr, batch_size, drop_rate, stride):
        self.cid = cid
        self.project_dir = project_dir
        self.model_name = model_name
        self.data = data
        self.device = device
        self.local_epoch = local_epoch
        self.lr = lr
        self.batch_size = batch_size
        
        self.dataset_sizes = self.data.train_dataset_sizes[cid]
        self.train_loader = self.data.train_loaders[cid]

        self.full_model = get_model(self.data.train_class_sizes[cid], drop_rate, stride)
        self.classifier = self.full_model.classifier.classifier
        self.full_model.classifier.classifier = nn.Sequential()
        self.model = self.full_model
        self.distance=0
        self.optimization = Optimization(self.train_loader, self.device)
        # print("class name size",class_names_size[cid])

    def train(self, federated_model, use_cuda):
        self.y_err = []
        self.y_loss = []
        import pickle
        
        device = torch.device("cuda")
        self.model.load_state_dict(federated_model.state_dict())
        self.model.classifier.classifier = self.classifier
        self.old_classifier = copy.deepcopy(self.classifier)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        
        criterion = nn.CrossEntropyLoss()


        since = time.time()
        
        print('Client', self.cid, 'start training')
        print('CE loss fedpav training')
        for epoch in range(self.local_epoch):
            print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
            print('-' * 10)

            scheduler.step()
            self.model.train(True)
            running_loss = 0.0
            running_corrects = 0.0
            
            for data in self.train_loader:
                inputs, labels = data
                b, c, h, w = inputs.shape
                if b < self.batch_size:
                    continue
                if use_cuda:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                optimizer.zero_grad()

                #start get feature_cov
                self.classifiercopy=copy.deepcopy(self.model.classifier.classifier).cuda()
                self.model.classifier.classifier=nn.Sequential()
                out_feature1 = self.model(inputs)
                #.state_dict()
                tensor4 = out_feature1.cpu().det  ach().numpy().transpose()
                # compute  covariance matrix
                covariance_matrix = np.cov(tensor4)
                
                #save covariance matrix
                import pickle
                a_dict1 = {'covariance_matrix':covariance_matrix}
                file = open('./server/covariance_matrix.pickle'+self.cid,'wb')
                pickle.dump(a_dict1,file)
                file.close
              
                optimizer.zero_grad()
                
                self.model.classifier.classifier=self.classifiercopy
                outputs = self.classifier(out_feature1)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item() * b
                running_corrects += float(torch.sum(preds == labels.data))

            used_data_sizes = (self.dataset_sizes - self.dataset_sizes % self.batch_size)
            epoch_loss = running_loss / used_data_sizes
            epoch_acc = running_corrects / used_data_sizes

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))

            self.y_loss.append(epoch_loss)
            self.y_err.append(1.0-epoch_acc)

            time_elapsed = time.time() - since
            print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - since
        print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)
        self.classifier = self.model.classifier.classifier
        self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
        self.model.classifier.classifier = nn.Sequential()

    def train_ISDAloss(self, federated_model, use_cuda):
        self.y_err = []
        self.y_loss = []
        import pickle
        
        device = torch.device("cuda")
        self.model.load_state_dict(federated_model.state_dict())
        self.model.classifier.classifier = self.classifier
        self.old_classifier = copy.deepcopy(self.classifier)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        
        criterion = nn.CrossEntropyLoss()


        since = time.time()
        
        print('Client', self.cid, 'start training')
        print('ISDA loss fedpav training')
        for epoch in range(self.local_epoch):
            print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
            print('-' * 10)

            scheduler.step()
            self.model.train(True)
            running_loss = 0.0
            running_corrects = 0.0
            
            with torch.no_grad():
             import pickle
             try:
               with open('./server/covariance_matrix_avg.pickle','rb') as f:
                 a_dict2 = pickle.load(f)
                 f.close
             except EOFError: 
                print ('covariance matrix skip over')
                pass
            
            for data in self.train_loader:
                inputs, labels = data
                b, c, h, w = inputs.shape
                if b < self.batch_size:
                    continue
                if use_cuda:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                fc_kg=list(self.model.classifier.classifier.state_dict().values())
                self.classifiercopy=copy.deepcopy(self.model.classifier.classifier).cuda()
                self.model.classifier.classifier=nn.Sequential()
                out_feature1 = self.model(inputs)
                #.state_dict()
                tensor4 = out_feature1.cpu().detach().numpy().transpose()
                # compute  covariance matrix
                covariance_matrix = np.cov(tensor4)
                
                #save covariance matrix
                import pickle
                a_dict1 = {'covariance_matrix':covariance_matrix}
                file = open('./server/covariance_matrix.pickle'+self.cid,'wb')
                pickle.dump(a_dict1,file)
                file.close
              
                optimizer.zero_grad()
                
                self.model.classifier.classifier=self.classifiercopy
                outputs = self.classifier(out_feature1)
                
                
                N = out_feature1.size(0)
                C = self.data.train_class_sizes[self.cid]
                A = out_feature1.size(1)
                
        
                weight_m=fc_kg[0]
                NxW_ij = weight_m.expand(N, C, A)
                NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
        
                s_CV_temp = a_dict2['covariance_matrix_avg']
                s_CV_temp = torch.tensor(s_CV_temp)
                s_CV_temp = s_CV_temp.expand(N,A ,A)
                #use beta calculate sigma_ij
                NxW_ij = NxW_ij.to(device)
                NxW_kj = NxW_kj.to(device)
                s_CV_temp = s_CV_temp.to(device)
                s_CV_temp = torch.tensor(s_CV_temp,dtype=torch.float32)
                
                sigma2 = torch.bmm(torch.bmm(NxW_ij - NxW_kj, s_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
                sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)
                aug_result = outputs +0.5*sigma2
                
                _, preds = torch.max(aug_result.data, 1)
                

                loss = criterion(aug_result, labels)
                #all_new_loss.append(new_loss)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * b
                running_corrects += float(torch.sum(preds == labels.data))
                
                
            used_data_sizes = (self.dataset_sizes - self.dataset_sizes % self.batch_size)
            epoch_loss = running_loss / used_data_sizes
            epoch_acc = running_corrects / used_data_sizes

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))

            self.y_loss.append(epoch_loss)
            self.y_err.append(1.0-epoch_acc)

            time_elapsed = time.time() - since
            print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - since
        print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)
        self.classifier = self.model.classifier.classifier
        self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
        self.model.classifier.classifier = nn.Sequential()
        
        self.model.classifier.classifier = nn.Sequential()
    
    def generate_soft_label(self, x, regularization):
        return self.optimization.kd_generate_soft_label(self.model, x, regularization)

    def get_model(self):
        return self.model

    def get_data_sizes(self):
        return self.dataset_sizes

    def get_train_loss(self):
        return self.y_loss[-1]

    def get_cos_distance_weight(self):
        return self.distance