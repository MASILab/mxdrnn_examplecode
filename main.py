

#from func.data.data_generator import Dataset
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets, transforms
import pandas as pd
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence
import crnn
from rnn_dataset import RMNIST, MNIST, MMNIST
from mnist_loader import get_train_valid_loader, get_test_loader
import numpy as np

class Net(nn.Module):              # This is our ToyNet
    def __init__(self, in_channel):           #Initialize the value of the instance. These values are generally used by other methods.
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 10, kernel_size=5)   # 2D convolutional layer
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
        self.conv2_drop = nn.Dropout2d()           # Dropout layer
        self.fc1 = nn.Linear(320, 50)             # fully connected layer
        self.fc2 = nn.Linear(50, 10)              # fully connected layer

    def forward(self, x):                    # define the network using the module in __init__
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ConvRNN(nn.Module):  #  This is xDRNN + ToyNet
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, time):
        super(ConvRNN, self).__init__()
        self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        self.time = time
        self.conv1 = nn.Conv2d(out_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
        self.conv2_drop = nn.Dropout2d()           # Dropout layer
        self.fc1 = nn.Linear(320, 50)             # fully connected layer
        self.fc2 = nn.Linear(50, num_classes)  
        
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(self.time):
            if i == 0:
                hx, cx = self.lstmcell(x[i])
            else:
                hx, cx = self.lstmcell(x[i], (hx, cx))
        x = F.relu(F.max_pool2d(self.conv1(hx), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MPConvRNN(nn.Module):   #  This is MxDRNN + ToyNet
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, time):
        super(MPConvRNN, self).__init__()

        self.time = time
        self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(time * time, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
        self.conv2_drop = nn.Dropout2d()           # Dropout layer
        self.fc1 = nn.Linear(320, 50)             # fully connected layer
        self.fc2 = nn.Linear(50, num_classes)  
    def forward(self, x0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # x.shape == (time, batch, :,:,:)
        x1 = x0[[2,1,0], :, :, :, :]   # for 3 steps
        x2 = x0[[0,2,1], :, :, :, :]   # for 3 steps
        #x1 = x0[[1,0], :, :, :, :]    # for 2 steps
        inp = []
        for x in [x0, x1, x2]:
            for i in range(self.time):
                if i == 0:
                    hx, cx = self.lstmcell(x[i])
                else:
                    hx, cx = self.lstmcell(x[i], (hx, cx))
            inp.append(hx)
        x = torch.cat(inp, 1)  # dim0: batch, dim1: channel
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
        if self.cfig['model_name'] == 'mxdrnn':
            print ('====== use  MxDRNN model ======')
            self.model = MPConvRNN(in_channels = 1, out_channels = self.cfig['time'], kernel_size = 3, num_classes = 10, time = self.cfig['time']).to(self.device)
            
        if self.cfig['model_name'] == 'xdrnn':
            print ('====== use xDRNN model ======')
            self.model = ConvRNN(in_channels = 1, out_channels = self.cfig['time'], kernel_size = 3, num_classes = 10, time = self.cfig['time']).to(self.device)
            
    
        if self.cfig['model_name'] == 'multi':
            self.model = Net(in_channel = self.cfig['time']).to(self.device)
            print ('======== use multi channel model =======')

        if self.cfig['model_name'] == 'bl':
            self.model = Net(in_channel = 1).to(self.device)
            print ('the baseline model')
            
        self.train_loader, self.val_loader = get_train_valid_loader(self.cfig['data_root'],
                       self.cfig['batch_size'],
                       model_name = self.cfig['model_name'],   time = self.cfig['time'],
                       random_seed = 1234,
                       augment=False,
                       valid_size=0.1,
                       shuffle=True,
                       show_sample=False,
                       num_workers=4,
                       pin_memory=True)
        self.test_loader = get_test_loader(self.cfig['data_root'], 
                self.cfig['batch_size'],
                model_name = self.cfig['model_name'], time = self.cfig['time'],
                shuffle=False,
                num_workers=4,
                pin_memory=True)
        print ('len test_loader: ', len(self.test_loader) )
        print ('len train_loader: ', len(self.train_loader) )
        print ('len val_loader: ', len(self.val_loader) )
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.cfig['learning_rate'], betas=(0.9, 0.999))

        
    def train(self):
        for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
            model_root = osp.join(self.cfig['save_path'], 'models')
            if not os.path.exists(model_root):
                os.mkdir(model_root)
            model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
            if os.path.exists(model_pth) and self.cfig['use_exist_model']:
                if self.device == 'cuda': #there is a GPU device
                    self.model.load_state_dict(torch.load(model_pth))
                else:
                    self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
            else:
                self.train_epoch(epoch)
                if self.cfig['savemodel']:
                    torch.save(self.model.state_dict(), model_pth)
            if self.cfig['iseval']:
                self.eval_epoch(epoch)
                self.test_epoch(epoch)
    
    def train_epoch(self, epoch):
        self.model.train()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        train_csv = os.path.join(self.csv_path, 'train.csv')
        pred_list, target_list, loss_list = [],[],[]
        print ('Epoch: ', epoch)
        for batch_idx, (data, target) in enumerate(self.train_loader):
#             if self.cfig['use_rnn']:
#                 sequence_length, input_size = 28, 28
#                 data = data.reshape(-1, sequence_length, input_size)
            data, target = data.to(self.device), target.to(self.device)
            self.optim.zero_grad()
            if batch_idx == 0: print (data.shape)
            data = data.type(torch.cuda.FloatTensor)
            if self.cfig['model_name'][-3:] == 'rnn':
                data = data.permute([1,0,2,3,4])
            pred = self.model(data)             # here should be careful
            if batch_idx == 0:
                print ('data.shape',data.shape)
                print ('pred.shape', pred.shape)
                print('Epoch: ', epoch)

            #loss = LossPool(pred, target, self.cfig, loss_name=self.cfig['loss_name']).get_loss()
            loss = nn.CrossEntropyLoss()(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optim.step()
            print_str = 'train epoch=%d, batch_idx=%d/%d, loss=%.4f\n' % (
            epoch, batch_idx, len(self.train_loader), loss.data[0])
            
            pred_cls = pred.data.max(1)[1]
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
            

        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(train_csv):
            csv_info = ['epoch', 'loss', 'accuracy']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(train_csv)
        df = pd.read_csv(train_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        #print('------------------', tmp_epoch)
        print ('train accuracy: ', accuracy)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)

        data['epoch'], data['loss'], data['accuracy'] = tmp_epoch, tmp_loss, tmp_acc
        data.to_csv(train_csv)
        print ('train acc: ', accuracy)


            
        
    def eval_epoch(self, epoch):  
       
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'eval.csv')
        pred_list, target_list, loss_list = [],[],[]
        print ()
        for batch_idx, (data, target) in enumerate(self.val_loader):
            
            data, target = data.to(self.device), target.to(self.device)
            self.optim.zero_grad()
            #print ('=================',data.shape)
            data = data.type(torch.cuda.FloatTensor)
            if self.cfig['model_name'][-3:] == 'rnn':
                data = data.permute([1,0,2,3,4])
            pred = self.model(data)             # here should be careful
            #loss = self.criterion(pred, target)
            loss = nn.CrossEntropyLoss()(pred, target)
            pred_cls = pred.data.max(1)[1]  # not test yet
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
            
        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss', 'accuracy']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        #print ('------------------', tmp_epoch)
        print ('val accuracy: ', accuracy)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        
        
        data['epoch'], data['loss'], data['accuracy'] =tmp_epoch, tmp_loss, tmp_acc
        print ('max val accuracy at: ', max(tmp_acc), tmp_acc.index(max(tmp_acc)))
        data.to_csv(eval_csv)
        

            
    def test_epoch(self, epoch):
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'test.csv')
        pred_list, target_list, loss_list = [],[],[]
        print ()
        for batch_idx, (data, target) in enumerate(self.test_loader):    # test_loader
            
            data, target = data.to(self.device), target.to(self.device)
            self.optim.zero_grad()
            #print ('=================',data.shape)
            data = data.type(torch.cuda.FloatTensor)
            if self.cfig['model_name'][-3:] == 'rnn':
                data = data.permute([1,0,2,3,4])
            pred = self.model(data)             # here should be careful
            #loss = self.criterion(pred, target)
            loss = nn.CrossEntropyLoss()(pred, target)
            pred_cls = pred.data.max(1)[1]  # not test yet
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())

        accuracy=accuracy_score(target_list,pred_list)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss', 'accuracy']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        
        print ('test accuracy: ', accuracy)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        
        
        data['epoch'], data['loss'], data['accuracy'] =tmp_epoch, tmp_loss, tmp_acc
        
        data.to_csv(eval_csv)
                     
        
import yaml
import shutil        
if __name__ == '__main__':
    f = open('mnist.yaml', 'r').read()
    cfig = yaml.load(f)
    shutil.copyfile('./mnist.yaml', cfig['save_path'] + '/tmp.yaml')
    trainer = Trainer(cfig)
    trainer.train()
