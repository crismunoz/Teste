from data import load_data
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class Config:
    initial_lr=0.001
    patience=200
    min_delta=1e-5
    
    hidden_size=32
    keep_prob=0.05
    device='cuda'
    batch_size=256
    epochs=400
    feature_dim = 5
    
def load_dataset():
    bs = np.linspace(0, 0.25, 10)
    gs = np.linspace(9.5, 9.89, 10)
    hs = np.linspace(0.5, 0.10, 10)
    n = 5

    from collections import defaultdict
    dataset= defaultdict(list)
    for g in gs:
        for h in hs:
            for b in bs:
                time_b, position_b, _ = load_data(g = g, h = h, b = b, n=n)
                dataset['time'].append(list(time_b))
                dataset['position'].append(list(position_b))
                dataset['params'].append([[g,h,b]]*n)
    return dataset

class DataPreprocessor:
    def __init__(self):
        self.time_scaler = StandardScaler()
        self.position_scaler = StandardScaler()
        self.params_scaler = StandardScaler()
    
    def fit_transform(self, time, position, params):
        self.time_scaler.fit(np.array(time)[:,None])
        self.position_scaler.fit(np.array(position)[:,None])
        self.params_scaler.fit(params)
        return self.transform(time, position, params)
        
    def transform(self, time, position, params):
        time = self.time_scaler.transform(np.array(time)[:,None])
        position = self.position_scaler.transform(np.array(position)[:,None])
        params = self.params_scaler.transform(params)
        
        time_np = time[:-1,]
        x0_np = position[:-1]
        y_np = position[1:]
        next_time_np = time[1:,]
        params_np = params[:-1,]
        return time_np, x0_np, params_np, y_np,next_time_np

class DataPreprocessorMaxMin:
    def __init__(self):
        self.time_scaler = 1
        self.position_scaler = 1
        self.params_scaler = 1
    
    def fit_transform(self, time, position, params):
        self.time_scaler = np.max(np.array(time)[:,None])
        self.position_scaler = np.max(np.array(position)[:,None])
        self.params_scaler = np.max(params, axis=0)
        return self.transform(time, position, params)
        
    def transform(self, time, position, params):
        #time = 2*(np.array(time)[:,None]/self.time_scaler)-1
        #position = 2*(np.array(position)[:,None]/self.position_scaler)-1
        #params = 2*(np.array(params)/self.params_scaler)-1
        time = np.array(time)[:,None]/self.time_scaler
        position = np.array(position)[:,None]/self.position_scaler
        params = np.array(params)/self.params_scaler
        
        time_np = time[:-1,]
        x0_np = position[:-1]
        y_np = position[1:]
        next_time_np = time[1:,]
        params_np = params[:-1,]
        return time_np, x0_np, params_np, y_np, next_time_np
    
def dataset_preprocessing2(dataset):
    time = np.concatenate(dataset['time'])
    position = np.concatenate(dataset['position'])
    params = np.concatenate(dataset['params'])
    
    time_np = (time/np.max(time))[:-1,]
    position_np = position/np.max(position)
    x0_np = position_np[:-1]
    y_np = position_np[1:]
    next_time_np = (time/np.max(time))[1:,]
    params_np = (params/np.max(params, axis=0))[:-1,]
    return time_np, x0_np, params_np, y_np, next_time_np


def train_model(X_train,X_test,y_train,y_test,next_time_train, next_time_test, config, name):
    from dataset import AdvDebDataset
    from trainer import TrainArgs, Trainer
    from model import AdversarialDebiasingModel
    from torch.utils.data import DataLoader
    train_args = TrainArgs(initial_lr = config.initial_lr,
                           patience = config.patience,
                           min_delta = config.min_delta,
                           epochs = config.epochs, 
                           adversary_loss_weight = config.adversary_loss_weight)
    traindataset = AdvDebDataset(X_train,y_train,next_time_train, device=config.device)
    testdataset = AdvDebDataset(X_test,y_test,next_time_test, device=config.device)
    train_dataloader = DataLoader(traindataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(testdataset, batch_size=config.batch_size, shuffle=True)
    adm = AdversarialDebiasingModel(config.feature_dim, config.hidden_size, config.keep_prob).to(config.device)

    trainer = Trainer(adm, train_dataloader, test_dataloader, train_args, 
                      use_debias=config.use_debias, name=name)
    running_loss = trainer.train()
    model= trainer.get_best_model()
    return running_loss , model

def inference(time, x0, params, model, device, mode):
    t = np.array(time)
    x0 = np.array(x0)
    pa = np.array(params)
    
    data_len = len(t)
    t = torch.Tensor(t.reshape([data_len, 1])).type(torch.FloatTensor).to(device)
    x0 = torch.Tensor(x0.reshape([data_len, -1])).type(torch.FloatTensor).to(device)
    pa = torch.Tensor(pa.reshape([data_len, -1])).type(torch.FloatTensor).to(device)

    if mode == 1:
        x00 = x0[0]
        pred_position = []
        for tt,paa in zip(t,pa):
            x_inp = torch.concat([tt[None,...], x00[None,...], paa[None,...]], axis=1)
            x00 = model.regressor(x_inp)[0]
            pred_position.append(x00.to('cpu').detach().numpy())
    
    elif mode == 2: 
        pred_position = []
        for tt,x00,paa in zip(t,x0,pa):
            x_inp = torch.concat([tt[None,...], x00[None,...], paa[None,...]], axis=1)
            x00_ = model.regressor(x_inp)[0]
            pred_position.append(x00_.to('cpu').detach().numpy())
        
    return np.array(pred_position).reshape(-1)