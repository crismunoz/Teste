from sklearn.metrics import mean_squared_error
from data import load_data
import numpy as np
import torch
    
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

def dataset_preprocessing(dataset):
    time = np.concatenate(dataset['time'])
    position = np.concatenate(dataset['position'])
    params = np.concatenate(dataset['params'])
    time_np = (time/np.max(time))[:-1,]
    position_np = position/np.max(position)
    x0_np = position_np[:-1]
    y_np = position_np[1:]
    params_np = (params/np.max(params, axis=0))[:-1,]
    return time_np, x0_np, params_np, y_np


def train_model(train, test, config, name):
    from dataset import AdvDebDataset
    from trainer import TrainArgs, Trainer
    from model import AdversarialDebiasingModel
    from torch.utils.data import DataLoader
    train_args = TrainArgs(initial_lr = config.initial_lr,
                           patience = config.patience,
                           min_delta = config.min_delta,
                           epochs = config.epochs, 
                           adversary_loss_weight = config.adversary_loss_weight)
    traindataset = AdvDebDataset(*train, device=config.device)
    testdataset = AdvDebDataset(*test, device=config.device)
    train_dataloader = DataLoader(traindataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(testdataset, batch_size=config.batch_size, shuffle=True)
    adm = AdversarialDebiasingModel(config.feature_dim, config.hidden_size, config.keep_prob).to(config.device)

    trainer = Trainer(adm, train_dataloader, test_dataloader, train_args, 
                      use_debias=config.use_debias, name=name)
    running_loss = trainer.train()
    model= trainer.get_best_model()
    return running_loss , model

def evaluate(time, position, params, model, device, mode):
    t = np.array(time)[:-1]
    x0 = np.array(position)[:-1]
    pa = np.array(params)[:-1,:]
    y = np.array(position)[1:]
    
    data_len = len(t)
    t = torch.Tensor(t.reshape([data_len,1])).type(torch.FloatTensor).to(device)
    x0 = torch.Tensor(x0.reshape([data_len, -1])).type(torch.FloatTensor).to(device)
    pa = torch.Tensor(pa.reshape([data_len, -1])).type(torch.FloatTensor).to(device)

    if mode == 1:
        x00 = x0[0]
        pred_position = []
        for tt,paa in zip(t,pa):
            x00 = model.regressor(tt[None,...], x00[None,...], paa[None,...])[0]
            pred_position.append(x00.to('cpu').detach().numpy())
    
    elif mode == 2: 
        pred_position = []
        for tt,x00,paa in zip(t,x0,pa):
            x00_ = model.regressor(tt[None,...], x00[None,...], paa[None,...])[0]
            pred_position.append(x00_.to('cpu').detach().numpy())
        
    return mean_squared_error(y.reshape(-1), np.array(pred_position).reshape(-1))