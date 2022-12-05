import matplotlib.pyplot as plt
from data import load_data
import numpy as np

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
            
time = np.concatenate(dataset['time'])
position = np.concatenate(dataset['position'])
params = np.concatenate(dataset['params'])

def data_preprocessing():
  import numpy as np
  time_np = (time/np.max(time))[:-1,]
  position_np = position/np.max(position)
  x0_np = position_np[:-1]
  y_np = position_np[1:]
  params_np = (params/np.max(params, axis=0))[:-1,]
  return time_np, x0_np, params_np, y_np

from sklearn.model_selection import train_test_split
data = data_preprocessing()
data = train_test_split(*data, test_size=0.2)
train, test = data[::2],data[1::2]

from dataset import AdvDebDataset
from trainer import TrainArgs, Trainer
from model import AdversarialDebiasingModel
from torch.utils.data import DataLoader

hidden_size=32
keep_prob=0.0
device='cuda'
batch_size=256
epochs=100
feature_dim =3

import numpy as np
use_debias=False
train_args = TrainArgs(epochs=epochs)
traindataset = AdvDebDataset(*train, device=device)
testdataset = AdvDebDataset(*test, device=device)
train_dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)
adm = AdversarialDebiasingModel(feature_dim, hidden_size, keep_prob).to(device)

trainer = Trainer(adm, train_dataloader, test_dataloader, train_args, use_debias=use_debias, name='normal')
running_loss = trainer.train()
model= trainer.get_best_model()

use_debias=True
running_losses_debias = []
adv_models = []
ad_los_ws = [0.001, 0.01, 0.02]
for i,adversary_loss_weight in enumerate(ad_los_ws):
    train_args = TrainArgs(epochs=epochs, adversary_loss_weight=adversary_loss_weight)
    adm = AdversarialDebiasingModel(feature_dim, hidden_size, keep_prob).to(device)

    traindataset = AdvDebDataset(*train, device=device)
    testdataset = AdvDebDataset(*test, device=device)
    train_dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)
    trainer_adv = Trainer(adm, train_dataloader, test_dataloader, train_args, use_debias=use_debias, name=f'adv-{ad_los_ws[i]}')
    running_loss_debias = trainer_adv.train()
    running_losses_debias.append(running_loss_debias)
    adv_models.append(trainer_adv.get_best_model())
    
    
    
import torch
from sklearn.metrics import mean_squared_error
errors = defaultdict(list)
for time, params, position in zip(dataset['time'],dataset['params'],dataset['position']):
    t = np.array(time)[:-1]
    x0 = np.array(position)[:-1]
    pa = np.array(params)[:-1,:]
    y = np.array(position)[1:]
    
    data_len = len(t)
    t = torch.Tensor(t.reshape([data_len,1])).type(torch.FloatTensor).to(device)
    x0 = torch.Tensor(x0.reshape([data_len, -1])).type(torch.FloatTensor).to(device)
    pa = torch.Tensor(pa.reshape([data_len, -1])).type(torch.FloatTensor).to(device)

    x00 = x0[0]
    pred_position = []
    for tt,paa in zip(t,pa):
        x00 = model.regressor(tt[None,...], x00[None,...], paa[None,...])[0]
        pred_position.append(x00.to('cpu').detach().numpy())
    errors['normal'].append(mean_squared_error(y.reshape(-1), np.array(pred_position).reshape(-1)))
    
    pred_positions_adv = []
    for i,adv_model in enumerate(adv_models):
        x00 = x0[0]
        pred_position_adv = []
        for tt,paa in zip(t,pa):
            x00 = adv_model.regressor(tt[None,...], x00[None,...], paa[None,...])[0]
            pred_position_adv.append(x00.to('cpu').detach().numpy())
        errors[f'adv-{ad_los_ws[i]}'].append(mean_squared_error(y.reshape(-1), np.array(pred_position_adv).reshape(-1)))
        

erros_case1 = {name:np.mean(errors[name]) for name in errors.keys()}
print(erros_case1)

import torch
from sklearn.metrics import mean_squared_error
errors = defaultdict(list)
for time, params, position in zip(dataset['time'],dataset['params'],dataset['position']):
    t = np.array(time)[:-1]
    x0 = np.array(position)[:-1]
    pa = np.array(params)[:-1,:]
    y = np.array(position)[1:]
    
    data_len = len(t)
    t = torch.Tensor(t.reshape([data_len,1])).type(torch.FloatTensor).to(device)
    x0 = torch.Tensor(x0.reshape([data_len, -1])).type(torch.FloatTensor).to(device)
    pa = torch.Tensor(pa.reshape([data_len, -1])).type(torch.FloatTensor).to(device)

    pred_position = []
    for tt,x00,paa in zip(t,x0,pa):
        x00_ = model.regressor(tt[None,...], x00[None,...], paa[None,...])[0].to('cpu').detach().numpy()
        pred_position.append(x00_)
    errors['normal'].append(mean_squared_error(y.reshape(-1), np.array(pred_position).reshape(-1)))
    
    pred_positions_adv = []
    for i,adv_model in enumerate(adv_models):
        pred_position_adv = []
        for tt,x00,paa in zip(t,x0,pa):
            x00_ = adv_model.regressor(tt[None,...], x00[None,...], paa[None,...])[0].to('cpu').detach().numpy()
            pred_position_adv.append(x00_)
        errors[f'adv-{ad_los_ws[i]}'].append(mean_squared_error(y.reshape(-1), np.array(pred_position_adv).reshape(-1)))
        
        
erros_case2 = {name:np.mean(errors[name]) for name in errors.keys()}
print(erros_case2)