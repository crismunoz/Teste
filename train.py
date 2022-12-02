import matplotlib.pyplot as plt
from data import load_data
import numpy as np

bs = np.linspace(0, 0.25, 10)
gs = np.linspace(9.5, 9.89, 10)
hs = np.linspace(0.5, 0.10, 10)
n = 5

time = []
position = []
g_param = []
h_param = []
b_param = []
for g in gs:
    for h in hs:
        for b in bs:
            time_b, position_b, _ = load_data(g = g, h = h, b = b, n=n)
            time+=list(time_b)
            position+=list(position_b)
            g_param+=[g]*n
            h_param+=[h]*n
            b_param+=[b]*n
            
time = np.array(time)
position = np.array(position)
g_param = np.array(g_param)
h_param = np.array(h_param)
b_param = np.array(b_param)

from dataset import AdvDebDataset
from trainer import TrainArgs, Trainer
from model import AdversarialDebiasingModel
from torch.utils.data import DataLoader

hidden_size=32
keep_prob=0.1
device='cuda'
batch_size=256
epochs=1000
feature_dim =3

import numpy as np
time_np = time/np.max(time)
position_np = position/np.max(position)
h_param_np = h_param/np.max(h_param)
b_param_np = b_param/np.max(b_param)
g_param_np = g_param/np.max(g_param)
params_np = np.stack((h_param_np[:-1], b_param_np[:-1], g_param_np[:-1]), axis=1)
use_debias=False
train_args = TrainArgs(epochs=epochs)
traindataset = AdvDebDataset(time_np[:-1,], position_np[:-1], params_np, position_np[1:], device=device)
dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
adm = AdversarialDebiasingModel(feature_dim, hidden_size, keep_prob).to(device)

trainer = Trainer(adm, dataloader, train_args, use_debias=use_debias)
running_loss = trainer.train()

use_debias=True
running_losses_debias = []
trainers_debias = []
ad_los_ws = [0.001, 0.01, 0.02]
for adversary_loss_weight in ad_los_ws:
    train_args = TrainArgs(epochs=epochs, adversary_loss_weight=adversary_loss_weight)
    traindataset = AdvDebDataset(time_np[:-1,], position_np[:-1], params_np, position_np[1:],  device=device)
    dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    adm = AdversarialDebiasingModel(feature_dim, hidden_size, keep_prob).to(device)
    trainer_debias = Trainer(adm, dataloader, train_args, use_debias=use_debias)
    trainers_debias.append(trainer_debias)
    running_loss_debias = trainer_debias.train()
    running_losses_debias.append(running_loss_debias)
    
import torch
t = time_np[:-1,]
p = position_np[:-1]

data_len = len(t)
t = torch.Tensor(t.reshape([data_len,1])).type(torch.FloatTensor).to(device)
p = torch.Tensor(p.reshape([data_len, -1])).type(torch.FloatTensor).to(device)
params = torch.Tensor(params_np.reshape([data_len, -1])).type(torch.FloatTensor).to(device)

pred_position = trainer.adverarial_debiasing_model.regressor(t, p, params).to('cpu').detach().numpy()

pred_positions_debias = []
for trainer_debias in trainers_debias:
    pred_position_debias = trainer_debias.adverarial_debiasing_model.regressor(t, p, params).to('cpu').detach().numpy()
    pred_positions_debias.append(pred_position_debias)
    

from sklearn.metrics import mean_squared_error
errors = [mean_squared_error(position_np[1:].reshape(-1), pred_position.reshape(-1))]
legends = ['no debias']
for i,pred_position_debias in enumerate(pred_positions_debias):
    errors.append(mean_squared_error(position_np[1:].reshape(-1), pred_position_debias.reshape(-1)))
    legends.append(f'debias-{ad_los_ws[i]}')
import pandas as pd
print(pd.DataFrame({'Strategy':legends, 'RMSE':errors}))