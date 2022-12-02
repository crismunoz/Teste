from dataset import AdvDebDataset
from trainer import TrainArgs, Trainer
from model import AdversarialDebiasingModel
from torch.utils.data import DataLoader

hidden_size=128
keep_prob=0.1
device='cuda'
batch_size=256
epochs=200
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
    
    