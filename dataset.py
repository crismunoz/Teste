import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
EPS = torch.finfo(torch.float32).eps

class AdvDebDataset(Dataset):
    """
    Dataset class to train adversarial debiasing pytorch model.
    """

    def __init__(self, time, x0, params, y, device):
        data_len = len(time)
        self.time = time.reshape([data_len,1])
        self.x0 = x0.reshape([data_len, -1])
        self.params = params.reshape([data_len, -1])
        self.y = y
        self.device = device

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        time = torch.tensor(self.time[idx]).type(torch.FloatTensor).to(self.device)
        x0 = torch.tensor(self.x0[idx]).type(torch.FloatTensor).to(self.device)
        y = torch.tensor(self.y[idx, None]).type(torch.FloatTensor).to(self.device)
        params = torch.tensor(self.params[idx]).type(torch.FloatTensor).to(self.device)
        neg_exp = torch.tensor([0]).type(torch.FloatTensor).to(self.device)+EPS
        pos_exp = torch.tensor([1]).type(torch.FloatTensor).to(self.device)-EPS
        inp_params = (time, x0, params)
        return inp_params, (y, neg_exp, pos_exp)
