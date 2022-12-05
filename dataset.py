import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
EPS = torch.finfo(torch.float32).eps

class AdvDebDataset(Dataset):
    """
    Dataset class to train adversarial debiasing pytorch model.
    """

    def __init__(self, X, y, next_time, device):
        self.X = X
        self.y = y
        self.next_time = next_time
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx]).type(torch.FloatTensor).to(self.device)
        y = torch.tensor(self.y[idx]).type(torch.FloatTensor).to(self.device)
        next_time = torch.tensor(self.next_time[idx]).type(torch.FloatTensor).to(self.device)
        neg_exp = torch.tensor([0]).type(torch.FloatTensor).to(self.device)+EPS
        pos_exp = torch.tensor([1]).type(torch.FloatTensor).to(self.device)-EPS
        return X, (y, next_time, neg_exp, pos_exp)
