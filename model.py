import torch
from torch import nn

class TruncateModel(nn.Module):
    "Regressor Model, You can change the layer configuration as you wish!"
    def __init__(self, feature_dim, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(feature_dim+2, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()

    def forward(self, time, x0, param):
        x = torch.cat((time, x0, param), 1)
        x = self.hidden(x)
        x = self.act(x)
        x = self.hidden2(x)
        x = self.act(x)
        return x

class RegressorModel(nn.Module):
    "Regressor Model, You can change the layer configuration as you wish!"
    def __init__(self, hidden_size, core_model):
        super().__init__()
        self.core_model = core_model
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()
        self.hidden2 = nn.Linear(hidden_size, 1)

    def forward(self, time, x0, param):
        x = self.core_model(time, x0, param)
        x = self.hidden1(x)
        x = self.act(x)
        x = self.hidden2(x)
        return x

class AdversarialModel(nn.Module):
    "Regressor Model, You can change the layer configuration as you wish!"
    def __init__(self, hidden_size, keep_prob, core_model):
        super().__init__()
        self.core_model = core_model
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(keep_prob)
        self.out_act = nn.Sigmoid()
        self.act = nn.Tanh()

    def forward(self, time, x1, param, trainable=True):
        x = self.core_model(time, x1, param)
        x = self.drop(x)
        x = self.hidden1(x)
        x = self.act(x)
        x = self.hidden2(x)
        p = self.out_act(x)
        if trainable:
            return p, x
        else:
            return p
        
class AdversarialDebiasingModel(nn.Module):
    """
    Complete system integrate classifier and adversarial submodels.
    """
    def __init__(self, feature_dim, hidden_size, keep_prob):
        super().__init__()
        core_model = TruncateModel(feature_dim, hidden_size)
        self.regressor = RegressorModel(hidden_size, core_model)
        self.adversarial = AdversarialModel(hidden_size, keep_prob, core_model)

    def forward(self, time, x0, param):
        y_logits = self.regressor(time, x0, param)
        z_prob, _ = self.adversarial(time, y_logits, param)
        return y_logits, z_prob