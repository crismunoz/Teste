import torch
from torch import nn

class TruncateModel(nn.Module):
    "Regressor Model, You can change the layer configuration as you wish!"
    def __init__(self, feature_dim, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(feature_dim, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.hidden(inputs)
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
        self.act = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.core_model(inputs)
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
        self.act = nn.ReLU()

    def forward(self, inputs, trainable=True):
        x = self.core_model(inputs)
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

    def forward(self, inputs):
        y_logits = self.regressor(inputs)
        z_prob, _ = self.adversarial(inputs)
        return y_logits, z_prob