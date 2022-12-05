import numpy as np
from train_utils import load_dataset, dataset_preprocessing
from sklearn.model_selection import train_test_split
from train_utils import evaluate,train_model

dataset = load_dataset()
data = dataset_preprocessing(dataset)
data = train_test_split(*data, test_size=0.2)
train, test = data[::2],data[1::2]

class Config:
    hidden_size=32
    keep_prob=0.0
    device='cuda'
    batch_size=256
    epochs=100
    feature_dim =3

config = Config()
config.use_debias = False
config.adversary_loss_weight = None
configs = {'normal':config}

ad_los_ws = [0.005, 0.01, 0.015]
for adversary_loss_weight in ad_los_ws:
    config = Config()
    config.use_debias = True
    config.adversary_loss_weight = adversary_loss_weight
    configs[f'adv-{adversary_loss_weight}'] = config

models = {}
running_losses = {}
errors_1 = {}
errors_2 = {}
for name,config in configs.items():
    running_loss , model = train_model(train, test, \
        config, name=name)
    
    error_1 = []
    error_2 = []
    for time, params, position in zip(dataset['time'],dataset['params'],dataset['position']):
        error_1.append(evaluate(time, position, params, model, config.device, mode=1))
        error_2.append(evaluate(time, position, params, model, config.device, mode=2))
                    
    running_losses[name] = running_loss
    models[name] = model
    errors_1[name] = np.mean(error_1)
    errors_2[name] = np.mean(error_2)
    
print(errors_1)
print(errors_2)