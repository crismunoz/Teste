from train_utils import load_dataset
from train_utils import inference, train_model, Config, DataPreprocessorMaxMin, DataPreprocessor
import pandas as pd
import numpy as np
import argparse
import os
import pickle
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversary_loss_weight', type=float, default=-0.01)
    args = parser.parse_args()
    
    data_prep = DataPreprocessorMaxMin()
    dataset = load_dataset()
    data_size = len(dataset['time'])
    indexes = list(range(data_size))
    random.shuffle(indexes)
    train_size = int(0.8*data_size)
    test_size = data_size - train_size
    print(f"train_size: {train_size}\ntest_size: {test_size}")
    train_index = indexes[:train_size]
    test_index = indexes[:train_size]
    select = lambda x,idx: np.concatenate([x[i] for i in idx])
    time = select(dataset['time'], train_index)
    position = select(dataset['position'],train_index)
    params = select(dataset['params'],train_index)
    train = data_prep.fit_transform(time=time, position=position, params=params)
    
    time = select(dataset['time'], test_index)
    position = select(dataset['position'],test_index)
    params = select(dataset['params'],test_index)
    test = data_prep.transform(time=time, position=position, params=params)

    config = Config()
    if args.adversary_loss_weight > 0:
        config.use_debias = True
        config.adversary_loss_weight = args.adversary_loss_weight
        name = f'adv-{args.adversary_loss_weight}'
    else:
        config.use_debias = False
        config.adversary_loss_weight = args.adversary_loss_weight
        name = 'normal'
        
    output_folder = os.path.join('models',name)
    os.makedirs(output_folder, exist_ok=True)
    
    running_loss , model = train_model(train, test, config, name=name)
    reg_loss = running_loss[0]['reg']
    adv_loss = running_loss[0]['adv']
    eval_loss = running_loss[1]
        
    y_trues = []
    y_pred_1s = []
    y_pred_2s = []
    select = lambda x,idx: [x[i] for i in idx]
    for ii in range(test_size):
        time = select(dataset['time'], test_index)[ii]
        position = select(dataset['position'], test_index)[ii]
        params = select(dataset['params'], test_index)[ii]
        time, x0, params, y_true = data_prep.transform(time=time, position=position, params=params)
    
        y_pred_1 = inference(time, x0, params, model, config.device, mode=1)
        y_pred_2 = inference(time, x0, params, model, config.device, mode=2)
        y_trues.append(y_true.reshape(-1))
        y_pred_1s.append(y_pred_1)
        y_pred_2s.append(y_pred_2)
    y_trues = np.concatenate(y_trues, axis=0)
    y_pred_1s = np.concatenate(y_pred_1s, axis=0)
    y_pred_2s = np.concatenate(y_pred_2s, axis=0)
        
    data = (reg_loss, adv_loss, eval_loss, y_trues, y_pred_1s, y_pred_2s)
    pickle.dump(data, open(os.path.join(output_folder,f"results.pl"), 'wb'))
    