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
    parser.add_argument('--adversary_loss_weight', type=float, default=-1.0)
    args = parser.parse_args()
    
    dataset = load_dataset()
    data_size = len(dataset['time'])
    indexes = list(range(data_size))
    random.shuffle(indexes)
    train_size = int(0.8*data_size)
    test_size = data_size - train_size
    train_index = indexes[:train_size]
    test_index = indexes[train_size:]
    print(f"train_size: {len(train_index)}\ntest_size: {len(test_index)}")
    data_prep = DataPreprocessor()

    def select_dataset(dataset, idx):
        select = lambda x,idx: np.concatenate([x[i] for i in idx])
        time = select(dataset['time'], idx)
        position = select(dataset['position'],idx)
        params = select(dataset['params'],idx)
        return time, position, params

    train = data_prep.fit_transform(*select_dataset(dataset, train_index))
    test = data_prep.transform(*select_dataset(dataset, test_index))

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
    
    X_train = np.concatenate(train[:3],axis=1)
    X_test = np.concatenate(test[:3],axis=1)
    y_train, next_time_train = train[-2:]
    y_test, next_time_test = test[-2:]
    print(f'X_train: {X_train.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'y_test: {y_test.shape}')
    print(f'next_time_train: {next_time_train.shape}')
    print(f'next_time_test: {next_time_test.shape}')
    
    
    running_loss , model = train_model(X_train, X_test, y_train, y_test, next_time_train, next_time_test, config, name=name)
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
        prep_dataset = data_prep.transform(time=time, position=position, params=params)
        y_true = prep_dataset[-2]
        
        y_pred_1 = inference(*prep_dataset[:3], model, config.device, mode=1)
        y_pred_2 = inference(*prep_dataset[:3], model, config.device, mode=2)
        y_trues.append(y_true.reshape(-1))
        y_pred_1s.append(y_pred_1)
        y_pred_2s.append(y_pred_2)
    y_trues = np.concatenate(y_trues, axis=0)
    y_pred_1s = np.concatenate(y_pred_1s, axis=0)
    y_pred_2s = np.concatenate(y_pred_2s, axis=0)
        
    data = (reg_loss, adv_loss, eval_loss, y_trues, y_pred_1s, y_pred_2s)
    pickle.dump(data, open(os.path.join(output_folder,f"results.pl"), 'wb'))
    