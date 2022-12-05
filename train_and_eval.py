from train_utils import load_dataset, dataset_preprocessing
from sklearn.model_selection import train_test_split
from train_utils import evaluate, train_model, Config
import pandas as pd
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversary_loss_weight', type=float, default=-1.0)
    args = parser.parse_args()
    
    dataset = load_dataset()
    data = dataset_preprocessing(dataset)
    data = train_test_split(*data, test_size=0.2)
    train, test = data[::2],data[1::2]

    config = Config()
    if args.adversary_loss_weight > 0:
        config.use_debias = True
        config.adversary_loss_weight = args.adversary_loss_weight
        name = f'adv-{args.adversary_loss_weight}'
    else:
        config.use_debias = False
        config.adversary_loss_weight = args.adversary_loss_weight
        name = 'normal'
        
    running_loss , model = train_model(train, test, config, name=name)

    y_trues = []
    y_pred_1s = []
    y_pred_2s = []
    for time, params, position in zip(dataset['time'],dataset['params'],dataset['position']):
        y_true, y_pred_1 = evaluate(time, position, params, model, config.device, mode=1)
        y_true, y_pred_2 = evaluate(time, position, params, model, config.device, mode=2)
        y_trues.append(y_true)
        y_pred_1s.append(y_pred_1)
        y_pred_2s.append(y_pred_2)
    y_trues = np.concatenate(y_trues, axis=0)
    y_pred_1s = np.concatenate(y_pred_1s, axis=0)
    y_pred_2s = np.concatenate(y_pred_2s, axis=0)
        
    df = pd.DataFrame()
    df['y_true'] = y_trues
    df['y_pred_1s'] = y_pred_1s
    df['y_pred_2s'] = y_pred_2s
                    
    df.to_csv(f"{name}.csv", index=False)