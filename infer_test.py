import os
import numpy as np
import pandas as pd
from tensorflow import keras
from deepcas13_test import seq_one_hot_code, fold_one_hot_code, get_fold

## Train model
def read_training_data(file_path):
    # read file
    if file_path.endswith('.csv'):
        df_data = pd.read_csv(file_path, names=['seq', 'LFC'], index_col=None)
    elif file_path.endswith('.txt'):
        df_data = pd.read_csv(file_path, names=['seq', 'LFC'], index_col=None, sep='\t')
    # create y_value
    x1 = -0.3
    y1 = 0.7
    x2 = 0
    y2 = 0.3
    param_n = 1
    param_a = (np.log(1.0/(1-y2)-1) - np.log(1.0/(1-y1)-1))/(param_n*x1 - param_n*x2)
    param_b = -1*np.log(1.0/(1-y1)-1)/param_a - param_n*x1
    df_data['y_value'] = [1 - 1/(1+np.exp(-1*param_a*(param_n*i+param_b))) for i in df_data['LFC'].to_list()]
    return df_data

df_train = read_training_data("data/training_data_10k.csv")

X_train = df_train.iloc[:,:]
y_train = df_train.iloc[:,2]
X_train_seq = [seq_one_hot_code(i) for i in X_train.seq.to_list()]
X_train_fold = [fold_one_hot_code(get_fold(i)) for i in X_train.seq.to_list()]
###
X_train_arr_seq = np.array(X_train_seq)
X_train_arr_fold = np.array(X_train_fold)
###
print(X_train_arr_seq.shape)
X_train_arr_seq = X_train_arr_seq[:, :30, :]
X_train_arr_fold = X_train_arr_fold[:, :30, :]
print(X_train_arr_seq.shape)
X_train_seq_CNN = np.reshape(X_train_arr_seq, (len(X_train_arr_seq), 1, 30, 4, 1)) 
print(X_train_seq_CNN.shape)
X_train_fold_CNN = np.reshape(X_train_arr_fold, (len(X_train_arr_fold), 1, 30, 3, 1))


model = "DL_model_test_jingyi_30"
basename = "DeepCas13_Model"

lst_model = []
for k in range(5):
    lst_model.append(keras.models.load_model(os.path.join(model, basename+str(k))))

    lst_model[0].evaluate(X_train_seq_CNN, y_train)
    break
