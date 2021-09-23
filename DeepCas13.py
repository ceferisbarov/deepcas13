################################################################
# Script Name : DeepCas13.py
# Description : Functions for DeepCas13 model
# Author      : Xiaolong Cheng from Dr. Wei Li Lab
# Affiliation : Children's National Hospital
# Email       : xcheng@childrensnational.org
################################################################


import pandas as pd
import numpy as np
import RNA
import os

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv2D, SpatialDropout2D
from tensorflow.keras.layers import Input, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Reshape
from tensorflow.keras.layers import Dropout, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import concatenate, TimeDistributed
from tensorflow.keras.layers import Activation, Embedding, GRU, LSTM, Bidirectional, SpatialDropout1D, SimpleRNN
import tensorflow.keras.backend as K
import tensorflow as tf


def get_fold(seq):
    for i in range(33-len(seq)):
        seq = seq + 'N'
    fc = RNA.fold_compound(seq)
    # compute MFE and MFE structure
    (mfe_struct, mfe) = fc.mfe()
    return mfe_struct

def seq_one_hot_code(seq):
    seq = seq.upper()
    lst_seq = list(seq)
    lst_seq.extend(['N' for i in range(33-len(seq))])
    df_onehot = pd.DataFrame(list(lst_seq), columns=['NT'])
    for col_feat in ['A','C','G','T','N']:
        df_onehot[col_feat]=df_onehot['NT'].apply(lambda x:1 if x==col_feat else 0)
    df_onehot = df_onehot.set_index('NT')
    df_onehot = df_onehot.drop(columns=['N'])
    return df_onehot.values

def fold_one_hot_code(seq):
    lst_seq = list(seq)
    lst_seq.extend(['N' for i in range(33-len(seq))])
    df_onehot = pd.DataFrame(list(lst_seq), columns=['NT'])
    for col_feat in ['(',')','.','N']:
        df_onehot[col_feat]=df_onehot['NT'].apply(lambda x:1 if x==col_feat else 0)
    df_onehot = df_onehot.set_index('NT')
    df_onehot = df_onehot.drop(columns=['N'])
    return df_onehot.values

def DL_Multi_Models(df_train, df_yvalue):
    from sklearn.model_selection import KFold
    lst_model = []
    kf = KFold(n_splits=5, shuffle=True, random_state=32)
    for train, test in kf.split(df_train):
        X_train = df_train.iloc[list(train),:]
        y_train = df_yvalue.iloc[list(train),0]
        X_train_seq = [seq_one_hot_code(i) for i in X_train.seq.to_list()]
        X_train_fold = [fold_one_hot_code(get_fold(i)) for i in X_train.seq.to_list()]
        ###
        X_train_arr_seq = np.array(X_train_seq)
        X_train_arr_fold = np.array(X_train_fold)
        ###
        X_train_seq_CNN = np.reshape(X_train_arr_seq, (len(X_train_arr_seq), 1, 33, 4, 1)) 
        X_train_fold_CNN = np.reshape(X_train_arr_fold, (len(X_train_arr_fold), 1, 33, 3, 1))
        ## Seq
        seq_input = Input(shape=(1, 33, 4, 1))
        seq_conv1 = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(seq_input)
        seq_norm1 = TimeDistributed(BatchNormalization())(seq_conv1)
        seq_conv2 = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(seq_norm1)
        seq_norm2 = TimeDistributed(BatchNormalization())(seq_conv2)
        seq_drop1 = TimeDistributed(Dropout(0.5))(seq_norm2)
        seq_pool1 = TimeDistributed(MaxPooling2D((2, 2)))(seq_drop1)
        seq_flat1 = TimeDistributed(Flatten())(seq_pool1)
        seq_lstm1 = LSTM(100)(seq_flat1)
        seq_drop2 = Dropout(0.3)(seq_lstm1)
        seq_output = Dense(64, activation='relu')(seq_drop2)
        ## Fold
        fold_input = Input(shape=(1, 33, 3, 1))
        fold_conv1 = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(fold_input)
        fold_norm1 = TimeDistributed(BatchNormalization())(fold_conv1)
        fold_conv2 = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(fold_norm1)
        fold_norm2 = TimeDistributed(BatchNormalization())(fold_conv2)
        fold_drop1 = TimeDistributed(Dropout(0.5))(fold_norm2)
        fold_pool1 = TimeDistributed(MaxPooling2D((2, 2)))(fold_drop1)
        fold_flat1 = TimeDistributed(Flatten())(fold_pool1)
        fold_lstm1 = LSTM(100)(fold_flat1)
        fold_drop2 = Dropout(0.3)(fold_lstm1)
        fold_output = Dense(64, activation='relu')(fold_drop2)
        ## Merge
        merged = concatenate([seq_output, fold_output], axis=1, name='merged')
        NN_drop1 = Dropout(0.3)(merged)
        NN_dense2 = Dense(64)(NN_drop1)
        NN_output = Dense(1, activation='sigmoid')(NN_dense2)
        ###
        NN_model = Model([seq_input, fold_input], NN_output)
        NN_model.compile(optimizer='Adam', loss='mse')
        ###
        NN_model.fit([X_train_seq_CNN, X_train_fold_CNN],  y_train, epochs=30, batch_size=128, shuffle=True, verbose=0)
        lst_model.append(NN_model)
    ###
    return lst_model

def train_and_predict(X_train, y_train, X_test):
    lst_model = DL_Multi_Models(X_train, y_train)
    ###
    X_test_seq = [seq_one_hot_code(i) for i in X_test.seq.to_list()]
    X_test_fold = [fold_one_hot_code(get_fold(i)) for i in X_test.seq.to_list()]
    ###
    X_test_arr_seq = np.array(X_test_seq)
    X_test_arr_fold = np.array(X_test_fold)
    ###
    X_test_seq_CNN = np.reshape(X_test_arr_seq, (len(X_test_arr_seq), 1, 33, 4, 1)) 
    X_test_fold_CNN = np.reshape(X_test_arr_fold, (len(X_test_arr_fold), 1, 33, 3, 1))
    ###
    df_rzlt = pd.DataFrame(index=X_test.index, columns=['Deep Score', 'M0', 'M1', 'M2', 'M3', 'M4'])
    for k in range(5):
        y_pred = lst_model[k].predict([X_test_seq_CNN, X_test_fold_CNN])
        df_rzlt['M'+str(k)] = [i[0] for i in y_pred]
    df_rzlt['Deep Score'] = df_rzlt[['M0', 'M1', 'M2', 'M3', 'M4']].mean(axis=1)
    return df_rzlt['Deep Score']

