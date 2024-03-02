import pandas as pd
import numpy as np

def read_training_data(file_path):
    # read file
    if file_path.endswith('.csv'):
        df_data = pd.read_csv(file_path,
                              index_col=0)
    elif file_path.endswith('.txt'):
        df_data = pd.read_csv(file_path,
                              index_col=0,
                              sep='\t')
        
    df_data["LFC"] = df_data["LFC"].astype(np.float64)

    # create y_value
    x1 = -0.3
    y1 = 0.7
    x2 = 0
    y2 = 0.3
    param_n = 1
    param_a = (np.log(1.0/(1-y2)-1) - np.log(1.0/(1-y1)-1))/(param_n*x1 - param_n*x2)
    param_b = -1*np.log(1.0/(1-y1)-1)/param_a - param_n*x1
    df_data['y_value'] = [1 - 1/(1+np.exp(-1*param_a*(param_n*i+param_b))) for i in df_data['LFC'].to_list()]
    df_data['label'] = df_data['LFC'].apply(lambda x: 1 if x <= -0.5 else 0)
    return df_data

df = read_training_data("DeepCas13_training_data.csv")

df[["seq", "label"]].to_csv("data/training_data_10k.csv", header=False, index=False)
