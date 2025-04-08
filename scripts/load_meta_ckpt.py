import os
import torch
import pickle

input_path = os.path.join("..","ckpt","maml_ilr_0.010000_olr_0.000010_bs_3_53000_31.46_30_2_256_10.pkl")
with open(input_path, 'rb') as file:
    init = pickle.load(file)

features = len(init['siren__model/siren_layer/linear']['b'])
print('features: ', features)
