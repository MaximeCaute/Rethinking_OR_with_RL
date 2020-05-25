"""
Date: 16/04/2020
Last Modification: 16/04/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to read and save data.
"""
import numpy as np

def save_table(table, index, folder='.', relative_path = ""):
    np.save('./'+relative_path+folder+'/table_{}.pkl.npy'.format(index), table, allow_pickle = True)

def load_table(index, folder='.', relative_path = ""):
    return np.load('./'+relative_path+folder+'/table_{}.pkl.npy'.format(index),allow_pickle = True)

def save_model(model_name, folder = '.', relative_path = ""):
    torch.save(model, './'+relative_path+folder+'/model_'+model_name+'.pth')

def load_model(model_name, folder = '.', relative_path = ""):
    return torch.load('./'+relative_path+folder+'/model_'+model_name+'.pth')
