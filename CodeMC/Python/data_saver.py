"""
Date: 16/04/2020
Last Modification: 16/04/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to read and save data.
"""
import numpy as np

def save_table(table, index, folder='.'):
    np.save('./NN_tables/'+folder+'/table_{}.pkl.npy'.format(index), table, allow_pickle = True)

def load_table(index, folder='.'):
    return np.load('./NN_tables/'+folder+'/table_{}.pkl.npy'.format(index),allow_pickle = True)
