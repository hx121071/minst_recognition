import numpy as np
import pandas as pd

def get_db(filename):

    data=pd.read_csv(filename)
    data_array=data.values
    if filename.find('train')!=-1:
        X=data_array[:,1:]
        y=data_array[:,0]
        X=X.reshape((-1,28,28,1))
        return X,y
    else:
        X=data_array.reshape((-1,28,28,1))
        return X
