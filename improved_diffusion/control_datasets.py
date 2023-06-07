import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd

def load_data(filename):
    D = []
    data = pd.read_csv(filename, sep=',', header='infer',encoding='utf-8')
    for index, row in data.iterrows():
         D.append((row['ref'], row['ref']))
    return D