import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np


class HDD(Dataset):
    def __init__(self, data_file, transform=None):
        self.xy = np.array(data_file)
        self.n_samples = self.xy.shape[0]
        self.x_data = torch.from_numpy(self.xy) 
        self.y_data = torch.from_numpy(self.xy[:, [-1]])
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.x_data)
    
    def shape(self):
        return self.xy.shape
    

def get_data(data_path) -> tuple:
    dataset = HDD(data_file=data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataset, dataloader

