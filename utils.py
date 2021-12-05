import numpy as np
import pandas as pd
import torch

def preprocess(Data_addr: str, x_min, x_range):
    Data = pd.read_csv(Data_addr, header = None, delimiter=',')
    Data[4] = Data[4]/Data[0]
    Data = (Data - x_min[:-2])/x_range[:-2]
    return Data

def inference(Data, model, x_min, x_range):
    Data_torch = torch.from_numpy(Data.to_numpy())
    Cl = model(Data_torch.float())
    Cl_map = torch.zeros_like(Cl)
    Cl_map[:, 0] = Cl[:, 0] * x_range[-2] + x_min[-2]
    Cl_map[:, 1] = Cl[:, 1] * x_range[-1] + x_min[-1]
    Cl_map = Cl_map.cpu().detach().numpy()
    return Cl_map