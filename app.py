import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import src.fastapi.utils as utils
from src.fastapi.network import Lift_base_network

app = FastAPI()

@app.post("/data/")
def data_prep(data: UploadFile = File(...)):
    
    x_min = np.loadtxt("src/fastapi/min_.csv")
    x_range = np.loadtxt("src/fastapi/range_.csv")

    file_location = f"src/fastapi/data/{data.filename}"
    with open(file_location, "wb+") as raw_data:
        raw_data.write(data.file.read())
    model = Lift_base_network()
    model.load_state_dict(torch.load("src/fastapi/model_state_dict_25Nov"))
    Data = utils.preprocess(file_location, x_min, x_range)
    Cl_map = utils.inference(Data, model, x_min, x_range)
    np.savetxt("src/fastapi/output/Cl_map.csv",Cl_map)

