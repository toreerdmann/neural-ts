import mlflow
import torch
import requests

from nts.config import Parameters
from nts.data import Standardize, TimeSeriesDataset
from nts.utils import make_plots, predict, train

MLFLOW_URI = "http://localhost:8080"

def main():

    try:
        # Check if server is running
        endpoint = "/api/2.0/mlflow/experiments/list"
        response = requests.get(f"{MLFLOW_URI}{endpoint}")
        mlflow.set_tracking_uri(uri=MLFLOW_URI)
    except requests.exceptions.ConnectionError:
        print("need to start mlflow server")
        exit()

    params = Parameters()
    print(params)
    torch.manual_seed(123)
    with mlflow.start_run():
        ## log the hyperparameters
        mlflow.log_params(params.dict())
        ## simulate data
        data = TimeSeriesDataset(params.n, params.nT)# , transform=Standardize())
        ## train and predict
        model = train(data, params)
        data_long = predict(model, data, params)
        ## save plot
        make_plots(data_long)
