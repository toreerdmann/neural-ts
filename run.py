import torch
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from utils import fit, split_sequence, sim
import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict()
    n: int = 10
    histlen: int = 12
    cutoff: int = 100
    n_epoch: int = 2 * 1000
    batch_size: int = 128
    lr: float = 1e-3
    device: torch.device = "cuda"
    @field_validator('device', mode='before')
    @classmethod
    def val_dev(cls, value: str) -> torch.device:
        if value == "cuda":
            return torch.device("cuda") if torch.cuda.is_available() else "cpu"
        else:
            return torch.device("cpu")

settings = Settings()

mlflow.set_tracking_uri(uri="http://localhost:8080")

def fit_and_predict(params):
    data = sim(params.n, 200) 
    splits = data.groupby("unique_id")["y"].apply(
        lambda y: split_sequence(
            (y.to_numpy() - y.mean()) / np.sqrt(y.var()), 
            params.histlen, 
            params.device
        )
    )
    # now do the split
    cutoff = 100
    dtrain = []
    dtest  = []
    for i in range(params.n):
        X, y = splits[i]
        # normalize together?
        #X = (X - X.mean()) / torch.sqrt(X.var())
        # (we subtract histlen to get 100 steps of test)
        Xtrain = X[:-cutoff]
        ytrain = y[:-cutoff]
        Xtest = X[-cutoff:]
        ytest = y[-cutoff:]
        ## add id column
        id = torch.ones(len(Xtrain), 1).to(params.device) * i
        Xtrain = torch.cat((Xtrain, id), dim=1)
        id = torch.ones(len(Xtest), 1).to(params.device) * i
        Xtest = torch.cat((Xtest, id), dim=1)
        dtrain.append((Xtrain, ytrain))
        dtest.append((Xtest, ytest))
    Xtrain = torch.concat([ d[0] for d in dtrain ]).to(params.device)
    ytrain = torch.concat([ d[1] for d in dtrain ]).unsqueeze(1).to(params.device)
    x_data = Variable(Xtrain)
    y_data = Variable(ytrain)
    x_test = torch.concat([ d[0] for d in dtest ]).to(params.device)
    print("predict (after training)", 4, model(new_var).item())
    y_test = torch.concat([ d[1] for d in dtest ]).unsqueeze(1).to(params.device)
    model = fit(x_data, y_data, params, x_test, y_test)
    ## predict and plot
    yfitted = model(x_data).cpu().detach().numpy().flatten()
    yfitted.shape
    data.loc[(data["ds"] >= params.histlen) & (data["ds"] < cutoff), "yhat"] = yfitted
    ypred   = model(x_test).cpu().detach().numpy().flatten()
    data.loc[(data["ds"] >= cutoff), "yhat"] = ypred
    make_plots(data)

def make_plots(data: pd.DataFrame):
    x = data.ds.unique()
    ids = data.unique_id.unique()
    for id in ids:
        fig, ax = plt.subplots()
        y = data.loc[data["unique_id"] == id]["y"]
        yhat = data.loc[data["unique_id"] == id]["yhat"]
        ax.plot(x, (y - y.mean()) / np.sqrt(y.var()))
        ax.plot(x, (yhat - yhat.mean()) / np.sqrt(yhat.var()))
        plt.savefig(f"figs/{id}.png")


def main():

    params = settings
    torch.manual_seed(123)
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params.dict())
        fit_and_predict(params=params)


if __name__ == "__main__":

    main()

