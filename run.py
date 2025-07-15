import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from utils import fit, split_sequence, sim, f, train
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

mlflow.set_tracking_uri(uri="http://localhost:8080")

def fit_and_predict(params = {}):
    data = sim(params["n"], 200) 
    #y = data.loc[data["ds"] < 100, "y"].to_numpy()
    #data.y = torch.tensor(data.y.to_numpy())
    # normalize together?
    #data["y"] = (torch.tensor(data["y"].to_numpy()) - data["y"].mean()) / torch.sqrt(torch.tensor([data["y"].var()]))
    splits = data.groupby("unique_id")["y"].apply(
        #lambda y: split_sequence((torch.tensor(y.to_numpy()) - torch.tensor([y.mean()])) / torch.sqrt(torch.tensor([y.var()])), histlen)
        lambda y: split_sequence((y.to_numpy() - y.mean()) / np.sqrt(y.var()), params["histlen"])
        #lambda y: split_sequence(y.to_numpy(), histlen)
    )
    # now do the split
    cutoff = 100
    dtrain = []
    dtest  = []
    for i in range(params["n"]):
        X, y = splits[i]
        # normalize together?
        #X = (X - X.mean()) / torch.sqrt(X.var())
        # (we subtract histlen to get 100 steps of test)
        Xtrain = X[:-cutoff]
        ytrain = y[:-cutoff]
        Xtest = X[-cutoff:]
        ytest = y[-cutoff:]
        ## add id column
        id = torch.ones(len(Xtrain), 1).to(device) * i
        Xtrain = torch.cat((Xtrain, id), dim=1)
        id = torch.ones(len(Xtest), 1).to(device) * i
        Xtest = torch.cat((Xtest, id), dim=1)
        dtrain.append((Xtrain, ytrain))
        dtest.append((Xtest, ytest))
    Xtrain = torch.concat([ d[0] for d in dtrain ]).to(device)
    ytrain = torch.concat([ d[1] for d in dtrain ]).to(device).unsqueeze(1)
    x_data = Variable(Xtrain)
    y_data = Variable(ytrain)
    x_test = torch.concat([ d[0] for d in dtest ]).to(device)
    y_test = torch.concat([ d[1] for d in dtest ]).to(device).unsqueeze(1)
    # train_dataloader = DataLoader([(x, y) for x,y in zip(x_data, y_data)],
    #                               batch_size=params["batch_size"], 
    #                               shuffle=True)
    # test_dataloader = DataLoader([(x, y) for x,y in zip(x_test, y_test)])
    # params["n_in"] = len(x_data[0])
    # model = train(train_dataloader, test_dataloader, params)
    model = fit(x_data, y_data, x_test, y_test, params)
    ## predict and plot
    yfitted = model(x_data).cpu().detach().numpy().flatten()
    yfitted.shape
    data.loc[(data["ds"] >= params["histlen"]) & (data["ds"] < cutoff), "yhat"] = yfitted
    ypred   = model(x_test).cpu().detach().numpy().flatten()
    data.loc[(data["ds"] >= cutoff), "yhat"] = ypred
    make_plots(data)
    # fig, ax = plt.subplots(nrows=3, ncols=3)
    # x = data.ds.unique()
    # data.groupby("unique_id")["y"].apply(lambda y: ax.plot(x, (y.to_numpy() - y.mean()) / np.sqrt(y.var())))
    # #data.groupby("unique_id")["y"].apply( lambda y: ax.plot(x, y))
    # data.groupby("unique_id")["yhat"].apply(lambda y: ax.plot(x, y))
    # plt.savefig("a")

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

    params = {"n": 10, 
              "histlen": 12, 
              "cutoff": 100,
              "n_epoch": 10 * 1000, 
              "batch_size": 128,
              "lr": 1e-3}
    torch.manual_seed(123)
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)
        fit_and_predict(params=params)


if __name__ == "__main__":

    main()

