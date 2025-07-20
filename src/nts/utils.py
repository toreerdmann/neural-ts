import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable

from itertools import chain

from nts.model import MLP
from nts.data import TimeSeriesDataset, Standardize
from torch.utils.data import DataLoader
from nts.config import Parameters

def make_plots(data: pd.DataFrame):
    figs = "figs"
    if not os.path.exists(figs):
        os.makedirs(figs)
    x = data.ds.unique()
    ids = data.unique_id.unique()
    for id in ids:
        fig, ax = plt.subplots()
        y = data.loc[data["unique_id"] == id]["y"]
        yhat = data.loc[data["unique_id"] == id]["yhat"]
        ax.plot(x, (y - y.mean()) / np.sqrt(y.var()))
        ax.plot(x, (yhat - yhat.mean()) / np.sqrt(yhat.var()))
        plt.savefig(f"figs/{id}.png")



def train(data, params):

    ## concat the samples
    train_joined = list(chain(*[data[i][0] for i in range(data.n)]))
    train_dataloader = DataLoader(
        train_joined,
        batch_size=params.batch_size, shuffle=True
    )
    ## get number of features
    n_in = len(train_joined[0][0])
    model = MLP(n_in, params.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = params.lr)
    if params.device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 4), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 4), "GB")
    for epoch in range(params.n_epoch):
        model.train()
        for batch_idx, (bx, by) in enumerate(train_dataloader):
            pred_y = model(bx)
            optimizer.zero_grad()
            loss = criterion(pred_y, by)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            model.eval()
            val_loss = 0
            if x_test is not None:
                val_loss = criterion(
                    model(x_test).flatten(),
                    # torch.tensor(y_test)
                    y_test,
                ).tolist()
            print("epoch {}, loss {}, val loss {}".format(epoch, loss.item(), val_loss))
            mlflow.log_metrics(
                {"batch_loss": loss.item(), "val_loss": val_loss},
                step=epoch * len(train_dataloader) + batch_idx,
            )
    return model


def preprocess(data, params):
    splits = data.groupby("unique_id")["y"].apply(
        lambda y: split_sequence(
            (y.to_numpy() - y.mean()) / np.sqrt(y.var()), params.histlen, params.device
        )
    )
    # now do the split
    dtrain = []
    dtest = []
    for i in range(params.n):
        X, y = splits[i]
        Xtrain = X[: -params.cutoff]
        ytrain = y[: -params.cutoff]
        Xtest = X[-params.cutoff :]
        ytest = y[-params.cutoff :]
        ## add id column
        id = torch.ones(len(Xtrain), 1).to(params.device) * i
        Xtrain = torch.cat((Xtrain, id), dim=1)
        id = torch.ones(len(Xtest), 1).to(params.device) * i
        Xtest = torch.cat((Xtest, id), dim=1)
        dtrain.append((Xtrain, ytrain))
        dtest.append((Xtest, ytest))
    Xtrain = torch.concat([d[0] for d in dtrain]).to(params.device)
    ytrain = torch.concat([d[1] for d in dtrain]).unsqueeze(1).to(params.device)
    x_train = Variable(Xtrain)
    y_train = Variable(ytrain)
    x_test = torch.concat([d[0] for d in dtest]).to(params.device)
    y_test = torch.concat([d[1] for d in dtest]).unsqueeze(1).to(params.device)
    train_dataloader = DataLoader(
        [(x, y) for x, y in zip(x_train, y_test)],
        batch_size=params.batch_size,
        shuffle=True,
    )
    return train_dataloader, x_test, y_test


def fit_and_predict(params):

    data = TimeSeriesDataset(params.n, params.nT, transform=Standardize())
    train, test = data[0]


    model = train(train_dataloader, params, x_test, y_test)
    ## add fitted values

    yfit = model(torch.concat([x for x,y in train_dataloader]))

    xtrain = torch.concat([x for x,y in train_dataloader])

    data

    xtrain.shape

    model()

    [y for x,y in train_dataloader]
    yfit


    yfitted = model(x_train).cpu().detach().numpy().flatten()
    data.loc[(data["ds"] >= params.histlen) & (data["ds"] < params.cutoff), "yhat"] = (
        yfitted
    )
    ## add predicted values
    ypred = model(x_test).cpu().detach().numpy().flatten()
    data.loc[(data["ds"] >= params.cutoff), "yhat"] = ypred
    ## save plot
    make_plots(data)
