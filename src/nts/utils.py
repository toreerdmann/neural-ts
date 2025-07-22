import os
from itertools import chain

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from nts.config import Parameters
from nts.data import Standardize, TimeSeriesDataset
from nts.model import MLP


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
    test_joined = list(chain(*[data[i][1] for i in range(data.n)]))
    train_dataloader = DataLoader(
        [
            (
                d[0].to(params.device),
                d[1].to(params.device),
            )
            for d in train_joined
        ],
        batch_size=params.batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        [
            (
                d[0].to(params.device),
                d[1].to(params.device),
            )
            for d in test_joined
        ],
        batch_size=params.batch_size,
        shuffle=True,
    )

    ## get number of features
    n_in = len(train_joined[0][0])
    model = MLP(n_in, params.device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)

    if params.device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 4), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 4), "GB")

    for epoch in range(params.n_epoch):
        model.train()
        total_loss = 0.0
        for batch_idx, (bx, by) in enumerate(train_dataloader):
            pred_y = model(bx)
            optimizer.zero_grad()
            loss = criterion(pred_y, by)
            total_loss += loss
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            model.eval()
            val_loss = sum([criterion(model(x), y) for x, y in test_dataloader])
            print("epoch {}, loss {}, val loss {}".format(epoch, total_loss, val_loss))
            mlflow.log_metrics(
                {"batch_loss": loss.item(), "val_loss": val_loss},
                step=epoch * len(train_dataloader) + batch_idx,
            )

    return model


def predict(model, data: TimeSeriesDataset, params):
    # data_long = data.data.melt(value_name="y")
    data_long = data.data.reset_index().melt(value_name="y", id_vars=["ds"])
    data_long["yhat"] = np.nan
    for i in range(len(data)):
        yfit  = [model(x.to(params.device)) for (x, y) in data[i][0]]
        ypred = [model(x.to(params.device)) for (x, y) in data[i][1]]
        yfit  = torch.concat(yfit).cpu().detach().numpy().flatten()
        ypred = torch.concat(ypred).cpu().detach().numpy().flatten()
        subset_indices = data_long.index[data_long["unique_id"] == i]
        indices_to_modify = subset_indices[data.n_steps:int(data.nT * data.cutoff)]
        data_long.loc[indices_to_modify, "yhat"] = yfit
        indices_to_modify = subset_indices[int(data.nT * data.cutoff) :]
        data_long.loc[indices_to_modify, "yhat"] = ypred
    return data_long

    # train_joined = list(chain(*[data[i][0] for i in range(data.n)]))
    # test_joined = list(chain(*[data[i][1] for i in range(data.n)]))
    # train_dataloader = DataLoader(
    #     [(
    #         d[0],
    #         d[1],
    #     ) for d in train_joined],
    # )
    # test_dataloader = DataLoader(
    #     [(
    #         d[0],
    #         d[1],
    #     ) for d in test_joined],
    # )
    # fitted_values = [model(x) for x,y in train_dataloader]
    # fitted_values = torch.concat(fitted_values).cpu().detach().numpy().flatten()
    # predicted_values = [model(x) for x,y in test_dataloader]
    # predicted_values = torch.concat(predicted_values).cpu().detach().numpy().flatten()
    # return fitted_values, predicted_values
