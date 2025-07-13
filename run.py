import torch
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from utils import fit, split_sequence, sim, f
import numpy as np

mlflow.set_tracking_uri(uri="http://localhost:8080")


def fit_and_predict(n: int = 3,
                    histlen: int = 6,
                    cutoff: int = 100,
                    params = {}):
    data = sim(n, 200) 
    #y = data.loc[data["ds"] < 100, "y"].to_numpy()
    #data.y = torch.tensor(data.y.to_numpy())
    # normalize together?
    #data["y"] = (torch.tensor(data["y"].to_numpy()) - data["y"].mean()) / torch.sqrt(torch.tensor([data["y"].var()]))
    splits = data.groupby("unique_id")["y"].apply(
        #lambda y: split_sequence((torch.tensor(y.to_numpy()) - torch.tensor([y.mean()])) / torch.sqrt(torch.tensor([y.var()])), histlen)
        lambda y: split_sequence((y.to_numpy() - y.mean()) / np.sqrt(y.var()), histlen)
        #lambda y: split_sequence(y.to_numpy(), histlen)
    )
    # now do the split
    cutoff = 100
    dtrain = []
    dtest  = []
    for i in range(n):
        X, y = splits[i]
        # normalize together?
        #X = (X - X.mean()) / torch.sqrt(X.var())
        # (we subtract histlen to get 100 steps of test)
        Xtrain = X[:-cutoff]
        ytrain = y[:-cutoff]
        Xtest = X[-cutoff:]
        ytest = y[-cutoff:]
        ## add id column
        id = torch.ones(len(Xtrain), 1) * i
        Xtrain = torch.cat((Xtrain, id), dim=1)
        id = torch.ones(len(Xtest), 1) * i
        Xtest = torch.cat((Xtest, id), dim=1)
        dtrain.append((Xtrain, ytrain))
        dtest.append((Xtest, ytest))
        Xtrain = torch.concat([ d[0] for d in dtrain ])
        ytrain = torch.concat([ d[1] for d in dtrain ])
        x_data = Variable(Xtrain)
        y_data = Variable(ytrain)
        x_test = torch.concat([ d[0] for d in dtest ])
        y_test = torch.concat([ d[1] for d in dtest ])
        model = fit(x_data, y_data, x_test, y_test, params)
        ## predict and plot
        model.eval()
        yfitted = model(x_data).detach().numpy().flatten()
        yfitted.shape
        data.loc[(data["ds"] >= histlen) & (data["ds"] < cutoff), "yhat"] = yfitted
        ypred   = model(x_test).detach().numpy().flatten()
        data.loc[(data["ds"] >= cutoff), "yhat"] = ypred
        fig, ax = plt.subplots()
        x = data.ds.unique()
        data.groupby("unique_id")["y"].apply(lambda y: ax.plot(x, (y.to_numpy() - y.mean()) / np.sqrt(y.var())))
        #data.groupby("unique_id")["y"].apply( lambda y: ax.plot(x, y))
        data.groupby("unique_id")["yhat"].apply(lambda y: ax.plot(x, y))
        plt.savefig("a")
        return None


def main():

    params = {"n": 3, "histlen": 12, 
              "n_epoch": 5 * 1000, 
              "batch_size": 1,
              "lr": 1e-3}
    torch.manual_seed(123)
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)
        fit_and_predict(n=params["n"], histlen=params["histlen"], params=params)


if __name__ == "__main__":

    print("Hi from run.py")
    main()

