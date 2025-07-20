import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

import random

import mlflow

import pandas as pd
import numpy as np

def split_sequence(sequence, n_steps, device):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return torch.tensor(np.array(X)).to(device), torch.tensor(y).to(device)


def f(ds, beta = torch.tensor([0., 0., 0., 0.]), sig = .01):
    x = (ds % 12) / 12 * torch.pi * 2
    X = torch.vstack([torch.ones(len(x)), 
                      ds, 
                      torch.sin(x), 
                      torch.sin(4*x)]).T
    return X, torch.matmul(X, beta) + sig * torch.randn(len(ds))

def sim(n: int = 10, l: int = 36):
    dfs = []
    for i in range(n):
        ds = torch.arange(.0, l)
        X, y = f(ds, torch.tensor([1., .1, 1., 1.]) * torch.randn(4), 
              torch.rand(1))
        dfs.append(pd.DataFrame({"ds": ds, 
                                 "y": y, 
                                 "unique_id": i}))
    return pd.concat(dfs)


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, n_in: int = 1):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(n_in, 1)  # One in and one out
        ## normal initialization method
        torch.nn.init.normal_(self.linear.weight, mean=0, std=10)
    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, n_in, device):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                #torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
        self.layers.apply(init_weights)
    def forward(self, x):
        return self.layers(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        optimizer.zero_grad()
        loss = loss_fn(pred, y)
        optimizer.step()
        #if batch % 100 == 0:
        loss, current = loss.item(), batch * dataloader.batch_size + len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    # print(f"Avg loss: {test_loss:>8f} \n")
    mlflow.log_metrics({
        "test_loss": test_loss, 
    })
def train(train_dataloader, test_dataloader, params):
    n_in = params.n_in
    model = MLP(n_in, params.device)
    loss_fn = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr = params.lr)
    for t in range(params.n_epoch):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        if t % 100 == 0:
            test_loop(test_dataloader, model, loss_fn)
    return model


def fit(x_data, y_data, params, x_test=None, y_test=None):
    n_in = len(x_data[0])
    # our model
    # model = LinearRegressionModel(n_in)
    model = MLP(n_in, params.device)
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    training_data = [(x, y) 
        for x,y in zip(x_data, y_data)]
    train_dataloader = DataLoader(training_data,
                                  batch_size=params.batch_size, 
                                  shuffle=True)
    # train_features, train_labels = next(iter(train_dataloader))
    if params.device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,4), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,4), 'GB')
    model.train()
    for epoch in range(params.n_epoch):
        for batch_idx, (bx, by) in enumerate(train_dataloader):
            pred_y = model(bx)
            optimizer.zero_grad()
            loss = criterion(pred_y, by)
            loss.backward()
            ## clips if L2-norm of all gradients > 1.0
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
            optimizer.step()
        if epoch % 100 == 0:
            val_loss = 0
            if x_test is not None:
                val_loss = criterion(
                    model(x_test).flatten(), 
                    #torch.tensor(y_test)
                    y_test,
                ).tolist()
            print('epoch {}, loss {}, val loss {}'.format(
                epoch, 
                loss.item(),
                val_loss
            ))
            mlflow.log_metrics({
                "batch_loss": loss.item(), 
                "val_loss": val_loss
            }, step=epoch * len(train_dataloader) + batch_idx)

    return model

if __name__ == "main":

    x_data = Variable(torch.Tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]))
    y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
    model = fit(x_data, y_data)

    new_var = Variable(torch.Tensor([5.0, 1.0]))
    pred_y = model(new_var)
    print("predict (after training)", 4, model(new_var).item())
