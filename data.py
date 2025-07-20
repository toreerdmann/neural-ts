import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


def f(ds, beta=torch.tensor([0.0, 0.0, 0.0, 0.0]), sig=0.01):
    """Generative model for time series data."""
    x = (ds % 12) / 12 * torch.pi * 2
    X = torch.vstack([torch.ones(len(x)), ds, torch.sin(x), torch.sin(4 * x)]).T
    return X, torch.matmul(X, beta) + sig * torch.randn(len(ds))


def sim(n: int = 10, nT: int = 36):
    """Simulate `n` time series of length `nT`."""
    dfs = []
    for i in range(n):
        ds = torch.arange(0.0, nT)
        X, y = f(ds, torch.tensor([1.0, 0.1, 1.0, 1.0]) * torch.randn(4), torch.rand(1))
        dfs.append(pd.DataFrame({"ds": ds, "y": y, "unique_id": i}))
    return pd.concat(dfs)

def split_sequence(sequence, n_steps: int = 20):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), y


class Standardize(object):
    def __init__(self):
        pass
    def __call__(self, y):
        return (y.to_numpy() - y.mean()) / np.sqrt(y.var())


class TimeSeriesDataset(Dataset):
    """Time series dataset"""

    def __init__(self, n, nT, transform=None, n_steps:int = 20, cutoff: float = 0.5, batch_size: int = 32):
        """
        Arguments:
            n  (int): Number of time series.
            nT (int): Number of time points per series.
            transform (callable,optional): Optional transform to perform.
        """
        self.n = n
        self.nT = nT
        self.data = sim(n, nT).pivot(columns="unique_id", index="ds")["y"]
        self.transform = transform
        self.n_steps = n_steps
        self.cutoff = cutoff
        self.batch_size = batch_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        ## create features (just lag values)
        X, y = split_sequence(sample, self.n_steps)

        ## do split
        Xtrain = X[: -int(self.cutoff * self.nT)]
        ytrain = y[: -int(self.cutoff * self.nT)]
        Xtest = X[ -int(self.cutoff * self.nT) :]
        ytest = y[ -int(self.cutoff * self.nT) :]

        # train samples (return like this so we can create the dataloader later)
        train = DataLoader(
            [(
                ## add indicator for idx
                torch.concat((torch.tensor(x), torch.tensor([idx]))), 
                torch.tensor(y)
            )
                for x, y in zip(Xtrain, ytrain)],
            batch_size=self.batch_size, shuffle=True
        )
        test = DataLoader(
            [(
                ## add indicator for idx
                torch.concat((torch.tensor(x), torch.tensor([idx]))), 
                torch.tensor(y)
            )
                for x, y in zip(Xtest, ytest)]
        )
        return train, test
