from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self, n_in: int = 1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(n_in, 1)  # One in and one out
        ## normal initialization method
        nn.init.normal_(self.linear.weight, mean=0, std=10)

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
            nn.Linear(64, 1),
        ).to(device)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight)

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)
