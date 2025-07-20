import mlflow
import torch

from nts.utils import fit_and_predict
from nts.config import Parameters


mlflow.set_tracking_uri(uri="http://localhost:8080")


def main():
    params = Parameters()
    torch.manual_seed(123)
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params.dict())
        fit_and_predict(params=params)


if __name__ == "__main__":
    main()
