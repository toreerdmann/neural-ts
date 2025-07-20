import mlflow
import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils import fit_and_predict


class Settings(BaseSettings):
    model_config = SettingsConfigDict()
    n: int = 10
    histlen: int = 12
    cutoff: int = 100
    n_epoch: int = 2 * 1000
    batch_size: int = 128
    lr: float = 1e-3
    device: torch.device = "cuda"

    @field_validator("device", mode="before")
    @classmethod
    def val_dev(cls, value: str) -> torch.device:
        if value == "cuda":
            return torch.device("cuda") if torch.cuda.is_available() else "cpu"
        else:
            return torch.device("cpu")


mlflow.set_tracking_uri(uri="http://localhost:8080")


def main():
    params = Settings()
    torch.manual_seed(123)
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params.dict())
        fit_and_predict(params=params)


if __name__ == "__main__":
    main()
