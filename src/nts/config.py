import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Parameters(BaseSettings):
    model_config = SettingsConfigDict()
    n: int = 10
    nT: int = 200
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
