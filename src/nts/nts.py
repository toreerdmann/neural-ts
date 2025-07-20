# import mlflow
# import torch
#
# from nts.utils import fit_and_predict
# from nts.config import Parameters
#
# def main():
#     mlflow.set_tracking_uri(uri="http://localhost:8080")
#     params = Parameters()
#     torch.manual_seed(123)
#     with mlflow.start_run():
#         ## log the hyperparameters
#         mlflow.log_params(params.dict())
#         ## simulate data
#         data = TimeSeriesDataset(params.n, params.nT, transform=Standardize())
#         ## train and predict
#         model = train(data, params)
#         data_long = predict(model, data, params)
#         ## save plot
#         make_plots(data_long)

