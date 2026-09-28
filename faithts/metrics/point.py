import torch


def mae(pred, true): return (pred - true).abs().mean().item()
def mse(pred, true): return (pred - true).pow(2).mean().item()
def rmse(pred, true): return mse(pred, true) ** 0.5


def mape(pred, true, eps=1e-3):
    return ((pred - true).abs() / (true.abs() + eps)).mean().item()


def smape(pred, true, eps=1e-3):
    return (2 * (pred - true).abs() / (pred.abs() + true.abs() + eps)).mean().item()


def all_point_metrics(pred, true):
    return dict(MAE=mae(pred, true), MSE=mse(pred, true), RMSE=rmse(pred, true),
                MAPE=mape(pred, true), sMAPE=smape(pred, true))
