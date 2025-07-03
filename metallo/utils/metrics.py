import torch
import gc
from typing import Dict
from transformers import EvalPrediction


def compute_regression_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Memory-efficient computation of regression metrics using PyTorch operations

    Args:
        eval_pred: EvalPrediction object with predictions and labels

    Returns:
        dict: Dictionary of computed regression metrics (MAE, MSE, RMSE)
    """
    predictions, labels = eval_pred
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)
    if predictions.device != labels.device:
        predictions = predictions.to(labels.device)
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    with torch.no_grad():
        diff = predictions - labels
        mse = torch.mean(diff**2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(diff))
        result = {
            "eval_mae": mae.item(),
            "eval_mse": mse.item(),
            "eval_rmse": rmse.item(),
        }

    del predictions, labels, diff, mse, rmse, mae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
