import torch
import gc
import numpy as np
from typing import Dict
from transformers import EvalPrediction


def compute_regression_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Memory-efficient computation of regression metrics using PyTorch operations.
    Fixes inconsistent prediction shapes from HuggingFace Trainer.
    """
    predictions, labels = eval_pred

    # --- ✅ 修复关键点 ---
    # 如果 predictions 是 list/tuple，则取第一个元素（防止 pred, aux_loss, gate_weights 一起传入）
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]

    # 转成 numpy 确保维度规整
    predictions = np.array(predictions)

    # 若为二维且第二维大于1（如 [B,2]），只取第一列（即主预测值 pred）
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = predictions[:, 0]

    # 转为 torch.Tensor
    predictions = torch.tensor(predictions, dtype=torch.float32)
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

    # 清理显存
    del predictions, labels, diff, mse, rmse, mae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
