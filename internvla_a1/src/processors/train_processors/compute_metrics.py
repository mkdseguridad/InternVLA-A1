import numpy as np
from transformers.trainer_utils import EvalPrediction

def compute_metrics(evaluation_prediction: EvalPrediction, **kwargs):
    """
    Compute metrics for evaluation.

    Args:
        evaluation_prediction: EvalPrediction

    Returns:
        metrics: Dict[str, float]
    """
    action_pred = evaluation_prediction.predictions
    action_label = evaluation_prediction.label_ids

    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]

    # Reshape both arrays to 2D for easier processing
    action_label_flat = action_label.reshape(-1, action_label.shape[-1])[:, 6:]
    action_pred_flat = action_pred.reshape(-1, action_pred.shape[-1])[:, 6:]

    # =======================================
    # chunk level, filter the padding actions
    # =======================================
    valid_mask = action_label_flat != -100

    # Calculate absolute difference
    diff = np.abs(action_label_flat - action_pred_flat)
    diff = diff[valid_mask]

    # get the percentage of diff lower than threshold for all action dimensions
    accuracies = np.zeros(len(thresholds))
    for idx, threshold in enumerate(thresholds):
        # Check if all dimensions are below threshold for each valid action
        accuracy = np.mean(diff < threshold)
        accuracies[idx] = accuracy

    metrics = {f"chunk_acc_{threshold}": accuracy for threshold, accuracy in zip(thresholds, accuracies)}

    # =============================================
    # next 1 step level, filter the padding actions
    # =============================================
    action_label_flat_next_1_step = evaluation_prediction.label_ids[:, 0, :]
    action_pred_flat_next_1_step = evaluation_prediction.predictions[:, 0, :]

    # Reshape both arrays to 2D for easier processing
    action_label_flat_next_1_step = action_label_flat_next_1_step.reshape(-1, action_label.shape[-1])
    action_pred_flat_next_1_step = action_pred_flat_next_1_step.reshape(-1, action_pred.shape[-1])

    diff_next_1_step = np.abs(action_label_flat_next_1_step - action_pred_flat_next_1_step)

    for idx, threshold in enumerate(thresholds):
        accuracy = np.mean(diff_next_1_step < threshold)
        metrics[f"next_1_step_acc_{threshold}"] = accuracy

    # =============================================
    # world model level, filter the padding actions
    # =============================================
    pred_wm_indices = evaluation_prediction.pred_wm_indices
    gt_wm_indices = evaluation_prediction.gt_wm_indices

    gen_acc_mean = np.mean(pred_wm_indices == gt_wm_indices)
    metrics["gen_acc_mean"] = gen_acc_mean

    return metrics
