import torch

def mase(pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Scaled Error (MASE).

    Args:
        pred_seq: Tensor of shape (batch, n_samples, output_len)
        target_seq: Tensor of shape (batch, output_len)

    Returns:
        mase: Scalar Tensor (mean over batch and samples)
    """
    # Shape checks
    assert pred_seq.ndim == 3
    assert target_seq.ndim == 2
    batch, n_samples, output_len = pred_seq.shape
    assert target_seq.shape == (batch, output_len)

    # Expand target to match pred shape: (batch, 1, output_len) → (batch, n_samples, output_len)
    target_exp = target_seq.unsqueeze(1).expand(-1, n_samples, -1)

    # MAE numerator: prediction error
    mae_pred = torch.abs(pred_seq - target_exp).mean(dim=-1)  # (batch, n_samples)

    # MAE denominator: in-sample naive forecast error (lag-1 difference)
    naive_diff = torch.abs(target_seq[:, 1:] - target_seq[:, :-1])  # (batch, output_len - 1)
    mae_naive = naive_diff.mean(dim=1, keepdim=True)  # (batch, 1)

    # Avoid division by zero
    eps = 1e-8
    mase = mae_pred / (mae_naive + eps)  # (batch, n_samples)

    # Return mean MASE across batch and samples
    return mase.mean()


def crps(pred_seq: torch.Tensor, target_seq: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """
    Compute CRPS approximation via quantile loss.

    Args:
        pred_seq: Tensor of shape (batch, n_samples, output_len)
        target_seq: Tensor of shape (batch, output_len)
        quantiles: List of quantile levels (e.g. [0.1, 0.2, ..., 0.9])

    Returns:
        Scalar tensor: averaged CRPS value over all quantiles, batch, and time steps
    """
    assert pred_seq.ndim == 3 and target_seq.ndim == 2
    q_levels = torch.tensor(quantiles, device=pred_seq.device)  # (Q,)

    # Estimate quantiles from the predictive distribution (dim=1: samples)
    pred_quantiles = torch.quantile(pred_seq, q_levels, dim=1)  # (Q, batch, output_len)

    # Expand target: (batch, output_len) → (Q, batch, output_len)
    target_exp = target_seq.unsqueeze(0).expand(len(q_levels), -1, -1)

    # Pinball loss: (Q, batch, output_len)
    diff = target_exp - pred_quantiles
    loss = torch.maximum(q_levels.view(-1, 1, 1) * diff, (q_levels.view(-1, 1, 1) - 1) * diff)

    # Return mean CRPS over quantiles, batch, and output_len
    return loss.mean()
