import torch

def mase(pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Scaled Error (MASE) using the median of pred_seq over samples.

    Args:
        pred_seq: Tensor of shape (batch, n_samples, output_len, dim)
        target_seq: Tensor of shape (batch, output_len, dim)

    Returns:
        mase: Scalar Tensor (mean over batch, output_len, and dim)
    """
    # Shape checks
    assert pred_seq.ndim == 4, "pred_seq must be (batch, n_samples, output_len, dim)"
    assert target_seq.ndim == 3, "target_seq must be (batch, output_len, dim)"
    batch, n_samples, output_len, dim = pred_seq.shape
    assert target_seq.shape == (batch, output_len, dim), "target_seq shape mismatch with pred_seq"

    # Take median over n_samples dimension
    pred_median = pred_seq.median(dim=1).values  # (batch, output_len, dim)

    # MAE numerator: prediction error
    mae_pred = torch.abs(pred_median - target_seq)  # (batch, output_len, dim)

    # MAE denominator: in-sample naive forecast error (lag-1 difference)
    naive_diff = torch.abs(target_seq[:, 1:, :] - target_seq[:, :-1, :])  # (batch, output_len - 1, dim)
    mae_naive = naive_diff.mean(dim=1, keepdim=True)  # (batch, 1, dim)
    # Expand mae_naive to match mae_pred shape: (batch, output_len, dim)
    mae_naive_exp = mae_naive.expand(-1, output_len, -1)

    # Avoid division by zero
    eps = 1e-8
    mase = mae_pred / (mae_naive_exp + eps)  # (batch, output_len, dim)

    return mase.mean()


def crps(pred_seq: torch.Tensor, target_seq: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """
    Compute CRPS approximation via quantile loss.

    Args:
        pred_seq: Tensor of shape (batch, n_samples, output_len, dim)
        target_seq: Tensor of shape (batch, output_len, dim)
        quantiles: List of quantile levels (e.g. [0.1, 0.2, ..., 0.9])

    Returns:
        Scalar tensor: averaged CRPS value over all quantiles, batch, output_len, and dim
    """
    assert pred_seq.ndim == 4 and target_seq.ndim == 3
    batch, n_samples, output_len, dim = pred_seq.shape
    assert target_seq.shape == (batch, output_len, dim)

    q_levels = torch.tensor(quantiles, device=pred_seq.device)  # (Q,)

    # Estimate quantiles from the predictive distribution (dim=1: samples)
    # Output: (Q, batch, output_len, dim)
    pred_quantiles = torch.quantile(pred_seq, q_levels, dim=1)

    # Expand target: (batch, output_len, dim) -> (Q, batch, output_len, dim)
    target_exp = target_seq.unsqueeze(0).expand(len(q_levels), -1, -1, -1)

    # Pinball loss: (Q, batch, output_len, dim)
    diff = target_exp - pred_quantiles
    q_levels_view = q_levels.view(-1, 1, 1, 1)
    loss = torch.maximum(q_levels_view * diff, (q_levels_view - 1) * diff)

    # Return mean CRPS over quantiles, batch, output_len, and dim
    return loss.mean()
