import torch

def repeat(tensor: torch.Tensor, n: int, dim: int = 0):
    """
    Repeat each element of a tensor n times along the specified dimension.
    
    Args:
        tensor: Input tensor to repeat elements from
        n: Number of times to repeat each element
        dim: Dimension along which to repeat (default: 0)
        
    Returns:
        torch.Tensor: Tensor with elements repeated n times along dim
    """
    return tensor.repeat_interleave(repeats=n, dim=dim)


def sliding_window_batch(x, L, H):
    """
    Create sliding windows from input tensor for batch processing.
    
    Takes a tensor of shape (B, L+H, C) and creates H sliding windows of length L,
    resulting in a tensor of shape (B, H, L, C) where each window is shifted by 1.
    
    Args:
        x: Input tensor of shape (B, L+H, C) or (B, L+H, C, 1)
        L: Length of each sliding window
        H: Number of sliding windows to create
        
    Returns:
        torch.Tensor: Tensor of shape (B, H, L, C) containing H sliding windows
        
    Raises:
        AssertionError: If total sequence length is insufficient for L+H
    """
    if len(x.shape) == 4:
        x = x.squeeze()
    B, total_len, C = x.shape
    assert total_len >= L + H, "Not enough sequence length for given L and H"

    windows = [x[:, h:h + L, :].unsqueeze(1) for h in range(H)]  # list of (B, 1, L, C)
    return torch.cat(windows, dim=1)  # (B, H, L, C)

