import torch

def get_sequence_from_prob(p: torch.Tensor, is_sample: bool, eps: float = 1e-6):
    """
    p: Tensor of shape (B, D) with probabilities
    Returns:
        best_sequences: Tensor of shape (B, D) with the most probable [1...1, 0...0] sequence
        best_probs: Tensor of shape (B,) with normalized probability of the best sequence
    """
    B, D = p.shape

    # Clamp p to avoid log(0) or log(1) instability
    p_clamped = p.clamp(min=eps, max=1 - eps)

    # Use log domain to compute cumulative products
    log_p = torch.log(p_clamped)
    log_1_minus_p = torch.log(1 - p_clamped)

    log_success = torch.cumsum(log_p, dim=1)  # shape (B, D)
    log_fail = torch.cumsum(log_1_minus_p.flip(dims=[1]), dim=1).flip(dims=[1])  # shape (B, D)

    # Pad with log(1) = 0 to align indexing
    zero = torch.zeros((B, 1), dtype=p.dtype, device=p.device)
    log_success = torch.cat([zero, log_success], dim=1)  # shape (B, D+1)
    log_fail = torch.cat([log_fail, zero], dim=1)  # shape (B, D+1)

    # Sum log-probs for each possible cutoff (index k: first 0 after all 1s)
    log_probs = log_success + log_fail  # shape (B, D+1)
    log_probs_max = torch.max(log_probs, dim=1, keepdim=True)[0]
    probs_normalized = torch.exp(log_probs - log_probs_max)
    probs_normalized = probs_normalized / probs_normalized.sum(dim=1, keepdim=True)

    # Sample or take the most probable index
    if is_sample:
        k = torch.multinomial(probs_normalized, num_samples=1)
    else:
        k = torch.argmax(probs_normalized, dim=1, keepdim=True)

    # Create the monotonic sequence [1,...,1,0,...,0]
    arange = torch.arange(D, device=p.device).unsqueeze(0)
    best_sequences = (arange < k).to(p.dtype)  # shape (B, D)

    best_probs = torch.gather(probs_normalized, dim=1, index=k).squeeze(1)

    return best_sequences, best_probs


def most_probable_monotonic_sequence(p: torch.Tensor):
    """
    Find the most probable monotonic sequence [1,...,1,0,...,0] given probabilities.
    
    For a 1D tensor of probabilities, finds the cutoff point k that maximizes
    the probability of the monotonic sequence where the first k elements are 1
    and the remaining elements are 0.
    
    Args:
        p: 1D tensor of shape (n,) containing probabilities for each position
        
    Returns:
        tuple: (best_sequence, probability)
            - best_sequence: 1D tensor of shape (n,) with the optimal [1,...,1,0,...,0] pattern
            - probability: float, normalized probability of the best sequence
    """
    n = p.shape[0]

    # Cumulative products for probability calculation
    left_cumprod = torch.cumprod(p, dim=0)  # prod(p[0] ... p[k-1])
    right_cumprod = torch.cumprod((1 - p).flip(0), dim=0).flip(0)  # prod(1-p[k] ... 1-p[n-1])

    # Pad with 1 at ends to handle edge cases (k = 0 and k = n)
    left = torch.cat([torch.tensor([1.0], dtype=p.dtype, device=p.device), left_cumprod[:-1]])
    right = right_cumprod

    # Element-wise multiplication to get probability of each sequence
    probs = left * right
    probs_normalized = probs / probs.sum()

    # Find the cutoff point with highest probability
    best_k = torch.argmax(probs_normalized).item()
    best_sequence = torch.cat([
        torch.ones(best_k, dtype=p.dtype, device=p.device),
        torch.zeros(n - best_k, dtype=p.dtype, device=p.device)
    ])

    return best_sequence, probs_normalized[best_k]