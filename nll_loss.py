import torch

def neg_log_likel_loss(probability):
    """
    Negative log likelihood loss: -log(p)
    Maximizing the likelihood(which is the product of all the likelihoods) means better predictions
    Log of a product is the sum of the logs of the individual numbers
    So, maximizing the log likelihood is equivalent to minimizing the negative log likelihood

    Args:
        probability (torch.Tensor): probability of the target character
    
    Returns:
        torch.Tensor: negative log likelihood loss
    """
    log = torch.log(probability)
    return -log