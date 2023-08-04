# %%
import torch
from nll_loss import neg_log_likel_loss
import pandas as pd

# %%
# load the data
N = torch.load('data/N.pt')
stoi = torch.load('data/stoi.pt')
itos = torch.load('data/itos.pt')

#%%
N += 1 # add 1 to avoid zero probabilities(model smoothing)
# calculate the probabilities
P = N.float()
P = P / P.sum(dim=1, keepdim=True)

# %%
# function to generate a function name
def sample_fn():
    """
    Sample a function name from the distribution

    Returns:
        str: function name

    Usage:
        >>> sample_fn()
        'get'
    """
    ix = 0
    fn = ''
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        ch = itos[ix]
        if ix == 0:
            break
        fn += ch
    return fn

# %%
# evaluate the model
def evaluate():
    """
    Evaluate the model using average negative log likelihood loss
    
    Returns:
        float: average negative log likelihood loss
    
    Usage:
        >>> evaluate()
        2.262559785748267
    """
    df_fn = pd.read_csv('data/function_names.csv')
    loss = 0
    n = 0
    for fn in df_fn['function_name'].str.casefold().values:
        fn = '.' + fn + '.'
        for ch1, ch2 in zip(fn, fn[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            p = P[ix1, ix2]
            n += 1
            loss += neg_log_likel_loss(p)
    return loss.item() / n
