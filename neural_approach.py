# %%
import pandas as pd
import torch
import torch.nn.functional as F

# %%
# load the data
stoi = torch.load('data/stoi.pt')
itos = torch.load('data/itos.pt')

# prepare the data for training
df_fn = pd.read_csv('data/function_names.csv')
fns = df_fn['function_name'].str.casefold().values
fns
# %%
xs = []
ys = []
for fn in fns:
    fn = '.' + fn + '.'
    for ch1, ch2 in zip(fn, fn[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
n = xs.nelement()
print("Number of pairs:", n)
# %%
# one-hot encode the input
xenc = F.one_hot(xs, len(stoi)).float()
xenc.shape
# %%
# initialize the Weights
W = torch.randn((len(stoi), len(stoi)), requires_grad=True)
W.shape
# %%
# train the model
for epoch in range(20):
    # forward pass
    logits = xenc @ W
    probs = logits.exp()
    probs = probs / probs.sum(dim=1, keepdim=True)

    ypred = probs[torch.arange(n), ys]
    loss = -torch.log(ypred).mean() # trying to be closer to the actual probabilities
    loss += 0.1 * (W ** 2).mean() # Model smoothing(L2 regularization), trying to make W zero by minimizing the accumulated weights
    print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # backward pass
    W.grad = None
    loss.backward()

    # update the weights
    W.data += (-100 * W.grad)
# %%
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
        xenc = F.one_hot(torch.tensor([ix]), len(stoi)).float()
        logits = xenc @ W
        probs = logits.exp()
        p = probs / probs.sum(dim=1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        ch = itos[ix]
        if ix == 0:
            break
        fn += ch
    return fn
