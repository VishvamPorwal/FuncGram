# %%
import torch
import pandas as pd
# %%
df_fn = pd.read_csv('data/function_names.csv')
df_fn['function_name'].str.casefold()
# %%
chars = sorted(list(set(''.join(df_fn['function_name'].str.casefold().values))))
chars.insert(0, '.')
chars
# %%
stoi = {ch: i for i, ch in enumerate(chars)}
stoi
# %%
itos = {i: ch for i, ch in enumerate(chars)}
itos
# %%
N = torch.zeros(len(chars), len(chars), dtype=torch.int64)
N.shape
# %%
for fn in df_fn['function_name'].str.casefold().values:
    fn = '.' + fn + '.'
    for ch1, ch2 in zip(fn, fn[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='rainbow')
for i in range(N.shape[0]):
    for j in range(N.shape[1]):
        chstr = itos[i] + itos[j]
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='white')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='white')
plt.axis('off')
plt.savefig('data/pair_matrix.png')
plt.show()
# %%
# save the data
torch.save(N, 'data/N.pt')
torch.save(stoi, 'data/stoi.pt')
torch.save(itos, 'data/itos.pt')
