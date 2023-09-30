# Install
```
%%capture
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
!pip install wandb
import wandb as wb
from skimage.io import imread
```
```
def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))
```
```
def plot(x):
    if type(x) == torch.Tensor :
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.show()
```
# Idenify
```
```
Defining Shape
```
X = X.reshape(X.shape[0],784)
X_test = X_test.reshape(X_test.shape[0],784)
```
```
X = X.T
```
```
x = X[:,0:64]
```
```
Y[0:64]
```
```
M = GPU(np.random.rand(10,784))
```
```
y = M@x
```
```
torch.max(y,0)
```
```
y = torch.argmax(y,0)
```
```
Y[0:64]
```
```
y == Y[0:64]
```
```
torch.sum((y == Y[0:64]))
```
```
torch.sum((y == Y[0:64]))/64
```
Making the Model
```
batch_size = 64

M = GPU(np.random.rand(10,784))

y = M@x

y = torch.argmax(y,0)

torch.sum((y == Y[0:batch_size]))/batch_size
```
```
M_Best = 0
Score_Best = 0

for i in range(100000):

    M_new = GPU(np.random.rand(10,784))

    y = M_new@x

    y = torch.argmax(y,0)

    Score = (torch.sum((y == Y[0:batch_size]))/batch_size).item()

    if Score > Score_Best:

        Score_Best = Score
        M_Best = M_new

        print(i,Score_Best)
```
