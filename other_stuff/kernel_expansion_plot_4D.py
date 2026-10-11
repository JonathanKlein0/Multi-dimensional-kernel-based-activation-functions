import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import os
import sys
sys.path.append(os.path.abspath("generalized_kafs"))
from generalized_kafs import generalized_KAF

act = generalized_KAF(4, 4, D=6, boundary=3.0, grid='sparse', no_boundary_nodes=True)
y = act.dict

gamma = act.gamma * 1

# Grid over x1, x2
grid_size = 150
x1 = torch.linspace(-3, 3, grid_size)
x2 = torch.linspace(-3, 3, grid_size)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')

# Build x' = (x1, x2, 1, 2)
ones = 0.65498165 * torch.ones_like(X1)
twos = 1.56981685 * torch.ones_like(X1)

# Shape: (G, G, 4)
X_prime = torch.stack([X1, X2, ones, twos], dim=-1)

# Compute f(x)
# X_prime: (G, G, 4)
# y: (n, 4)
diff = X_prime.unsqueeze(2) - y.unsqueeze(0).unsqueeze(0)  # (G, G, n, 4)
sq_dist = (diff ** 2).sum(dim=-1)  # (G, G, n)
f = torch.exp(-gamma * sq_dist).sum(dim=2)  # (G, G)

# Convert to numpy
X1_np = X1.numpy()
X2_np = X2.numpy()
f_np = f.numpy()

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X1_np,
    X2_np,
    f_np,
    cmap='viridis',
    edgecolor='none'
)

fig.colorbar(surf, ax=ax, shrink=0.6, label='f(x)')

ax.set_title("3D Surface with 4D y_i and embedded x'")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x)")

plt.show()