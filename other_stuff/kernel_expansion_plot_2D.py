import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import os
import sys
sys.path.append(os.path.abspath("generalized_kafs"))
from generalized_kafs import generalized_KAF

act = generalized_KAF(2, 2, D=6, boundary=3.0, grid='sparse', no_boundary_nodes=True)
y = act.dict

gamma = 2.0

# Grid
grid_size = 150
x1 = torch.linspace(-4, 4, grid_size)
x2 = torch.linspace(-4, 4, grid_size)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
X = torch.stack([X1, X2], dim=-1)

# Compute f(x)
diff = X.unsqueeze(2) - y.unsqueeze(0).unsqueeze(0)
sq_dist = (diff ** 2).sum(dim=-1)
f = torch.exp(-gamma * sq_dist).sum(dim=2)

# Convert to numpy
X1_np = X1.numpy()
X2_np = X2.numpy()
f_np = f.numpy()

# Plot 3D surface
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

ax.set_title("3D Surface of f(x)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x)")

# Optional: show y_i points projected onto surface (z=0 or interpolated)
ax.scatter(y[:, 0].numpy(), y[:, 1].numpy(), torch.zeros(len(y)).numpy(),
           c='red', s=30, label='y_i')
ax.legend()

plt.show()