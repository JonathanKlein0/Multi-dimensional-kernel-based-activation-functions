import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = torch.load('other_stuff/alpha_before_and_after_RSigELUD.pt')
old_alpha = data['old_alpha']
new_alpha = data['new_alpha']
gamma = data['gamma']
dict = data['dict']

# Grid
grid_size = 150
x1 = torch.linspace(-4, 4, grid_size)
x2 = torch.linspace(-4, 4, grid_size)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
X = torch.stack([X1, X2], dim=-1)

def RSigELUD(x):
    return torch.where(x < 0, 0.2 * (torch.exp(x) - 1), torch.where(x <= 1, x, x + 0.3 * x / (1 + torch.exp(-x))))

# Compute f(x)
diff = X.unsqueeze(2) - dict.unsqueeze(0).unsqueeze(0)
sq_dist = (diff ** 2).sum(dim=-1)
f_old = (torch.exp(-gamma * sq_dist) * old_alpha[0][0].view(1, 1, -1)).sum(dim=2)
f_new = (torch.exp(-gamma * sq_dist) * new_alpha[0][0].view(1, 1, -1)).sum(dim=2)
f_old += RSigELUD(X).sum(dim=2)
f_new += RSigELUD(X).sum(dim=2)

# Convert to numpy
X1_np = X1.numpy()
X2_np = X2.numpy()
f_np = f_old.numpy()

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
ax.scatter(dict[:, 0].numpy(), dict[:, 1].numpy(), torch.zeros(len(dict)).numpy(),
           c='red', s=30, label='y_i')
ax.legend()

plt.savefig("other_stuff/old_alpha_plot_RSigELUD.png", dpi=300, bbox_inches='tight')
plt.close()

f_np = f_new.numpy()

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
ax.scatter(dict[:, 0].numpy(), dict[:, 1].numpy(), torch.zeros(len(dict)).numpy(),
           c='red', s=30, label='y_i')
ax.legend()

plt.savefig("other_stuff/new_alpha_plot_RSigELUD.png", dpi=300, bbox_inches='tight')
plt.close()