import os
import sys
sys.path.append(os.path.abspath("generalized_kafs"))
from generalized_kafs import generalized_KAF

import numpy as np
import torch
import matplotlib.pyplot as plt

greedy_KAF = generalized_KAF(2, 2, D=32, grid='p_greedy')

grid = greedy_KAF.dict.numpy()


plt.figure(figsize=(6, 6))
plt.scatter(grid[:, 0], grid[:, 1])
plt.title('P-greedy grid with 32 points from lattice rule grid with 256 points')
plt.savefig('grid_pngs/greedy_8.png')
