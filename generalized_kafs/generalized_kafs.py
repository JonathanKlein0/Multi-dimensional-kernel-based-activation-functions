import numpy as np
import qmcpy as qp
from itertools import product, combinations_with_replacement
import torch
from torch.nn import Module
from torch.nn.init import normal_
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

import os
import sys
sys.path.append(os.path.abspath("p_greedy/VKOGA/src/vkoga"))
from pgreedy import PGreedy
from kernels import Gaussian

class generalized_KAF(Module):
    """ Implementation of the generalized kernel activation function for arbitrary dimension of dictionary elements
        and different grid choices.
    
    Parameters
    ----------
    num_parameters: int
        Size of the input layer (number of neurons).
    dim: int
        Dimension of dictionary elements.
    grid_to_copy: generalized_KAF or None, optional.
        If given, copies the grid directly or from the other KAF. Defaults to None.
    D: int, optional
        Determines size of the dictionary for each neuron if grid_to_copy not given.
        For the uniform grid, D positions per dimension are used, thus D^dim grid points in total.
        For the Monte Carlo grid, D random grid points of dimension dim are sampled.
        For the sparse grid, D is the level of the grid.
        For the lattice rules, D total points of dimension dim are generated, D must be a power of 2.
        For P-greedy, a grid of D points is constructed.
    conv: bool, optional
        True if this is a convolutive layer, False for a feedforward layer. Default to False.
    boundary: float, optional
        Dictionary elements are sampled in [-boundary, boundary] if grid_to_copy not given. Defaults to 4.0.
    kernel: {'gaussian', 'relu', 'softplus'}, optional
        Kernel function to be used. Defaults to 'gaussian'.
    grid: {'uniform', 'monte_carlo', 'sparse', 'lattice', 'p_greedy'}, optional
        Grid type to be used if grid_to_copy not given. Defaults to 'uniform'.
    no_boundary_nodes: boolean, optional.
        Removes boundary nodes from the grid when using sparse grids. Defaults to False.
    kernel_bandwidth_constant: float, optional
        Sets the constant for computing the kernel bandwidth, only needed when using Gaussian kernel.
    """

    def __init__(self, num_input_params, dim, grid_to_copy=None, D=None, conv=False, boundary=3.0, kernel='gaussian', grid='uniform', no_boundary_nodes=False, kernel_bandwidth_constant=None, RSigELUD=False):
        
        # Check that the number of parameters fits with the dimension
        if np.mod(num_input_params, dim) != 0:
            raise ValueError('The number of parameters must be a multiple of the chosen dimension.')

        # Select appropriate kernel function
        if not (kernel in ['gaussian', 'relu', 'softplus']):
            raise ValueError('Kernel not recognized (must be {gaussian, relu, softplus})')
        
        # Select appropriate grid
        if not (grid in ['uniform', 'monte_carlo', 'sparse', 'lattice', 'p_greedy']):
            raise ValueError('Grid not recognized (must be {uniform, monte_carlo, sparse, lattice, p_greedy})')

        super().__init__()
        self.num_input_params, self.num_output_params, self.dim, self.conv, self.grid, self.boundary, self.kernel_bandwidth_constant, self.no_boundary_nodes, self.RSigELUD = num_input_params, num_input_params // dim, dim, conv, grid, boundary, kernel_bandwidth_constant, no_boundary_nodes, RSigELUD
        
        # Initialize the dictionary
        if grid_to_copy is not None:
            if grid_to_copy.dict.shape[-1] != dim:
                raise ValueError('Grid points must have dimension dim.')
            self.register_buffer('dict', grid_to_copy.dict)
            self.grid = grid_to_copy.grid
            self.boundary = grid_to_copy.boundary
            self.no_boundary_nodes = grid_to_copy.no_boundary_nodes
        
        elif D == None:
            raise ValueError('Must give input for grid_to_copy or D.')
        
        elif grid == 'uniform':
            axis = np.linspace(-boundary, boundary, D, dtype=np.float32)
            mesh = np.array(list(product(*([axis] * dim))), dtype=np.float32)
            self.register_buffer('dict', torch.from_numpy(mesh))
        
        elif grid == 'monte_carlo':
            points = np.random.uniform(low=-boundary, high=boundary, size=(D, dim)).astype(np.float32)
            self.register_buffer('dict', torch.from_numpy(points))
        
        elif grid == 'sparse':
            self.register_buffer('dict', torch.from_numpy(build_sparse_grid(dim, D, boundary, no_boundary_nodes).astype(np.float32)))
        
        elif grid == 'lattice':
            sampler = qp.Lattice(dimension=dim, randomize=True)
            points = sampler.gen_samples(D).astype(np.float32)
            mapped_points = 2 * boundary * points - boundary
            self.register_buffer('dict', torch.from_numpy(mapped_points))

        elif grid == 'p_greedy':
            if kernel != 'gaussian':
                raise ValueError('Kernel must be Gaussian for P-greedy')
            self.dict_size = D
            centers = pgreedy_select_points(10000, D, self.dim, boundary=self.boundary, kernel_par=self.initialize_gamma()).astype(np.float32)
            self.register_buffer('dict', torch.from_numpy(centers))
        
        self.dict_size = self.dict.shape[-2]

        # Special cases (for faster computations): dim = 2 or dim = 4, kernel = 'gaussian'
        self.special_case_2D = (self.dim == 2) & (kernel == 'gaussian')
        self.special_case_4D = (self.dim == 4) & (kernel == 'gaussian')

        if self.special_case_2D:
            
            # Initialize the fixed dictionary
            self.register_buffer('Dx', self.dict[:,0].reshape(1, 1, -1))
            self.register_buffer('Dy', self.dict[:,1].reshape(1, 1, -1))

            # Initialize gamma
            self.gamma = self.initialize_gamma()

            # Mixing coefficients
            self.alpha = Parameter(torch.FloatTensor(1, self.num_output_params, self.dict_size))

            # Kernel function
            self.kernel_fcn = self.gauss_2d_kernel

        elif self.special_case_4D:

            # Initialize the fixed dictionary
            self.register_buffer('D0', self.dict[:,0].reshape(1, 1, -1))
            self.register_buffer('D1', self.dict[:,1].reshape(1, 1, -1))
            self.register_buffer('D2', self.dict[:,2].reshape(1, 1, -1))
            self.register_buffer('D3', self.dict[:,3].reshape(1, 1, -1))

            # Initialize gamma
            self.gamma = self.initialize_gamma()

            # Mixing coefficients
            self.alpha = Parameter(torch.FloatTensor(1, self.num_output_params, self.dict_size))

            # Kernel function
            self.kernel_fcn = self.gauss_4d_kernel

        # General case
        else:
            # Initialize the kernel
            if conv:
                if kernel == 'gaussian':
                    self.kernel_fcn = self.gaussian_kernel_conv
                elif kernel == 'relu':
                    self.kernel_fcn = self.relu_kernel_conv
                else:
                    self.kernel_fcn = self.softplus_kernel_conv
            else:
                if kernel == 'gaussian':
                    self.kernel_fcn = self.gaussian_kernel
                elif kernel == 'relu':
                    self.kernel_fcn = self.relu_kernel
                else:
                    self.kernel_fcn = self.softplus_kernel

            # Initialize gamma for gaussian kernel
            if kernel == 'gaussian':
                self.gamma = self.initialize_gamma()

            # Mixing coefficients
            if conv:
                self.alpha = Parameter(torch.FloatTensor(1, self.num_output_params, 1, 1, self.dict.shape[0]))
                if grid_to_copy is None:
                    self.dict.data = self.dict.data.view(1, 1, 1, 1, self.dict.shape[0], self.dict.shape[1])
            else:
                self.alpha = Parameter(torch.FloatTensor(self.num_output_params, self.dict.shape[0]))

        # Initialize the parameters
        self.reset_parameters()

    def initialize_gamma(self):
        if self.kernel_bandwidth_constant is None:
            if self.dim == 2 and self.grid == 'sparse' and not self.no_boundary_nodes:
                self.kernel_bandwidth_constant = 1/16
            elif (self.dim == 2 and self.grid == 'p_greedy') or (self.dim == 4 and self.grid in ['uniform', 'monte_carlo']):
                self.kernel_bandwidth_constant = 1/8
            elif (self.dim == 2 and self.grid in ['uniform', 'lattice']) or (self.dim == 4 and self.grid in ['lattice', 'p_greedy']):
                self.kernel_bandwidth_constant = 1/4
            elif self.dim in [2, 4] and self.grid == 'sparse' and self.no_boundary_nodes:
                self.kernel_bandwidth_constant = 1/2
            elif self.dim == 2 and self.grid == 'monte_carlo':
                self.kernel_bandwidth_constant = 1.0
            else:
                raise ValueError('kernel_bandwidth_constant can not be None for this choice of grid and dimension')
        return self.kernel_bandwidth_constant * self.dict_size**(2/self.dim) / (self.boundary**2)

    def reset_parameters(self):
        normal_(self.alpha.data, std=np.sqrt(0.3))

    def forward(self, input):
        # Check for special cases
        if self.special_case_2D:
            s = input.size()
            # Compute the 2D Gaussian kernel
            K = self.gauss_2d_kernel(input, s)
            if len(s) == 4:
                y = torch.sum(K * self.alpha.view(1, -1, 1, 1, self.dict_size), 4)
            else:
                y = torch.sum(K * self.alpha, 2)
            if not self.RSigELUD:
                return y
            elif not self.conv:
                return y + RSigELUD(input.view(input.shape[0], self.num_output_params, self.dim)).sum(dim=2)
            else:
                return y + RSigELUD(input.view(input.shape[0], self.num_output_params, self.dim, input.shape[2], input.shape[3])).sum(dim=2)
        
        elif self.special_case_4D:
            s = input.size()
            # Compute the 2D Gaussian kernel
            K = self.gauss_4d_kernel(input, s)
            if len(s) == 4:
                y = torch.sum(K * self.alpha.view(1, -1, 1, 1, self.dict_size), 4)
            else:
                y = torch.sum(K * self.alpha, 2)
            if not self.RSigELUD:
                return y
            elif not self.conv:
                return y + RSigELUD(input.view(input.shape[0], self.num_output_params, self.dim)).sum(dim=2)
            else:
                return y + RSigELUD(input.view(input.shape[0], self.num_output_params, self.dim, input.shape[2], input.shape[3])).sum(dim=2)

        # General case
        else:
            K = self.kernel_fcn(input)
            if not self.RSigELUD:
                return (K * self.alpha).sum(dim=-1)
            elif not self.conv:
                return (K * self.alpha).sum(dim=-1) + RSigELUD(input.view(input.shape[0], self.num_output_params, self.dim)).sum(dim=2)
            else:
                return (K * self.alpha).sum(dim=-1) + RSigELUD(input.view(input.shape[0], self.num_output_params, self.dim, input.shape[2], input.shape[3])).sum(dim=2)

    # Functions for kernels depending on kernel choice and conv

    # Special case: 2D Gaussian kernel
    def gauss_2d_kernel(self, X, s):
        if len(s) == 4: # Convolutional
            tmp = -torch.mul((torch.add(X[:, 0::2].unsqueeze(4), - (self.Dx).view(1, 1, 1, 1, -1))) ** 2, self.gamma) - \
                  torch.mul((torch.add(X[:, 1::2].unsqueeze(4), - (self.Dy).view(1, 1, 1, 1, -1))) ** 2, self.gamma)
        else:
            tmp = -torch.mul((torch.add(X[:, 0::2].unsqueeze(2), - (self.Dx)))**2, self.gamma) - \
                  torch.mul((torch.add(X[:, 1::2].unsqueeze(2), - (self.Dy))) ** 2, self.gamma)
        return torch.exp(tmp)
    
    # Special case: 4D Gaussian kernel
    def gauss_4d_kernel(self, X, s):
        if len(s) == 4: # Convolutional
            tmp = -torch.mul((torch.add(X[:, 0::4].unsqueeze(4), - (self.D0).view(1, 1, 1, 1, -1))) ** 2, self.gamma) - \
                  torch.mul((torch.add(X[:, 1::4].unsqueeze(4), - (self.D1).view(1, 1, 1, 1, -1))) ** 2, self.gamma) - \
                  torch.mul((torch.add(X[:, 2::4].unsqueeze(4), - (self.D2).view(1, 1, 1, 1, -1))) ** 2, self.gamma) - \
                  torch.mul((torch.add(X[:, 3::4].unsqueeze(4), - (self.D3).view(1, 1, 1, 1, -1))) ** 2, self.gamma)
        else:
            tmp = -torch.mul((torch.add(X[:, 0::4].unsqueeze(2), - (self.D0))) ** 2, self.gamma) - \
                  torch.mul((torch.add(X[:, 1::4].unsqueeze(2), - (self.D1))) ** 2, self.gamma) - \
                  torch.mul((torch.add(X[:, 2::4].unsqueeze(2), - (self.D2))) ** 2, self.gamma) - \
                  torch.mul((torch.add(X[:, 3::4].unsqueeze(2), - (self.D3))) ** 2, self.gamma)
        return torch.exp(tmp)

    def gaussian_kernel(self, input):
        grouped_input = input.view(input.shape[0], self.num_output_params, self.dim)
        diff = grouped_input.unsqueeze(2) - self.dict
        dist_sq = (diff ** 2).sum(dim=-1)
        return torch.exp(- self.gamma * dist_sq)
    
    def gaussian_kernel_conv(self, input):
        B, C, H, W = input.shape
        grouped_input = input.view(B, self.num_output_params, self.dim, H, W).permute(0, 1, 3, 4, 2)
        diff = grouped_input.unsqueeze(-2) - self.dict
        dist_sq = (diff ** 2).sum(dim=-1)
        return torch.exp(-self.gamma * dist_sq)
    
    def relu_kernel(self, input):
        grouped_input = input.view(input.shape[0], self.num_output_params, self.dim)
        return F.relu(torch.matmul(grouped_input, self.dict.t()))
    
    def relu_kernel_conv(self, input):
        B, C, H, W = input.shape
        grouped_input = input.view(B, self.num_output_params, self.dim, H, W).permute(0, 1, 3, 4, 2)
        dot_prod = (grouped_input.unsqueeze(-2) * self.dict).sum(-1)
        return F.relu(dot_prod)
    
    def softplus_kernel(self, input):
        grouped_input = input.view(input.shape[0], self.num_output_params, self.dim)
        return F.softplus(torch.matmul(grouped_input, self.dict.t()))
    
    def softplus_kernel_conv(self, input):
        B, C, H, W = input.shape
        grouped_input = input.view(B, self.num_output_params, self.dim, H, W).permute(0, 1, 3, 4, 2)
        dot_prod = (grouped_input.unsqueeze(-2) * self.dict).sum(-1)
        return F.softplus(dot_prod)

    def __repr__(self):
        if self.conv:
            return self.__class__.__name__ + ' (input neurons: ' + str(self.num_input_params) + ', output neurons: ' \
            + str(self.num_output_params) + ', grid: ' + self.grid + ', dictionary size: ' + str(self.dict_size) + ')'
        else:
            return self.__class__.__name__ + ' (input neurons: ' + str(self.num_input_params) + ', output neurons: ' \
            + str(self.num_output_params) + ', grid: ' + self.grid + ', dictionary size: ' + str(self.dict_size) + ')'

# functions for sparse grid computation
def multiindices_exact_sum(dim, n):
    for comb in combinations_with_replacement(range(dim), n):
        counts = [comb.count(i) for i in range(dim)]
        yield tuple(counts)

def multiindices_upto_sum(dim, n):
    for total in range(n + 1):
        yield from multiindices_exact_sum(dim, total)

def build_sparse_grid(dim, level, boundary, no_boundary_nodes):
    multiindices = list(multiindices_upto_sum(dim, level))
    all_points = set()
    for mi in multiindices:
        axes = [np.linspace(-boundary, boundary, 2**l + 1) for l in mi]
        for pt in product(*axes):
            all_points.add(tuple(pt))
    grid_points = np.array(sorted(all_points))
    if no_boundary_nodes == True:
        mask = np.all((grid_points > -boundary) & (grid_points < boundary), axis=1)
        return grid_points[mask]
    else:
        return grid_points

# functionality for computing grid using p-greedy algorithm
class FixedDictionary:
    def __init__(self, X):
        self.X = X
        self.d = X.shape[1]
    def sample(self):
        return self.X
    
def pgreedy_select_points(N, n, dim, boundary=3.0, kernel=Gaussian(), kernel_par=2.0):
    X = np.random.uniform(-boundary, boundary, size=(N, dim))
    dictionary = FixedDictionary(X)
    model = PGreedy(kernel=kernel, kernel_par=kernel_par, max_iter=n, tol_p=0.0, verbose=False)
    model.fit(dictionary, reorth=0)
    return model.ctrs_

# RSigELUD function
def RSigELUD(x):
    return torch.where(x < 0, 0.2 * (torch.exp(x) - 1), torch.where(x <= 1, x, x + 0.3 * x / (1 + torch.exp(-x))))
