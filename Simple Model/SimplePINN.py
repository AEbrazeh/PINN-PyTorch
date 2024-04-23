import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PINNDataset(Dataset):
    '''
    A custom PyTorch dataset designed for Physics-Informed Neural Networks (PINNs) that generates random data points for both the interior and boundary of a domain.
    This dataset is particularly useful for training PINNs, where the distinction between interior and boundary points is crucial for the model to learn the underlying physical laws represented by differential equations.
    
    Args:
        interiorSize (int): The number of interior data points randomly generated within the domain.
        boundarySize (int): The number of boundary data points randomly generated on the domain's boundary.
    '''
    def __init__(self, interiorSize, boundarySize):
        """
        Initializes the dataset by generating random interior and boundary data points suitable for PINN applications.

        The interior points are sampled uniformly within the domain, while the boundary points are sampled on the edges of the domain, which can be used to apply boundary conditions in PINNs.

        Args:
            interiorSize (int): The number of interior data points to generate.
            boundarySize (int): The number of boundary data points to generate.
        """
        
        # Generate interior points
        x_interior = torch.rand(interiorSize, 2)
        # Generate boundary points on the edges of a unit square domain
        x_boundary1 = torch.cat((torch.randint(0, 2, (boundarySize, 1)), torch.rand(boundarySize, 1)), dim=-1)
        x_boundary2 = torch.cat((torch.rand(boundarySize, 1), torch.randint(0, 2, (boundarySize, 1))), dim=-1)
        self.x = torch.cat((x_interior, x_boundary1, x_boundary2), dim=0)
        
    def __len__(self):
        """
        Returns the total number of data points in the dataset.

        Returns:
            int: The sum of interior and boundary data points.
        """
        return self.x.shape[0]
    
    def __getitem__(self, index):
        """
        Retrieves a data point at the specified index.

        Args:
            index (int): The index of the desired data point.

        Returns:
            torch.Tensor: The data point at the specified index, which can be either an interior or boundary point.
        """
        return self.x[index]

class fourierActivation(nn.Module):
    """
    A custom activation layer that applies a Fourier series transformation to the input features.

    This layer expands the input features into a Fourier series, allowing the neural network to capture periodic and complex patterns in the data. It is especially useful for functions with known periodicity or when modeling wave-like phenomena.

    Args:
        inputDim (int): The dimensionality of the input features.
        expansionOrder (int): The order of the Fourier series expansion.
    """
    def __init__(self, inputDim, expansionOrder):
        """
        Initializes the FourierActivationLayer with the specified input dimension and expansion order.

        Args:
            inputDim (int): The dimensionality of the input features.
            expansionOrder (int): The order of the Fourier series expansion.
        """
        super(fourierActivation, self).__init__()
        self.n = nn.Parameter(torch.arange(expansionOrder)+1, requires_grad=False)
        self.logL = nn.Parameter(torch.randn(1, inputDim, 1), requires_grad=True)
        self.phase = nn.Parameter(np.pi*torch.rand(inputDim, expansionOrder), requires_grad=True)
        self.weight = nn.Parameter(torch.randn(inputDim, expansionOrder) / self.logL.exp(), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(inputDim)/inputDim, requires_grad=True)
    def forward(self, input):
        """
        Applies the Fourier activation function to the input features.
        The activation function is a weighted sum of sine functions with different frequencies and phases.

        Args:
            input (torch.Tensor): The input features to the layer.

        Returns:
            torch.Tensor: The output features after applying the Fourier activation function.
        """
        return (self.weight * torch.sin(2 * self.n * np.pi * input.unsqueeze(-1) / self.logL.exp() + self.phase.unsqueeze(0)) / torch.lgamma(self.n + 1).exp()).sum(-1) + self.bias

class basePIMLP(nn.Module):
    """
    A foundational class for constructing Physics-Informed Multilayer Perceptrons (PIMLPs).
    This class provides a base architecture for PIMLPs, which are neural networks designed to incorporate physical laws (typically in the form of differential equations) into the learning process.
    The architecture includes a sequence of linear layers followed by a Fourier activation layer.

    Args:
        layers (list of int): Defines the number of neurons in each layer of the network.
        expansionOrder (int): The order of the Fourier series expansion for the final activation layer.
        activation (nn.Module): The activation function to be used between linear layers. Defaults to nn.ReLU().
    """
    def __init__(self, layers, expansionOrder, activation = nn.ReLU()):
        """
        Initializes the BasePIMLP with a specified architecture and activation function.
        The network is composed of a sequence of linear layers with the specified activation function applied between them.
        The final layer is a custom Fourier activation layer that applies a Fourier series transformation to the output of the previous layers.

        Args:
            layers (list of int): A list defining the number of neurons in each layer of the network.
            expansionOrder (int): The order of the Fourier series expansion for the final activation layer.
            activation (nn.Module): The activation function to be used between linear layers. Defaults to nn.ReLU().
        """
        super(basePIMLP, self).__init__()
        self.network = nn.Sequential()
        for ii in range(len(layers) - 2):
            self.network.append(nn.Linear(layers[ii], layers[ii+1]))
            self.network.append(activation)
        self.network.append(nn.Linear(layers[-2], layers[-1]))
        self.inputDim, self.outputDim = layers[0], layers[-1]
        self.finalLayer = fourierActivation(layers[-1], expansionOrder)

    def forward(self, x):
        """
        Propagates the input through the network and applies the final Fourier activation layer.
        This method is the forward pass of the network. It takes an input tensor 'x', passes it through the sequential layers of the network, and then applies the final Fourier activation layer to produce the output.

        Args:
            x (torch.Tensor): The input tensor to the network.

        Returns:
            torch.Tensor: The output tensor after passing through the network and the final Fourier activation layer.
        """
        return self.finalLayer(self.network(x))
        
    def jvpDeriv(self, x, n):
        """
        Computes the Jacobian-Vector product (JVP), which is equivalent to the derivative of each output with respect to a specified input dimension.
        This method is useful for calculating derivatives in a computationally efficient manner, especially when dealing with high-dimensional outputs.

        Args:
            x (torch.Tensor): The input tensor to the network.
            n (int): The index of the input dimension with respect to which the derivatives are computed.

        Returns:
            torch.Tensor: The JVP, representing the derivative of each output with respect to the specified input dimension.
        """
        b = x.size(0)
        v = torch.zeros((b, self.inputDim), device=x.device)
        v[:, n] = 1
        return torch.autograd.functional.jvp(self, x, v, True)
    
    def vjpDeriv(self, x, m):
        """
        Computes the Vector-Jacobian product (VJP), which is equivalent to the derivative of a specified output dimension with respect to each input.
        This method is useful for calculating gradients in a computationally efficient manner, especially when dealing with high-dimensional inputs and when only a subset of the gradients are needed.

        Args:
            x (torch.Tensor): The input tensor to the network.
            m (int): The index of the output dimension with respect to which the gradients are computed.

        Returns:
            torch.Tensor: The VJP, representing the derivative of the specified output dimension with respect to each input.
        """
        b = x.size(0)
        v = torch.zeros((b, self.outputDim), device=x.device)
        v[:, m] = 1
        return torch.autograd.functional.vjp(self, x, v, True)
    
    def jacobian(self, x):
        """
        Computes the Jacobian matrix for the network's outputs with respect to its inputs.
        The Jacobian matrix is a representation of all first-order partial derivatives of a vector-valued function.
        In the context of this network, it captures the derivatives of each output with respect to each input feature.

        Args:
            x (torch.Tensor): The input tensor for which the Jacobian matrix is computed.

        Returns:
            torch.Tensor: A tensor representing the Jacobian matrix. The size of the tensor is (batch_size, output_dimension, input_dimension).
        """
        jac = torch.zeros((x.size(0), self.outputDim, self.inputDim), device=x.device)
        if self.outputDim < self.inputDim:
            for ii in range(self.outputDim):
                y = self.vjpDeriv(x, ii)[1]
                jac[:, ii] = y
        else:
            for ii in range(self.inputDim):
                y = self.jvpDeriv(x, ii)[1]
                jac[:, :, ii] = y
        return jac
                    
    def hessian(self, x, laplacian = False):
        """
        Computes the Hessian matrix for the network's outputs with respect to its inputs.
        The Hessian matrix is a square matrix of second-order partial derivatives of a scalar-valued function.
        It describes the local curvature of the function. This method computes the Hessian matrix for each output of the network with respect to each input feature.

        Args:
            x (torch.Tensor): The input tensor for which the Hessian matrix is computed.
            laplacian (bool, optional): If True, computes the Laplacian of the outputs instead of the full Hessian matrix. The Laplacian is the trace of the Hessian matrix and is used in various physical equations. Defaults to False.

        Returns:
            torch.Tensor: A tensor representing the Hessian matrix.
            If 'laplacian' is False, the size of the tensor is (batch_size, output_dimension, input_dimension, input_dimension).
            If 'laplacian' is True, the size is (batch_size, output_dimension).
        """
        b = x.size(0)
        hess = torch.zeros((b, self.outputDim, self.inputDim, self.inputDim), device=x.device)
        for ii in range(self.inputDim):
            v = torch.zeros((b, self.inputDim), device=x.device)
            v[:, ii] = 1
            y = torch.autograd.functional.jvp(self.jacobian, x, v, True)[1]
            hess[:, :, ii] = y
                
        if laplacian:
            eye = torch.eye(self.inputDim, device=x.device).reshape(1, 1, self.inputDim, self.inputDim)
            hess = (hess * eye).sum(-1)
        return hess
        
    def save(self, file):
        torch.save(self, file)