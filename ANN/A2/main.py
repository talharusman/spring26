import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Set up double precision (float64) to keep everything stable during training
# and to match what we're doing in NumPy
torch.set_default_dtype(torch.float64)


def softplus_np(x: np.ndarray) -> np.ndarray:
    """Softplus activation: smooth approximation of ReLU (log(1 + exp(x)))."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Standard sigmoid function: squashes values to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def latin_hypercube(n_samples: int, n_dims: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a Latin hypercube sample for diverse, well-spaced data points.
    This doesn't require any special packages—just NumPy!
    """
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.random((n_samples, n_dims))
    a = cut[:-1].reshape(-1, 1)
    b = cut[1:].reshape(-1, 1)
    rdpoints = a + (b - a) * u

    H = np.zeros_like(rdpoints)
    for j in range(n_dims):
        order = rng.permutation(n_samples)
        H[:, j] = rdpoints[order, j]

    return low + H * (high - low)


def toy_function_f(X: np.ndarray) -> np.ndarray:
    """First test function: exponential + log-sum-exp + tanh + sine."""
    x = X[:, 0]
    y = X[:, 1]
    t = X[:, 2]
    z = X[:, 3]
    return np.exp(-0.5 * x) + np.log1p(np.exp(0.4 * y)) + np.tanh(t) + np.sin(z) - 0.4


def toy_function_g(X: np.ndarray) -> np.ndarray:
    """Second test function: product of exponential, quadratic, tanh, and sine terms."""
    x = X[:, 0]
    y = X[:, 1]
    t = X[:, 2]
    z = X[:, 3]
    fx = np.exp(-0.3 * x)
    fy = (0.15 * y) ** 2
    ft = np.tanh(0.3 * t)
    fz = 0.2 * np.sin(0.5 * z + 2.0) + 0.5
    return fx * fy * fz * ft


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute mean squared error loss."""
    return float(np.mean((pred - target) ** 2))


def save_dataset_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """Save feature matrix X and targets y to a CSV file."""
    arr = np.hstack([X, y])
    header = "x,y,t,z,target"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


@dataclass
class NumpyParameter:
    """Represents a trainable parameter with optional positivity constraint.
    
    Stores both the raw (unconstrained) values and momentum/velocity for Adam.
    Includes transformations to enforce constraints like positivity.
    """
    raw: np.ndarray
    positive: bool = False

    def __post_init__(self) -> None:
        # Initialize Adam momentum and velocity for adaptive learning
        self.momentum = np.zeros_like(self.raw)
        self.velocity = np.zeros_like(self.raw)

    def effective(self) -> np.ndarray:
        """Return the actual parameter values (after applying any constraints)."""
        if self.positive:
            # Apply softplus to keep values positive
            return softplus_np(self.raw)
        return self.raw

    def to_raw_grad(self, grad_effective: np.ndarray) -> np.ndarray:
        """Convert gradients w.r.t. effective values to gradients w.r.t. raw values.
        Handles chain rule for constraint transformations.
        """
        if self.positive:
            # Chain rule: d/dx_raw = d/dx_eff * dx_eff/dx_raw
            # where dx_eff/dx_raw = sigmoid(raw) for softplus
            return grad_effective * sigmoid_np(self.raw)
        return grad_effective

    def adam_step(
        self,
        grad_effective: np.ndarray,
        step: int,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Perform one Adam optimization step.
        
        Updates raw parameter values using exponential moving averages of
        gradients (momentum) and squared gradients (velocity) with bias correction.
        """
        grad_raw = self.to_raw_grad(grad_effective)
        self.momentum = beta1 * self.momentum + (1.0 - beta1) * grad_raw
        self.velocity = beta2 * self.velocity + (1.0 - beta2) * (grad_raw ** 2)
        # Apply bias correction for early iterations
        momentum_corrected = self.momentum / (1.0 - beta1 ** step)
        velocity_corrected = self.velocity / (1.0 - beta2 ** step)
        self.raw = self.raw - lr * momentum_corrected / (np.sqrt(velocity_corrected) + epsilon)


class NumpyModelBase:
    """Base class for all NumPy-based neural network models.
    
    Manages parameters, their optimization, and caching for backprop.
    Subclasses implement specific architectures.
    """
        self.rng = rng
        self.params = {}
        self.step_count = 0
        self.cache = {}

    def _weight_init(self, out_dim: int, in_dim: int, positive: bool) -> np.ndarray:
        """Initialize weights using appropriate strategies.
        
        For positive weights: initialize low so softplus gives small positive values.
        For unconstrained weights: use Xavier/Glorot initialization.
        """
        if positive:
            # Initialize negative values so softplus gives us small positive values
            return self.rng.normal(loc=-2.0, scale=0.1, size=(out_dim, in_dim))
        # Xavier initialization
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        return self.rng.uniform(-limit, limit, size=(out_dim, in_dim))

    def add_param(self, name: str, shape: tuple[int, ...], positive: bool = False, is_bias: bool = False) -> None:
        """Add a trainable parameter to the model.
        
        Biases start at zero, weights use appropriate initialization.
        """
        if is_bias:
            raw = np.zeros(shape)
        else:
            if len(shape) != 2:
                raise ValueError(f"Expected 2D weight shape for {name}, got {shape}")
            raw = self._weight_init(shape[0], shape[1], positive)
        self.params[name] = NumpyParameter(raw=raw, positive=positive)

    def W(self, name: str) -> np.ndarray:
        """Get the effective value of a parameter (after applying constraints)."""
        return self.params[name].effective()

    def step(self, grads_effective: dict[str, np.ndarray], lr: float) -> None:
        """Perform one optimization step using Adam."""
        self.step_count += 1
        for name, grad in grads_effective.items():
            self.params[name].adam_step(grad, self.step_count, lr)


class ISNN1Numpy(NumpyModelBase):
    """
    A numpy-based version of ISNN-1 model.
    
    We implement the matrix operations manually and compute gradients by hand
    to show exactly what's happening during backprop.
    
    Each branch (y, z, t, x) has its own activation function:
    - y branch: softplus (to keep it convex and monotone)
    - z branch: sigmoid (just for diversity in these toy problems)
    - t branch: sigmoid (same)
    - x branch: softplus (convex + monotone like y)
    """

    def __init__(self, rng: np.random.Generator, width: int = 10, depth: int = 2) -> None:
        super().__init__(rng)
        self.width = width
        self.depth = depth

        # y branch: keep weights positive to maintain convex + monotone property
        for i in range(depth):
            in_dim = 1 if i == 0 else width
            self.add_param(f"W_yy_{i}", (width, in_dim), positive=True)
            self.add_param(f"b_y_{i}", (1, width), is_bias=True)

        # z branch: no constraints on weights (arbitrary nonlinearity)
        for i in range(depth):
            in_dim = 1 if i == 0 else width
            self.add_param(f"W_zz_{i}", (width, in_dim), positive=False)
            self.add_param(f"b_z_{i}", (1, width), is_bias=True)

        # t branch: positive weights for monotone property
        for i in range(depth):
            in_dim = 1 if i == 0 else width
            self.add_param(f"W_tt_{i}", (width, in_dim), positive=True)
            self.add_param(f"b_t_{i}", (1, width), is_bias=True)

        # x branch (combines input and branch outputs)
        self.add_param("W_xx0", (width, 1), positive=False)
        self.add_param("W_xy", (width, width), positive=True)
        self.add_param("W_xz", (width, width), positive=False)
        self.add_param("W_xt", (width, width), positive=True)
        self.add_param("b_x0", (1, width), is_bias=True)

        # Hidden x layers: use positive weights to maintain convexity
        for i in range(1, depth):
            self.add_param(f"W_xx_{i}", (width, width), positive=True)
            self.add_param(f"b_x_{i}", (1, width), is_bias=True)

        # Output layer: positive weights ensure a valid convex combination
        self.add_param("W_out", (1, width), positive=True)
        self.add_param("b_out", (1, 1), is_bias=True)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        x0 = X[:, 0:1]
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        y_inputs = [y0]
        y_pre = []
        y_act = []
        y = y0
        for i in range(self.depth):
            p = y @ self.W(f"W_yy_{i}").T + self.W(f"b_y_{i}")
            y = softplus_np(p)
            y_pre.append(p)
            y_act.append(y)
            y_inputs.append(y)

        z_inputs = [z0]
        z_pre = []
        z_act = []
        z = z0
        for i in range(self.depth):
            p = z @ self.W(f"W_zz_{i}").T + self.W(f"b_z_{i}")
            z = sigmoid_np(p)
            z_pre.append(p)
            z_act.append(z)
            z_inputs.append(z)

        t_inputs = [t0]
        t_pre = []
        t_act = []
        t = t0
        for i in range(self.depth):
            p = t @ self.W(f"W_tt_{i}").T + self.W(f"b_t_{i}")
            t = sigmoid_np(p)
            t_pre.append(p)
            t_act.append(t)
            t_inputs.append(t)

        x_pre = []
        x_act = []
        x_inputs = []

        p0 = (
            x0 @ self.W("W_xx0").T
            + y @ self.W("W_xy").T
            + z @ self.W("W_xz").T
            + t @ self.W("W_xt").T
            + self.W("b_x0")
        )
        x = softplus_np(p0)
        x_pre.append(p0)
        x_act.append(x)
        x_inputs.append(x0)

        for i in range(1, self.depth):
            p = x @ self.W(f"W_xx_{i}").T + self.W(f"b_x_{i}")
            x_inputs.append(x)
            x = softplus_np(p)
            x_pre.append(p)
            x_act.append(x)

        out = x @ self.W("W_out").T + self.W("b_out")

        if training:
            self.cache = {
                "x0": x0,
                "y0": y0,
                "z0": z0,
                "t0": t0,
                "y_pre": y_pre,
                "y_act": y_act,
                "y_inputs": y_inputs,
                "z_pre": z_pre,
                "z_act": z_act,
                "z_inputs": z_inputs,
                "t_pre": t_pre,
                "t_act": t_act,
                "t_inputs": t_inputs,
                "x_pre": x_pre,
                "x_act": x_act,
                "x_inputs": x_inputs,
            }
        return out

    def backward(self, d_out: np.ndarray) -> dict[str, np.ndarray]:
        """Backpropagation through the network.
        
        Starting from the output gradient, traces back through each branch
        to compute gradients for all parameters.
        Returns a dictionary mapping parameter names to their gradients.
        """
        grads = {}
        c = self.cache

        # Start from the output layer and work backwards
        x_last = c["x_act"][-1]
        grads["W_out"] = d_out.T @ x_last
        grads["b_out"] = np.sum(d_out, axis=0, keepdims=True)
        dx = d_out @ self.W("W_out")

        # Initialize gradients for branches (will be filled during x backprop)
        dy = None
        dz = None
        dt = None

        # Backprop through x layers
        for i in reversed(range(self.depth)):
            pre = c["x_pre"][i]
            # Gradient of softplus = sigmoid
            dpre = dx * sigmoid_np(pre)

            if i == 0:
                grads["W_xx0"] = dpre.T @ c["x0"]
                grads["W_xy"] = dpre.T @ c["y_act"][-1]
                grads["W_xz"] = dpre.T @ c["z_act"][-1]
                grads["W_xt"] = dpre.T @ c["t_act"][-1]
                grads["b_x0"] = np.sum(dpre, axis=0, keepdims=True)

                dy = dpre @ self.W("W_xy")
                dz = dpre @ self.W("W_xz")
                dt = dpre @ self.W("W_xt")
            else:
                x_in = c["x_act"][i - 1]
                grads[f"W_xx_{i}"] = dpre.T @ x_in
                grads[f"b_x_{i}"] = np.sum(dpre, axis=0, keepdims=True)

            if i > 0:
                dx = dpre @ self.W(f"W_xx_{i}")

        for i in reversed(range(self.depth)):
            pre = c["y_pre"][i]
            dpre = dy * sigmoid_np(pre)
            y_in = c["y0"] if i == 0 else c["y_act"][i - 1]
            grads[f"W_yy_{i}"] = dpre.T @ y_in
            grads[f"b_y_{i}"] = np.sum(dpre, axis=0, keepdims=True)
            if i > 0:
                dy = dpre @ self.W(f"W_yy_{i}")

        for i in reversed(range(self.depth)):
            pre = c["z_pre"][i]
            sig = sigmoid_np(pre)
            dpre = dz * sig * (1.0 - sig)
            z_in = c["z0"] if i == 0 else c["z_act"][i - 1]
            grads[f"W_zz_{i}"] = dpre.T @ z_in
            grads[f"b_z_{i}"] = np.sum(dpre, axis=0, keepdims=True)
            if i > 0:
                dz = dpre @ self.W(f"W_zz_{i}")

        for i in reversed(range(self.depth)):
            pre = c["t_pre"][i]
            sig = sigmoid_np(pre)
            dpre = dt * sig * (1.0 - sig)
            t_in = c["t0"] if i == 0 else c["t_act"][i - 1]
            grads[f"W_tt_{i}"] = dpre.T @ t_in
            grads[f"b_t_{i}"] = np.sum(dpre, axis=0, keepdims=True)
            if i > 0:
                dt = dpre @ self.W(f"W_tt_{i}")

        return grads


class ISNN2Numpy(NumpyModelBase):
    """
    ISNN-2 implementation—a simpler variant with just one layer for each branch
    and two layers in the x pathway. This is what the paper experiments used.
    """

    def __init__(self, rng: np.random.Generator, width: int = 15) -> None:
        super().__init__(rng)
        self.width = width

        # First layer for each branch (single depth)
        self.add_param("W_yy0", (width, 1), positive=True)
        self.add_param("b_y0", (1, width), is_bias=True)

        self.add_param("W_zz0", (width, 1), positive=False)
        self.add_param("b_z0", (1, width), is_bias=True)

        self.add_param("W_tt0", (width, 1), positive=True)
        self.add_param("b_t0", (1, width), is_bias=True)

        # First x layer: combines raw input with branch outputs
        self.add_param("W_xx0", (width, 1), positive=False)
        self.add_param("W_xy0", (width, 1), positive=True)
        self.add_param("W_xz0", (width, 1), positive=False)
        self.add_param("W_xt0", (width, 1), positive=True)
        self.add_param("b_x0", (1, width), is_bias=True)

        # Second x layer: uses branch outputs from above plus a skip connection
        self.add_param("W_xx1", (width, width), positive=True)
        self.add_param("W_xx0_skip", (width, 1), positive=False)
        self.add_param("W_xy1", (width, width), positive=True)
        self.add_param("W_xz1", (width, width), positive=False)
        self.add_param("W_xt1", (width, width), positive=True)
        self.add_param("b_x1", (1, width), is_bias=True)

        self.add_param("W_out", (1, width), positive=True)
        self.add_param("b_out", (1, 1), is_bias=True)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: compute outputs by processing inputs through all layers."""
        x0 = X[:, 0:1]
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        # Process each branch one layer at a time
        y_pre = y0 @ self.W("W_yy0").T + self.W("b_y0")
        y1 = softplus_np(y_pre)

        z_pre = z0 @ self.W("W_zz0").T + self.W("b_z0")
        z1 = sigmoid_np(z_pre)

        t_pre = t0 @ self.W("W_tt0").T + self.W("b_t0")
        t1 = sigmoid_np(t_pre)

        # First x layer: mix input with branch outputs
        x1_pre = (
            x0 @ self.W("W_xx0").T
            + y0 @ self.W("W_xy0").T
            + z0 @ self.W("W_xz0").T
            + t0 @ self.W("W_xt0").T
            + self.W("b_x0")
        )
        x1 = softplus_np(x1_pre)

        # Second x layer: use previous layer output plus skip connection from x0
        x2_pre = (
            x1 @ self.W("W_xx1").T
            + x0 @ self.W("W_xx0_skip").T
            + y1 @ self.W("W_xy1").T
            + z1 @ self.W("W_xz1").T
            + t1 @ self.W("W_xt1").T
            + self.W("b_x1")
        )
        x2 = softplus_np(x2_pre)

        # Final output
        out = x2 @ self.W("W_out").T + self.W("b_out")

        if training:
            self.cache = {
                "x0": x0,
                "y0": y0,
                "z0": z0,
                "t0": t0,
                "y_pre": y_pre,
                "y1": y1,
                "z_pre": z_pre,
                "z1": z1,
                "t_pre": t_pre,
                "t1": t1,
                "x1_pre": x1_pre,
                "x1": x1,
                "x2_pre": x2_pre,
                "x2": x2,
            }
        return out

    def backward(self, d_out: np.ndarray) -> dict[str, np.ndarray]:
        grads = {}
        c = self.cache

        grads["W_out"] = d_out.T @ c["x2"]
        grads["b_out"] = np.sum(d_out, axis=0, keepdims=True)
        dx2 = d_out @ self.W("W_out")

        dpre2 = dx2 * sigmoid_np(c["x2_pre"])
        grads["W_xx1"] = dpre2.T @ c["x1"]
        grads["W_xx0_skip"] = dpre2.T @ c["x0"]
        grads["W_xy1"] = dpre2.T @ c["y1"]
        grads["W_xz1"] = dpre2.T @ c["z1"]
        grads["W_xt1"] = dpre2.T @ c["t1"]
        grads["b_x1"] = np.sum(dpre2, axis=0, keepdims=True)

        dx1 = dpre2 @ self.W("W_xx1")
        dy1 = dpre2 @ self.W("W_xy1")
        dz1 = dpre2 @ self.W("W_xz1")
        dt1 = dpre2 @ self.W("W_xt1")

        dpre1 = dx1 * sigmoid_np(c["x1_pre"])
        grads["W_xx0"] = dpre1.T @ c["x0"]
        grads["W_xy0"] = dpre1.T @ c["y0"]
        grads["W_xz0"] = dpre1.T @ c["z0"]
        grads["W_xt0"] = dpre1.T @ c["t0"]
        grads["b_x0"] = np.sum(dpre1, axis=0, keepdims=True)

        dpre_y = dy1 * sigmoid_np(c["y_pre"])
        grads["W_yy0"] = dpre_y.T @ c["y0"]
        grads["b_y0"] = np.sum(dpre_y, axis=0, keepdims=True)

        sig_z = sigmoid_np(c["z_pre"])
        dpre_z = dz1 * sig_z * (1.0 - sig_z)
        grads["W_zz0"] = dpre_z.T @ c["z0"]
        grads["b_z0"] = np.sum(dpre_z, axis=0, keepdims=True)

        sig_t = sigmoid_np(c["t_pre"])
        dpre_t = dt1 * sig_t * (1.0 - sig_t)
        grads["W_tt0"] = dpre_t.T @ c["t0"]
        grads["b_t0"] = np.sum(dpre_t, axis=0, keepdims=True)

        return grads


class ConstrainedLinearTorch(nn.Module):
    """A linear layer where we can optionally force weights to stay positive.
    This helps us maintain desired properties like convexity or monotonicity.
    """
        super().__init__()
        self.positive = positive
        if positive:
            init = torch.randn(out_features, in_features) * 0.1 - 2.0
        else:
            bound = np.sqrt(6.0 / (in_features + out_features))
            init = torch.empty(out_features, in_features).uniform_(-bound, bound)
        self.weight_raw = nn.Parameter(init)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def weight(self) -> torch.Tensor:
        # If we need positive weights, apply softplus to keep them in a valid range
        if self.positive:
            return F.softplus(self.weight_raw)
        return self.weight_raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight().T
        if self.bias is not None:
            y = y + self.bias
        return y


class ISNN1Torch(nn.Module):
    """PyTorch version of ISNN-1 using automatic differentiation.
    This is the same model structure as the NumPy version,
    but PyTorch handles all the gradient computation for us.
    """
        super().__init__()
        self.width = width
        self.depth = depth

        self.y_layers = nn.ModuleList()
        self.z_layers = nn.ModuleList()
        self.t_layers = nn.ModuleList()
        for i in range(depth):
            in_dim = 1 if i == 0 else width
            self.y_layers.append(ConstrainedLinearTorch(in_dim, width, positive=True, bias=True))
            self.z_layers.append(ConstrainedLinearTorch(in_dim, width, positive=False, bias=True))
            self.t_layers.append(ConstrainedLinearTorch(in_dim, width, positive=True, bias=True))

        self.x_from_x0 = ConstrainedLinearTorch(1, width, positive=False, bias=True)
        self.x_from_y = ConstrainedLinearTorch(width, width, positive=True, bias=False)
        self.x_from_z = ConstrainedLinearTorch(width, width, positive=False, bias=False)
        self.x_from_t = ConstrainedLinearTorch(width, width, positive=True, bias=False)

        self.x_hidden = nn.ModuleList()
        for _ in range(1, depth):
            self.x_hidden.append(ConstrainedLinearTorch(width, width, positive=True, bias=True))

        self.out = ConstrainedLinearTorch(width, 1, positive=True, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x0 = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        z = X[:, 3:4]

        for layer in self.y_layers:
            y = F.softplus(layer(y))

        for layer in self.z_layers:
            z = torch.sigmoid(layer(z))

        for layer in self.t_layers:
            t = torch.sigmoid(layer(t))

        x = F.softplus(self.x_from_x0(x0) + self.x_from_y(y) + self.x_from_z(z) + self.x_from_t(t))
        for layer in self.x_hidden:
            x = F.softplus(layer(x))

        return self.out(x)


class ISNN2Torch(nn.Module):
    """The simpler ISNN-2 variant in PyTorch.
    One branch layer and two x layers—same setup as the paper.
    """
        super().__init__()
        self.width = width

        self.y0_layer = ConstrainedLinearTorch(1, width, positive=True, bias=True)
        self.z0_layer = ConstrainedLinearTorch(1, width, positive=False, bias=True)
        self.t0_layer = ConstrainedLinearTorch(1, width, positive=True, bias=True)

        self.x0_first = ConstrainedLinearTorch(1, width, positive=False, bias=True)
        self.y0_first = ConstrainedLinearTorch(1, width, positive=True, bias=False)
        self.z0_first = ConstrainedLinearTorch(1, width, positive=False, bias=False)
        self.t0_first = ConstrainedLinearTorch(1, width, positive=True, bias=False)

        self.x_prev_second = ConstrainedLinearTorch(width, width, positive=True, bias=True)
        self.x0_skip_second = ConstrainedLinearTorch(1, width, positive=False, bias=False)
        self.y_second = ConstrainedLinearTorch(width, width, positive=True, bias=False)
        self.z_second = ConstrainedLinearTorch(width, width, positive=False, bias=False)
        self.t_second = ConstrainedLinearTorch(width, width, positive=True, bias=False)

        self.out = ConstrainedLinearTorch(width, 1, positive=True, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x0 = X[:, 0:1]
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        y1 = F.softplus(self.y0_layer(y0))
        z1 = torch.sigmoid(self.z0_layer(z0))
        t1 = torch.sigmoid(self.t0_layer(t0))

        x1 = F.softplus(self.x0_first(x0) + self.y0_first(y0) + self.z0_first(z0) + self.t0_first(t0))
        x2 = F.softplus(
            self.x_prev_second(x1)
            + self.x0_skip_second(x0)
            + self.y_second(y1)
            + self.z_second(z1)
            + self.t_second(t1)
        )

        return self.out(x2)


def build_torch_model(arch: str) -> nn.Module:
    """Create a PyTorch model based on the architecture type."""
    if arch == "isnn1":
        return ISNN1Torch(width=10, depth=2)
    if arch == "isnn2":
        return ISNN2Torch(width=15)
    raise ValueError(f"Unsupported arch: {arch}")


def build_numpy_model(arch: str, seed: int):
    """Create a NumPy model based on the architecture type."""
    rng = np.random.default_rng(seed)
    if arch == "isnn1":
        return ISNN1Numpy(rng=rng, width=10, depth=2)
    if arch == "isnn2":
        return ISNN2Numpy(rng=rng, width=15)
    raise ValueError(f"Unsupported arch: {arch}")


def train_torch_model(
    arch: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_grid: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a PyTorch model with Adam optimization.
    Returns training loss, test loss, and predictions on the grid.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_torch_model(arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)
    X_grid_tensor = torch.from_numpy(X_grid)

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            predictions_test = model(X_test_tensor)
            test_loss = criterion(predictions_test, y_test_tensor)

        train_losses[epoch] = float(loss.item())
        test_losses[epoch] = float(test_loss.item())

    model.eval()
    with torch.no_grad():
        grid_predictions = model(X_grid_tensor).detach().cpu().numpy().reshape(-1)

    return train_losses, test_losses, grid_predictions


def train_numpy_model(
    arch: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_grid: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a NumPy model with manual backpropagation.
    This shows exactly how gradients flow through the network.
    """
    model = build_numpy_model(arch, seed)

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    batch_size = X_train.shape[0]
    for epoch in range(epochs):
        predictions = model.forward(X_train, training=True)
        train_loss = mse_np(predictions, y_train)

        # MSE gradient: 2 * (prediction - target) / batch_size
        output_gradient = 2.0 * (predictions - y_train) / batch_size
        gradients = model.backward(output_gradient)
        model.step(gradients, lr=lr)

        predictions_test = model.forward(X_test, training=False)
        test_loss = mse_np(predictions_test, y_test)

        train_losses[epoch] = train_loss
        test_losses[epoch] = test_loss

    grid_predictions = model.forward(X_grid, training=False).reshape(-1)
    return train_losses, test_losses, grid_predictions


def plot_loss_curves(
    dataset_name: str,
    metrics: dict,
    out_path: Path,
) -> None:
    """Plot training and test loss curves across all models and runs.
    Shows both mean and ±1 standard deviation bands.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = np.arange(1, len(next(iter(metrics.values()))["train_mean"]) + 1)

    for label, item in metrics.items():
        train_mean = item["train_mean"]
        train_std = item["train_std"]
        test_mean = item["test_mean"]
        test_std = item["test_std"]

        # Plot training curves on the left
        axes[0].plot(epochs, train_mean, label=label, linewidth=2)
        axes[0].fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)

        # Plot test curves on the right
        axes[1].plot(epochs, test_mean, label=label, linewidth=2)
        axes[1].fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.2)

    axes[0].set_title(f"{dataset_name}: Epoch vs Training Loss")
    axes[1].set_title(f"{dataset_name}: Epoch vs Testing Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[1].set_ylabel("MSE Loss")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_behavior(
    dataset_name: str,
    grid: np.ndarray,
    y_true: np.ndarray,
    train_upper: float,
    behavior_preds: dict,
    out_path: Path,
) -> None:
    """Visualize how each model behaves across the input range.
    Green zone = training region, orange zone = extrapolation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_list = axes.reshape(-1)

    for axis, (label, predictions) in zip(axes_list, behavior_preds.items()):
        prediction_mean = predictions.mean(axis=0)
        prediction_std = predictions.std(axis=0)

        # Highlight the training region (green)
        axis.axvspan(0.0, train_upper, color="#d8f3dc", alpha=0.4)
        # Highlight the extrapolation region (orange)
        axis.axvspan(train_upper, grid.max(), color="#ffe8d6", alpha=0.35)
        
        # Plot the true function and model predictions
        axis.plot(grid, y_true, color="black", linewidth=2.2, label="True function")
        axis.plot(grid, prediction_mean, color="#1f77b4", linewidth=2.0, label="Predicted mean")
        axis.fill_between(grid, prediction_mean - prediction_std, prediction_mean + prediction_std, color="#1f77b4", alpha=0.2, label="±1 std")
        
        # Mark the boundary between train and extrapolation
        axis.axvline(train_upper, color="#444444", linestyle="--", linewidth=1.3)
        axis.set_title(label)
        axis.set_xlabel("x = y = t = z")
        axis.set_ylabel("Output")
        axis.grid(alpha=0.3)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{dataset_name}: Behavioral Response (Figure 4/6 style)", y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_aggregated_history_csv(path: Path, metric_entry: dict) -> None:
    """Save training history (mean and std of losses) to a CSV file."""
    epochs = len(metric_entry["train_mean"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mean", "train_std", "test_mean", "test_std"])
        for i in range(epochs):
            writer.writerow(
                [
                    i + 1,
                    metric_entry["train_mean"][i],
                    metric_entry["train_std"][i],
                    metric_entry["test_mean"][i],
                    metric_entry["test_std"][i],
                ]
            )


def run_all(args: argparse.Namespace) -> None:
    """Main entry point: generate datasets, train models, and produce plots."""
    # Set up output directories
    project_root = Path(__file__).resolve().parent
    out_root = project_root / "outputs"
    datasets_dir = out_root / "datasets"
    results_dir = out_root / "results"
    plots_dir = out_root / "plots"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define the test problems we're using
    datasets_cfg = {
        "toy_eq12": {
            "function": toy_function_f,
            "train_upper": 4.0,
            "test_upper": 6.0,
            "seed": 123,
            "behavior_upper": 6.0,
        },
        "toy_eq13": {
            "function": toy_function_g,
            "train_upper": 4.0,
            "test_upper": 10.0,
            "seed": 456,
            "behavior_upper": 10.0,
        },
    }

    # List of models to train: (architecture, implementation, display_name)
    model_specs = [
        ("isnn1", "torch", "ISNN-1 (PyTorch)"),
        ("isnn1", "numpy", "ISNN-1 (NumPy manual BP)"),
        ("isnn2", "torch", "ISNN-2 (PyTorch)"),
        ("isnn2", "numpy", "ISNN-2 (NumPy manual BP)"),
    ]

    # Container for all results
    summary = {"config": vars(args), "datasets": {}}

    # Process each test problem
    for dataset_name, cfg in datasets_cfg.items():
        # Generate training and test data using Latin Hypercube sampling
        rng = np.random.default_rng(cfg["seed"])
        X_train = latin_hypercube(args.n_train, 4, 0.0, cfg["train_upper"], rng)
        X_test = latin_hypercube(args.n_test, 4, 0.0, cfg["test_upper"], rng)
        y_train = cfg["function"](X_train).reshape(-1, 1)
        y_test = cfg["function"](X_test).reshape(-1, 1)

        # Normalize data for better optimization convergence
        # We'll rescale predictions back to original scale before computing final metrics
        x_mean = X_train.mean(axis=0, keepdims=True)
        x_std = X_train.std(axis=0, keepdims=True) + 1e-8
        y_mean = y_train.mean(axis=0, keepdims=True)
        y_std = y_train.std(axis=0, keepdims=True) + 1e-8
        y_scale_sq = float((y_std ** 2).item())

        X_train_n = (X_train - x_mean) / x_std
        X_test_n = (X_test - x_mean) / x_std
        y_train_n = (y_train - y_mean) / y_std
        y_test_n = (y_test - y_mean) / y_std

        save_dataset_csv(datasets_dir / f"{dataset_name}_train.csv", X_train, y_train)
        save_dataset_csv(datasets_dir / f"{dataset_name}_test.csv", X_test, y_test)

        # Create evaluation grid for behavioral analysis
        grid = np.linspace(0.0, cfg["behavior_upper"], 500)
        X_grid = np.column_stack([grid, grid, grid, grid])
        X_grid_n = (X_grid - x_mean) / x_std
        y_true = cfg["function"](X_grid)

        # Storage for results from all model runs
        metric_store = {}
        behavior_store = {}
        summary["datasets"][dataset_name] = {}

        # Train all model variants
        for architecture, implementation, label in model_specs:
            # Run multiple times with different random seeds to get error bars
            train_runs = []
            test_runs = []
            prediction_runs = []

            for run_idx in range(args.runs):
                seed = args.base_seed + run_idx

                if implementation == "torch":
                    train_loss_hist, test_loss_hist, grid_pred = train_torch_model(
                        architecture,
                        X_train_n,
                        y_train_n,
                        X_test_n,
                        y_test_n,
                        X_grid_n,
                        epochs=args.epochs,
                        lr=args.lr_torch,
                        seed=seed,
                    )
                else:
                    train_loss_hist, test_loss_hist, grid_pred = train_numpy_model(
                        architecture,
                        X_train_n,
                        y_train_n,
                        X_test_n,
                        y_test_n,
                        X_grid_n,
                        epochs=args.epochs,
                        lr=args.lr_numpy,
                        seed=seed,
                    )

                train_runs.append(train_loss_hist * y_scale_sq)
                test_runs.append(test_loss_hist * y_scale_sq)
                prediction_runs.append((grid_pred.reshape(-1, 1) * y_std + y_mean).reshape(-1))

            # Compute statistics across all runs
            train_array = np.array(train_runs)
            test_array = np.array(test_runs)
            predictions_array = np.array(prediction_runs)

            # Store mean and standard deviation
            metric_store[label] = {
                "train_mean": train_array.mean(axis=0),
                "train_std": train_array.std(axis=0),
                "test_mean": test_array.mean(axis=0),
                "test_std": test_array.std(axis=0),
            }
            behavior_store[label] = predictions_array

            # Save detailed history to CSV
            safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            write_aggregated_history_csv(results_dir / f"{dataset_name}_{safe_label}_history.csv", metric_store[label])

            # Store final metrics in summary
            summary["datasets"][dataset_name][safe_label] = {
                "final_train_loss_mean": float(metric_store[label]["train_mean"][-1]),
                "final_train_loss_std": float(metric_store[label]["train_std"][-1]),
                "final_test_loss_mean": float(metric_store[label]["test_mean"][-1]),
                "final_test_loss_std": float(metric_store[label]["test_std"][-1]),
            }

        # Generate visualization plots
        plot_loss_curves(dataset_name, metric_store, plots_dir / f"{dataset_name}_loss_curves.png")
        plot_behavior(
            dataset_name,
            grid,
            y_true,
            cfg["train_upper"],
            behavior_store,
            plots_dir / f"{dataset_name}_behavior.png",
        )

    # Save all results to a JSON file for easy reference
    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("All training runs completed successfully!")
    print("="*60)
    print(f"\nDatasets: {datasets_dir}")
    print(f"Results:  {results_dir}")
    print(f"Plots:    {plots_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and compare ISNN models using both PyTorch and NumPy implementations")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs for each run")
    parser.add_argument("--runs", type=int, default=10, help="Number of random initializations per model")
    parser.add_argument("--n-train", type=int, default=500, help="Number of training samples per toy dataset")
    parser.add_argument("--n-test", type=int, default=5000, help="Number of test samples per toy dataset")
    parser.add_argument("--lr-torch", type=float, default=2e-3, help="Learning rate for PyTorch models")
    parser.add_argument("--lr-numpy", type=float, default=2e-3, help="Learning rate for NumPy models")
    parser.add_argument("--base-seed", type=int, default=2026, help="Base seed for repeated initializations")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_all(args)
