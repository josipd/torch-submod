"""Argmin-differentiable total variation functions."""
from __future__ import division, print_function

# Must import these first, gomp issues with pytorch.
from prox_tv import tv1w_2d, tv1_2d, tv1w_1d, tv1_1d

import numpy as np
from sklearn.isotonic import isotonic_regression as isotonic

import torch
from torch.autograd import Function
from .blocks import blockwise_means, blocks_2d

__all__ = ("TotalVariation2d", "TotalVariation2dWeighted", "TotalVariation1d")


class TotalVariation2dWeighted(Function):
    r"""A two dimensional total variation function.

    Specifically, given as input the unaries `x`, positive row weights
    :math:`\mathbf{r}` and positive column weights :math:`\mathbf{c}`, the
    output is computed as

    .. math::

        \textrm{argmin}_{\mathbf z}
            \frac{1}{2} \|\mathbf{x}-\mathbf{z}\|^2 +
            \sum_{i, j} r_{i,j} |z_{i, j} - z_{i, j + 1}| +
            \sum_{i, j} c_{i,j} |z_{i, j} - z_{i + 1, j}|.

    Arguments
    ---------
        refine: bool
            If set the solution will be refined with isotonic regression.
        avearge_2d: bool
            How to compute the approximate derivative.

            If ``True``, will average within each connected component.
            If ``False``, it will average within each block of equal values.
            Typically, you want this set to true.
        tv_args: dict
            The dictionary of arguments passed to the total variation solver.
        """
    def __init__(self, refine=True, average_connected=True, tv_args=None):
        self.tv_args = tv_args if tv_args is not None else {}
        self.refine = refine
        self.average_connected = average_connected

    def forward(self, x, weights_row, weights_col):
        r"""Solve the total variation problem and return the solution.

        Arguments
        ---------
            x: :class:`torch:torch.Tensor`
                A tensor with shape ``(m, n)`` holding the input signal.
            weights_row: :class:`torch:torch.Tensor`
                The horizontal edge weights.

                Tensor of shape ``(m, n - 1)``, or ``(1,)`` if all weights
                are equal.
            weights_col: :class:`torch:torch.Tensor`
                The vertical edge weights.

                Tensor of shape ``(m - 1, n)``, or ``(1,)`` if all weights
                are equal.

        Returns
        -------
        :class:`torch:torch.Tensor`
            The solution to the total variation problem, of shape ``(m, n)``.
        """
        opt = tv1w_2d(x.numpy(), weights_col.numpy(), weights_row.numpy(),
                      **self.tv_args)
        if self.refine:
            opt = self._refine(opt, x, weights_row, weights_col)
        opt = torch.Tensor(opt).view_as(x)
        self.save_for_backward(opt)
        return opt

    def _grad_x(self, opt, grad_output):
        if self.average_connected:
            blocks = blocks_2d(opt.numpy())
        else:
            _, blocks = np.unique(opt.numpy().ravel(), return_inverse=True)
        grad_x = blockwise_means(blocks.ravel(), grad_output.numpy().ravel())
        # We need the clone as there seems to e a double-free error in py27,
        # namely, torch free()s the array after numpy has already free()d it.
        return torch.from_numpy(grad_x).view(opt.size()).clone()

    def _grad_w_row(self, opt, grad_x):
        """Compute the derivative with respect to the row weights."""
        diffs_row = torch.sign(opt[:, :-1] - opt[:, 1:])
        return - diffs_row * (grad_x[:, :-1] - grad_x[:, 1:])

    def _grad_w_col(self, opt, grad_x):
        """Compute the derivative with respect to the column weights."""
        diffs_col = torch.sign(opt[:-1, :] - opt[1:, :])
        return - diffs_col * (grad_x[:-1, :] - grad_x[1:, :])

    def backward(self, grad_output):
        opt, = self.saved_tensors
        grad_weights_row, grad_weights_col = None, None
        grad_x = self._grad_x(opt, grad_output)

        if self.needs_input_grad[1]:
            grad_weights_row = self._grad_w_row(opt, grad_x)

        if self.needs_input_grad[2]:
            grad_weights_col = self._grad_w_col(opt, grad_x)

        return grad_x, grad_weights_row, grad_weights_col

    def _refine(self, opt, x, weights_row, weights_col):
        """Refine the solution by solving an isotonic regression.

        The weights can either be two-dimensional tensors, or of shape (1,)."""
        idx = np.argsort(opt.ravel())  # Will pick an arbitrary order cone.
        ordered_vec = np.zeros_like(idx, dtype=np.float)
        ordered_vec[idx] = np.arange(np.size(opt))
        f = self._linearize(ordered_vec.reshape(opt.shape),
                            weights_row.numpy(), weights_col.numpy())
        opt_idx = isotonic((x.view(-1).numpy() - f.ravel())[idx])
        opt = np.zeros_like(opt_idx)
        opt[idx] = opt_idx
        return opt

    def _linearize(self, y, weights_row, weights_col):
        """Compute a linearization of the graph-cut function at the given point.

        Arguments
        ---------
        y : numpy.ndarray
            The point where the linearization is computed, shape ``(m, n)``.
        weights_row : numpy.ndarray
            The non-negative row weights, with shape ``(m, n - 1)``.
        y : numpy.ndarray
            The non-negative column weights, with shape ``(m - 1, n)``.

        Returns
        -------
        numpy.ndarray
            The linearization of the graph-cut function at ``y``."""
        diffs_col = np.sign(y[1:, :] - y[:-1, :])
        diffs_row = np.sign(y[:, 1:] - y[:, :-1])

        f = np.zeros_like(y)  # The linearization.
        f[:, 1:] += diffs_row * weights_row
        f[:, :-1] -= diffs_row * weights_row
        f[1:, :] += diffs_col * weights_col
        f[:-1, :] -= diffs_col * weights_col

        return f


class TotalVariation2d(TotalVariation2dWeighted):
    r"""A two dimensional total variation function with tied edge weights.

    Specifically, given as input the unaries `x` and edge weight ``w``, the
    returned value is given by:

    .. math::

        \textrm{argmin}_{\mathbf z}
            \frac{1}{2} \|\mathbf{x}-\mathbf{z}\|^2 +
            \sum_{i, j} w |z_{i, j} - z_{i, j + 1}| +
            \sum_{i, j} w |z_{i, j} - z_{i + 1, j}|.

    Arguments
    ---------
        refine: bool
            If set the solution will be refined with isotonic regression.
        avearge_2d: bool
            How to compute the approximate derivative.

            If ``True``, will average within each connected component.
            If ``False``, it will average within each block of equal values.
            Typically, you want this set to true.
        tv_args: dict
            The dictionary of arguments passed to the total variation solver.
        """
    def __init__(self, refine=True, average_connected=True, tv_args=None):
        super(TotalVariation2d, self).__init__(
            refine=refine,
            average_connected=average_connected,
            tv_args=tv_args)

    def forward(self, x, w):
        r"""Solve the total variation problem and return the solution.

        Arguments
        ---------
            x: :class:`torch:torch.Tensor`
                A tensor with shape ``(m, n)`` holding the input signal.
            weights_row: :class:`torch:torch.Tensor`
                The horizontal edge weights.

                Tensor of shape ``(m, n - 1)``, or ``(1,)`` if all weights
                are equal.
            weights_col: :class:`torch:torch.Tensor`
                The vertical edge weights.

                Tensor of shape ``(m - 1, n)``, or ``(1,)`` if all weights
                are equal.

        Returns
        -------
        :class:`torch:torch.Tensor`
            The solution to the total variation problem, of shape ``(m, n)``.
        """
        assert w.size() == (1,)
        opt = tv1_2d(x.numpy(), w.numpy()[0], **self.tv_args)

        if self.refine:  # Should we improve it with isotonic regression.
            opt = self._refine(opt, x, w, w)

        opt = torch.Tensor(opt).view_as(x)
        self.save_for_backward(opt)
        return opt

    def backward(self, grad_output):
        opt, = self.saved_tensors
        grad_x = self._grad_x(opt, grad_output)
        grad_w = None

        if self.needs_input_grad[1]:
            grad_w = (torch.sum(self._grad_w_row(opt, grad_x)) +
                      torch.sum(self._grad_w_col(opt, grad_x)))
            grad_w = torch.Tensor([grad_w])

        return grad_x, grad_w


class TotalVariation1d(TotalVariation2dWeighted):
    r"""A one dimensional total variation function.

    Specifically, given as input the signal `x` and weights :math:`\mathbf{w}`,
    the output is computed as

    .. math::

        \textrm{argmin}_{\mathbf z}
            \frac{1}{2} \|\mathbf{x}-\mathbf{z}\|^2 +
            \sum_{i=1}^{n-1} w_i |z_i - z_{i+1}|.

    Arguments
    ---------
        average_connected: bool
            How to compute the approximate derivative.

            If ``True``, will average within each connected component.
            If ``False``, it will average within each block of equal values.
            Typically, you want this set to true.
        tv_args: dict
            The dictionary of arguments passed to the total variation solver.
        """
    def __init__(self, average_connected=True, tv_args=None):
        if tv_args is None:
            self.tv_args = {}
        else:
            self.tv_args = tv_args
        self.average_connected = average_connected

    def forward(self, x, weights):
        r"""Solve the total variation problem and return the solution.

        Arguments
        ---------
            x: :class:`torch:torch.Tensor`
                A tensor with shape ``(n,)`` holding the input signal.
            weights: :class:`torch:torch.Tensor`
                The edge weights.

                Shape ``(n-1,)``, or ``(1,)`` if all weights are equal.

        Returns
        -------
        :class:`torch:torch.Tensor`
            The solution to the total variation problem, of shape ``(m, n)``.
        """
        self.equal_weights = weights.size() == (1,)
        if self.equal_weights:
            opt = tv1_1d(x.numpy().ravel(), weights.numpy()[0],
                         **self.tv_args)
        else:
            opt = tv1w_1d(x.numpy().ravel(), weights.numpy().ravel(),
                          **self.tv_args)
        opt = torch.Tensor(opt).view_as(x)

        self.save_for_backward(opt)
        return opt

    def backward(self, grad_output):
        opt, = self.saved_tensors
        grad_weights = None

        opt = opt.view((1, -1))
        grad_x = self._grad_x(opt, grad_output)

        if self.needs_input_grad[1]:
            grad_weights = self._grad_w_row(opt, grad_x).view(-1)
            if self.equal_weights:
                grad_weights = torch.Tensor([torch.sum(grad_weights)])

        return grad_x.view(-1), grad_weights
