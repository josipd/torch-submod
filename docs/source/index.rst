########################################
Welcome to torch-submod's documentation!
########################################

************
Introduction
************

A library implementing layers that solve the min-norm problem for submodular
functions.
The computation of the Jacobian (i.e., backpropagation) is done using the
methods from :cite:`djolonga17learning`. At the moment only graph-cuts on two
dimensional grids are implemented, in which case the min-norm problem is also
known as a total variation problem.

At the moment only one- and two-dimensional graph cut functions have been
implemented, so that this package provides differentiable (with respect to the
input signal *and* the weights) total variation solvers.

************
Installation
************

Once you install PyTorch (following
`these instructions <http://www.pytorch.org>`_) , you can install
the package as::

  python setup.py install

*****
Usage
*****

For example, let us try to learn row- and column- weights that will denoise
a simple image.
Let us create an image that is zero everywhere, except its left-right corner
that is filled with ones. Then, we will corrupt it with normal noise, and try
to recover it using a total-variation solver with learned weights.

Note that an extended version of the example below, together with visualization
is provided in the repository as a
`jupyter notebook <https://github.com/josipd/torch-submod/blob/master/notebooks/denoising.ipynb>`_).


 >>> from __future__ import division, print_function
 >>> import torch
 >>> from torch.autograd import Variable
 >>> from torch_submod.graph_cuts import TotalVariation2dWeighted as tv2d
 >>>
 >>> torch.manual_seed(0)
 >>> m, n = 50, 100  # The image dimensions.
 >>> std = 1e-1  # The standard deviation of noise.
 >>> x = torch.zeros((m, n))
 >>> x[:m//2, :n//2] += 1
 >>> x_noisy = x + torch.normal(torch.zeros(x.size()))
 >>>
 >>> x = Variable(x, requires_grad=False)
 >>> x_noisy = Variable(x_noisy, requires_grad=False)
 >>>
 >>> # The learnable parameters.
 >>> log_w_row = Variable(- 3 * torch.ones(1), requires_grad=True)
 >>> log_w_col = Variable(- 3 * torch.ones(1), requires_grad=True)
 >>> scale = Variable(torch.ones(1), requires_grad=True)
 >>>
 >>> optimizer = torch.optim.SGD([log_w_row, log_w_col, scale], lr=.5)
 >>> losses = []
 >>> for iter_no in range(1000):
 >>>     w_row = torch.exp(log_w_row)
 >>>     w_col = torch.exp(log_w_col)
 >>>     y = tv2d()(scale * x_noisy,
 >>>                w_row.expand((m, n-1)), w_col.expand((m - 1, n)))
 >>>     optimizer.zero_grad()
 >>>     loss = torch.mean((y - x)**2)
 >>>     loss.backward()
 >>>     if iter_no % 100 == 0:
 >>>         losses.append(loss.data[0])
 >>>     optimizer.step()
 >>> print('\n'.join(map(str, losses)))
 0.809337258339
 0.100806325674
 0.0123300831765
 0.00451330607757
 0.00304582691751
 0.00262771383859
 0.00248298258521
 0.00242520542815
 0.00239872303791
 0.00239089410752



================
Function classes
================

^^^^^^^^^^
Graph cuts
^^^^^^^^^^

To solve the total-variation problem we are using the
`prox_tv <https://pythonhosted.org/prox_tv/>`_  library. Please refer to the
documentation accompanying that package to find out more about the set of
available methods. Namely, each function accepts a ``tv_args`` dictionary
argument, which is passed onto the solver. The idea to average within the
connected components, enabled when ``average_connected=True``, first appeared
for the one-dimensional case in :cite:`niculae2017regularized`.

*Note*: At the moment the total variation problems can be solved only
*on the CPU*, so please make sure that all variables are placed on the CPU.


.. autoclass:: torch_submod.graph_cuts.TotalVariation2dWeighted
    :members: __init__, forward

.. autoclass:: torch_submod.graph_cuts.TotalVariation2d
    :members: __init__, forward

.. autoclass:: torch_submod.graph_cuts.TotalVariation1d
    :members: __init__, forward

************
Bibliography
************

.. bibliography:: refs.bib


==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
