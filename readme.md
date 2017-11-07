# `torch-submod`
[![Documentation Status](https://readthedocs.org/projects/torch-submod/badge/?version=latest)](http://torch-submod.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/josipd/torch-submod.svg?branch=master)](https://travis-ci.org/josipd/torch-submod)

A PyTorch library for differentiable submodular minimization.

At the moment only one- and two-dimensional graph cut functions have been
implemented, so that this package provides differentiable (with respect to the
input signal *and* the weights) total variation solvers.

Please refer to the [documentation](https://torch-submod.readthedocs.io)
for more information about the project.
You can also have a look at the following [notebook](notebooks/denoising.ipynb)
that showcases how to learn weights for image denoising.

### Installation

After installing PyTorch, you can install the package with:

```
python setup.py install
```

### Testing

To run the tests you simply have to run:

```
python setup.py test
```

### Bibliography

  * *[DK17]* J. Djolonga and A. Krause. Differentiable learning of submodular models.  In Advances in Neural Information Processing Systems (NIPS), 2017.
  * *[NB17]* V. Niculae and M. Blondel. A regularized framework for sparse and structured neural attention. arXiv preprint arXiv:1705.07704, 2017.
