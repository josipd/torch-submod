from __future__ import division, print_function
import torch
from torch.autograd import Variable
from torch_submod.graph_cuts import (
        TotalVariation2d, TotalVariation2dWeighted, TotalVariation1d)
from torch.autograd.gradcheck import gradcheck
from hypothesis import given, settings
import hypothesis.strategies as st


@given(st.integers(10, 100), st.floats(0.1, 10))
def test_1d(n, w):
    x = Variable(torch.randn(n), requires_grad=True)
    w = Variable(torch.Tensor([w]), requires_grad=True)
    tv_args = {'method': 'condattautstring'}
    assert gradcheck(TotalVariation1d(tv_args=tv_args), (x, w),
                     eps=1e-5, atol=1e-2, rtol=1e-3)


@given(st.integers(10, 100), st.floats(0.1, 10))
def test_1dw(n, w):
    x = Variable(10 * torch.randn(n), requires_grad=True)
    w = Variable(0.1 + w * torch.rand(n - 1), requires_grad=True)
    tv_args = {'method': 'tautstring'}
    assert gradcheck(TotalVariation1d(tv_args=tv_args), (x, w),
                     eps=5e-5, atol=5e-2, rtol=1e-2)


@settings(deadline=30000, max_examples=30, timeout=120)
@given(st.integers(5, 20), st.integers(5, 20), st.floats(0.1, 10))
def test_2d(n, m, w):
    x = Variable(torch.randn(n, m), requires_grad=True)
    w = Variable(0.1 + torch.Tensor([w]), requires_grad=True)
    tv_args = {'method': 'dr', 'max_iters': 1000, 'n_threads': 6}
    assert gradcheck(TotalVariation2d(tv_args=tv_args), (x, w),
                     eps=1e-5, atol=1e-2, rtol=1e-3)


@settings(deadline=30000, max_examples=30, timeout=120)
@given(st.integers(5, 10), st.integers(5, 10), st.floats(0.1, 10))
def test_2dw(n, m, w):
    x = Variable(torch.randn(n, m), requires_grad=True)
    w_r = Variable(0.1 + w * torch.rand(n, m-1), requires_grad=True)
    w_c = Variable(0.1 + w * torch.rand(n-1, m), requires_grad=True)
    tv_args = {'max_iters': 1000, 'n_threads': 6}
    assert gradcheck(TotalVariation2dWeighted(tv_args=tv_args), (x, w_r, w_c),
                     eps=1e-5, atol=5e-2, rtol=1e-3)
