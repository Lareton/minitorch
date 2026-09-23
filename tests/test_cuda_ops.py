"""CUDA edge cases. Can also run with NUMBA_ENABLE_CUDASIM=1."""

import numpy as np
import pytest
from numba import cuda

import minitorch

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
backend = minitorch.TensorBackend(minitorch.CudaOps)


def tensor(values, grad=False):
    return minitorch.tensor(np.asarray(values).tolist(), backend=backend, requires_grad=grad)


def host(t):
    return t.contiguous()._tensor._storage.copy_to_host().reshape(t.shape)


@pytest.mark.task3_3
def test_cuda_broadcast_permute():
    a = tensor(np.arange(6.).reshape(2, 3)).permute(1, 0)
    b = tensor([10., 20.])
    expected = np.arange(6.).reshape(2, 3).T
    np.testing.assert_allclose(host(-a), -expected)
    np.testing.assert_allclose(host(a + b), expected + [10., 20.])
    assert a.permute(1, 0)._tensor._storage is a._tensor._storage


@pytest.mark.task3_3
@pytest.mark.parametrize("size", [1, 31, 32, 33, 1024, 1025, 2051])
def test_cuda_long_reduce(size):
    a_np = np.arange(2 * size, dtype=float).reshape(size, 2) / 10000
    a = tensor(a_np).permute(1, 0)
    result = a.sum(1)
    assert result.shape == (2, 1)
    np.testing.assert_allclose(host(result), a_np.T.sum(axis=1, keepdims=True))


@pytest.mark.task3_3
def test_cuda_reduce_start():
    a_np = np.full((2, 1025), 1.001)
    a = tensor(a_np)
    for fn, start, expected in [
        (minitorch.operators.add, 7., a_np.sum(axis=1, keepdims=True) + 7),
        (minitorch.operators.mul, 2., a_np.prod(axis=1, keepdims=True) * 2),
    ]:
        result = minitorch.CudaOps.reduce(fn, start)(a, 1)
        np.testing.assert_allclose(host(result), expected)


@pytest.mark.task3_3
def test_cuda_backward():
    a_np = np.array([[0.2, 0.7], [1.1, 0.4]])
    a = tensor(a_np, grad=True)
    b = tensor([0.5, 0.8], grad=True)
    ((a * b).relu().sigmoid().log().exp()).sum().backward()
    s = 1 / (1 + np.exp(-a_np * [0.5, 0.8]))
    np.testing.assert_allclose(host(a.grad), s * (1 - s) * [0.5, 0.8], rtol=1e-4)
    np.testing.assert_allclose(host(b.grad), (s * (1 - s) * a_np).sum(axis=0), rtol=1e-4)


@pytest.mark.task3_4
@pytest.mark.parametrize("broadcast_left", [False, True])
def test_cuda_matmul_strides(broadcast_left):
    rng = np.random.default_rng(42)
    batches_a, batches_b = (1, 2) if broadcast_left else (2, 1)
    a_np = rng.random((batches_a, 17, 33))
    b_np = rng.random((batches_b, 19, 17))
    a = tensor(a_np).permute(0, 2, 1)
    b = tensor(b_np).permute(0, 2, 1)
    np.testing.assert_allclose(host(a @ b), a_np.transpose(0, 2, 1) @ b_np.transpose(0, 2, 1))


@pytest.mark.task3_4
def test_cuda_matmul_backward():
    a_np = np.arange(12., dtype=float).reshape(2, 2, 3) / 10
    b_np = np.arange(6., dtype=float).reshape(1, 3, 2) / 10
    a, b = tensor(a_np, True), tensor(b_np, True)
    (a @ b).sum().backward()
    g = np.ones((2, 2, 2))
    np.testing.assert_allclose(host(a.grad), g @ b_np.transpose(0, 2, 1))
    np.testing.assert_allclose(host(b.grad), (a_np.transpose(0, 2, 1) @ g).sum(axis=0, keepdims=True))
