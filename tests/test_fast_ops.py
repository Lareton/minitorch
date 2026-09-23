"""Regression checks for parallel indexing and the stride-aligned shortcuts."""

import numpy as np
import pytest
from numba import njit

from minitorch import operators
from minitorch.fast_ops import tensor_map, tensor_reduce, tensor_zip
from minitorch.tensor_data import TensorData

pytestmark = pytest.mark.task3_1


@pytest.fixture(scope="module")
def kernels():
    return (
        tensor_map(njit()(operators.neg)),
        tensor_zip(njit()(operators.add)),
        tensor_reduce(njit()(operators.add)),
        tensor_reduce(njit()(operators.mul)),
    )


def data(values):
    values = np.asarray(values, dtype=np.float64)
    return TensorData(values.ravel().copy(), values.shape)


def values(tensor):
    return np.array([tensor.get(i) for i in np.ndindex(tensor.shape)]).reshape(
        tensor.shape
    )


def output(shape, permuted):
    if permuted:
        return data(np.zeros(tuple(reversed(shape)))).permute(
            *reversed(range(len(shape)))
        )
    return data(np.zeros(shape))


@pytest.mark.parametrize("permuted", [False, True])
def test_aligned_layouts(kernels, permuted):
    map_fn, zip_fn, _, _ = kernels
    a = data(np.arange(24.0).reshape(2, 3, 4))
    b = data(np.ones((2, 3, 4)))
    out = data(np.zeros((2, 3, 4)))
    if permuted:
        a, b, out = (t.permute(2, 0, 1) for t in (a, b, out))
    map_fn(*out.tuple(), *a.tuple())
    np.testing.assert_allclose(values(out), -values(a))
    zip_fn(*out.tuple(), *a.tuple(), *b.tuple())
    np.testing.assert_allclose(values(out), values(a) + values(b))


@pytest.mark.parametrize("permuted", [False, True])
def test_broadcast_and_different_layouts(kernels, permuted):
    map_fn, zip_fn, _, _ = kernels
    # a and a contiguous out have equal strides but different shapes.
    a = data([[10.0, 20.0, 30.0]])
    b = data(np.arange(12.0).reshape(3, 4)).permute(1, 0)
    out = output((4, 3), permuted)
    for _ in range(3):
        map_fn(*out.tuple(), *a.tuple())
        np.testing.assert_allclose(values(out), -np.broadcast_to(values(a), (4, 3)))
        zip_fn(*out.tuple(), *a.tuple(), *b.tuple())
        np.testing.assert_allclose(values(out), values(a) + values(b))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reduce_permuted_with_initial_value(kernels, axis):
    _, _, add_fn, mul_fn = kernels
    a = data(np.arange(1.0, 25.0).reshape(2, 3, 4)).permute(2, 0, 1)
    shape = list(a.shape)
    shape[axis] = 1
    out = output(tuple(shape), True)
    for kernel, start, reduce_fn in [(add_fn, 7.0, np.sum), (mul_fn, 2.0, np.prod)]:
        out._storage[:] = start
        kernel(*out.tuple(), *a.tuple(), axis)
        expected = reduce_fn(values(a), axis=axis, keepdims=True)
        expected = expected + start if kernel is add_fn else expected * start
        np.testing.assert_allclose(values(out), expected)
