import numpy as np
import pytest

from opytimark.markers import n_dimensional
from opytimark.markers.cec import year_2005, year_2010, year_2013


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ([1.0, 0.0], 1),
        ([0.0, 1.0], 1_000_000),
        ([1.0, 1.0, 1.0], 1_001_001),
        ([2.0], 4),
    ],
)
def test_elliptic_coefficients(point, expected):
    assert n_dimensional.HighConditionedElliptic()(point) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("benchmark_type", "bias"),
    [(year_2005.F3, -450), (year_2010.F1, 0), (year_2013.F1, 0)],
)
def test_shifted_elliptic_coefficients(benchmark_type, bias):
    benchmark = benchmark_type(auxiliary_data=())
    benchmark.o = np.zeros(2)
    benchmark.M2 = np.eye(2)

    assert benchmark([0.0, 1.0]) == pytest.approx(1_000_000 + bias)


@pytest.mark.parametrize("dimension", [1, 2, 5, 1000])
def test_diagonal_coefficients(dimension):
    matrix = year_2013.T_diagonal(dimension, 100)
    expected = 100 ** np.linspace(0, 0.5, dimension)

    np.testing.assert_allclose(np.diag(matrix), expected)
    np.testing.assert_array_equal(matrix, np.diag(np.diag(matrix)))


@pytest.mark.parametrize("dimension", [1, 2, 1000])
def test_cec_rastrigin_uses_first_coordinate(dimension):
    benchmark = year_2013.F2(auxiliary_data=())
    benchmark.o = np.zeros(dimension)
    x = np.zeros(dimension)
    x[0] = 1

    with np.errstate(invalid="raise", divide="raise"):
        assert benchmark(x) == pytest.approx(1)


def test_asymmetry_only_transforms_positive_coordinates():
    x = np.array([-4, 0, 4])
    original = x.copy()

    with np.errstate(invalid="raise", divide="raise"):
        transformed = year_2013.T_asymmetry(x, 0.5)

    np.testing.assert_allclose(transformed, [-4, 0, 16])
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
def test_asymmetry_preserves_floating_precision(dtype):
    x = np.array([-4, 0, 4], dtype=dtype)
    transformed = year_2013.T_asymmetry(x, 0.5)

    assert transformed.dtype == np.result_type(dtype, np.float64)
    np.testing.assert_allclose(transformed, [-4, 0, 16])


@pytest.mark.parametrize("number", [2, 3, 5, 6, 9, 10])
def test_diagonal_consumers_match_dense_reference(number, monkeypatch):
    benchmark = getattr(year_2013, f"F{number}")()
    x = benchmark.o + np.linspace(-0.2, 0.3, benchmark.dims)
    original = x.copy()
    monkeypatch.setattr(np.random, "permutation", np.arange)

    def transformed(z):
        z = year_2013.T_asymmetry(year_2013.T_irregularity(z), 0.2)
        return z @ year_2013.T_diagonal(z.size, 10)

    shifted = x - benchmark.o
    if number in (2, 3):
        function = n_dimensional.Rastrigin() if number == 2 else n_dimensional.Ackley1()
        expected = function(transformed(shifted))
    else:
        expected = 0
        start = 0
        for size, weight in zip(benchmark.S, benchmark.W):
            rotation = getattr(benchmark, f"R{size}")
            group = rotation @ shifted[start : start + size]
            expected += weight * benchmark.f(transformed(group))
            start += size
        if start < shifted.size:
            expected += benchmark.f(transformed(shifted[start:]))

    assert benchmark(x) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("number", [4, 5, 6, 9, 10, 11, 14, 15, 16])
def test_grouped_rotation_uses_both_matrix_axes(number, monkeypatch):
    benchmark = getattr(year_2010, f"F{number}")(auxiliary_data=(), group_size=2)
    benchmark.o = np.zeros(8)
    benchmark.M = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]])
    benchmark.f = n_dimensional.Sphere()
    monkeypatch.setattr(np.random, "permutation", np.arange)

    expected = 5 * 1_000_000 + 199 if number <= 6 else 204
    assert benchmark(np.arange(1.0, 9.0)) == pytest.approx(expected)


@pytest.mark.parametrize("number", [7, 8])
def test_single_group_scale(number, monkeypatch):
    benchmark = getattr(year_2010, f"F{number}")(auxiliary_data=(), group_size=2)
    benchmark.o = np.zeros(8)
    benchmark.f_1 = benchmark.f_2 = n_dimensional.Sphere()
    monkeypatch.setattr(np.random, "permutation", np.arange)

    assert benchmark(np.arange(1.0, 9.0)) == pytest.approx(5 * 1_000_000 + 199)
