from math import sqrt

import numpy as np
import pytest
from compas.geometry import Line
from numpy.typing import NDArray

from compas_dr.constraints import LineConstraint
from compas_dr.numdata import InputData
from compas_dr.solvers import dr
from compas_dr.solvers import dr_constrained_numpy
from compas_dr.solvers import dr_numpy


@pytest.mark.parametrize("constrained", [False, True])
def test_loaded_network_converges_to_equilibrium(constrained: bool) -> None:
    """Check static equilibrium independently of the DR iteration scheme."""
    # Set stiffness and other prestress terms to their zero defaults. The
    # equilibrium of the resulting constant-q network is 4X = A + 3B + P.
    indata = InputData(
        vertices=[[1, 0, 0], [0, 0, 0], [4, 0, 0]],
        edges=[(0, 1), (0, 2)],
        fixed=[1, 2],
        loads=[[0, 0, -4], [0, 0, 0], [0, 0, 0]],
        qpre=[1, 3],
    )
    iterations: list[tuple[int, float]] = []
    marker = ["example"]

    def callback(iteration: int, xyz: NDArray[np.float64], residual: float, displacement: float, args: object) -> None:
        assert xyz.shape == (3, 3)
        assert np.isfinite(displacement)
        assert args is marker
        iterations.append((iteration, residual))

    with np.errstate(divide="ignore", invalid="ignore"):
        if constrained:
            result = dr_constrained_numpy(indata=indata, constraints=[None, None, None], kmax=1000, tol1=1e-8, tol2=0, callback=callback, callback_args=marker)
        else:
            result = dr_numpy(indata=indata, kmax=1000, tol1=1e-8, tol2=0, callback=callback, callback_args=marker)

    assert iterations[0][0] == 0
    assert iterations[-1][0] < 999
    assert iterations[-1][1] < 1e-8
    # The convergence criterion bounds the force error, with an allowance
    # for propagating the corresponding coordinate error to the edge forces.
    np.testing.assert_allclose(result.xyz, [[3, 0, -1], [0, 0, 0], [4, 0, 0]], rtol=0, atol=1e-8)
    np.testing.assert_allclose(result.residuals, [[0, 0, 0], [3, 0, -1], [-3, 0, -3]], rtol=0, atol=1e-8)
    np.testing.assert_allclose(result.q, [[1], [3]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.lengths, [[sqrt(10)], [sqrt(2)]], rtol=0, atol=1e-8)
    np.testing.assert_allclose(result.forces, [[sqrt(10)], [3 * sqrt(2)]], rtol=0, atol=1e-8)


def test_pure_python_solver_reaches_the_same_equilibrium() -> None:
    xyz, q, forces, lengths, residuals = dr(
        vertices=[[1, 0, 0], [0, 0, 0], [4, 0, 0]],
        edges=[(0, 1), (0, 2)],
        fixed=[1, 2],
        loads=[[0, 0, -4], [0, 0, 0], [0, 0, 0]],
        qpre=[1, 3],
        kmax=1000,
        tol1=1e-8,
        tol2=0,
    )

    np.testing.assert_allclose(xyz, [[3, 0, -1], [0, 0, 0], [4, 0, 0]], rtol=0, atol=1e-8)
    np.testing.assert_allclose(q, [1, 3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(forces, [sqrt(10), 3 * sqrt(2)], rtol=0, atol=1e-8)
    np.testing.assert_allclose(lengths, [sqrt(10), sqrt(2)], rtol=0, atol=1e-8)
    np.testing.assert_allclose(residuals[0], [0, 0, 0], rtol=0, atol=1e-8)


def test_sliding_support_converges_to_its_equilibrium_line_location() -> None:
    indata = InputData(
        vertices=[[0, 0, 0], [2, 1, 0], [1, 0, 0]],
        edges=[(0, 2), (1, 2)],
        fixed=[0, 1],
        loads=[[0, 0, 0]] * 3,
        qpre=[1, 1],
    )
    constraint = LineConstraint(Line([0, 0, 0], [3, 0, 0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = dr_constrained_numpy(indata=indata, constraints=[constraint, None, None], tol1=1e-8, tol2=0, kmax=1000)

    # The free node is the midpoint. In equilibrium the sliding support is
    # directly below the fixed one. DR checks residuals at free nodes only,
    # so the baseline stops with a small remaining error at the support.
    np.testing.assert_allclose(result.xyz, [[2, 0, 0], [2, 1, 0], [2, 0.5, 0]], rtol=0, atol=1e-5)
    np.testing.assert_allclose(result.residuals[2], [0, 0, 0], rtol=0, atol=1e-8)
    np.testing.assert_allclose(tuple(constraint.normal), [0, 0.5, 0], rtol=0, atol=1e-12)
