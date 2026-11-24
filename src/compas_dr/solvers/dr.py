from collections.abc import Callable
from collections.abc import Sequence
from math import sqrt
from typing import Any
from typing import Literal

K = [
    [0.0],
    [0.5, 0.5],
    [0.5, 0.0, 0.5],
    [1.0, 0.0, 0.0, 1.0],
]


class Coeff:
    def __init__(self, c: float) -> None:
        self.c = c
        self.a = (1 - c * 0.5) / (1 + c * 0.5)
        self.b = 0.5 * (1 + self.a)


def norm_vector(vector: Sequence[float]) -> float:
    """
    Calculate the length of a vector.

    Parameters
    ----------
    vector : list
        XYZ components of the vector.

    Returns
    -------
    float
        The L2 norm, or *length* of the vector.

    Examples
    --------
    >>>
    """
    return sqrt(sum(axis**2 for axis in vector))


def norm_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    """
    Calculate the norm of each vector in a list of vectors.

    Parameters
    ----------
    vectors : list
        A list of vectors

    Returns
    -------
    list
        A list with the lengths of all vectors.

    Examples
    --------
    >>>
    """
    return [norm_vector(vector) for vector in vectors]


def adjacency_from_edges(edges: Sequence[Sequence[int]]) -> dict[int, list[int]]:
    """Construct an adjacency dictionary from a set of edges.

    Parameters
    ----------
    edges : list
        A list of index pairs.

    Returns
    -------
    dict
        A dictionary mapping each index in the list of index pairs
        to a list of adjacent indices.

    Examples
    --------
    >>>
    """
    adj: dict[int, list[int]] = {}
    for i, j in iter(edges):
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    return adj


def dr(
    vertices: list[list[float]],
    edges: Sequence[Sequence[int]],
    fixed: Sequence[int],
    loads: Sequence[Sequence[float]],
    qpre: Sequence[float] | None,
    fpre: Sequence[float] | None = None,
    lpre: Sequence[float] | None = None,
    linit: Sequence[float] | None = None,
    E: Sequence[float] | None = None,
    radius: Sequence[float] | None = None,
    kmax: int = 100,
    dt: float = 1.0,
    tol1: float = 1e-3,
    tol2: float = 1e-6,
    c: float = 0.1,
    callback: Callable[[int, list[list[float]], tuple[float, float], Any], object] | None = None,
    callback_args: object = None,
) -> tuple[list[list[float]], list[float], list[float], list[float], list[list[float]]]:
    """Implementation of dynamic relaxation with RK integration scheme in pure Python.

    Parameters
    ----------
    vertices : list
        XYZ coordinates of the vertices.
    edges : list
        Connectivity of the vertices.
    fixed : list
        Indices of the fixed vertices.
    loads : list
        XYZ components of the loads on the vertices.
    qpre : list
        Prescribed force densities in the edges.
    fpre : list, optional
        Prescribed forces in the edges.
    lpre : list, optional
        Prescribed lengths of the edges.
    linit : list, optional
        Initial edge lengths. Accepted for compatibility with the NumPy
        solvers.
    E : list, optional
        Edge stiffnesses. Accepted for compatibility with the NumPy solvers.
    radius : list, optional
        Edge radii. Accepted for compatibility with the NumPy solvers.
    kmax : int, optional
        Maximum number of iterations.
    dt : float, optional
        The time step.
    tol1 : float, optional
        Convergence criterion for the residual forces.
    tol2 : float, optional
        Convergence criterion for displacement between iterations.
    c : float, optional
        Damping factor for viscous damping.
    callback : callable, optional
        A user-defined callback that is called after every iteration.
        The callback will be called with ``k`` the current iteration,
        ``X`` the coordinates at iteration ``k``,
        ``crit1, crit2`` the values of the stoppage criteria at iteration ``k``,
        and ``callback_args`` the optional additional arguments.
    callback_args : tuple, optional
        Additional arguments to be passed to the callback.

    Returns
    -------
    xyz : list of list of float
        XYZ coordinates of the equilibrium geometry, in vertex order.
    q : list of float
        Force densities in edge order.
    f : list of float
        Axial forces in edge order.
    l : list of float
        Lengths in edge order.
    r : list of list of float
        Residual forces in vertex order.

    Notes
    -----
    The solver uses four-step Runge–Kutta integration and updates the input
    vertex list in place. The force model includes prescribed force densities,
    forces and lengths. Elastic stiffness terms are available in the NumPy
    solvers.
    """
    if callback:
        if not callable(callback):
            raise Exception("The callback is not callable.")

    # --------------------------------------------------------------------------
    # preprocess
    # --------------------------------------------------------------------------

    n = len(vertices)
    e = len(edges)

    i_nbrs = adjacency_from_edges(edges)

    ij_e = {(i, j): index for index, (i, j) in enumerate(edges)}
    ij_e.update({(j, i): index for (i, j), index in ij_e.items()})

    coeff = Coeff(c)
    ca = coeff.a
    cb = coeff.b
    free = list(set(range(n)) - set(fixed))

    # --------------------------------------------------------------------------
    # attribute arrays
    # --------------------------------------------------------------------------

    X = vertices
    P = loads
    Qpre = qpre or [0.0 for _ in range(e)]
    Fpre = fpre or [0.0 for _ in range(e)]
    Lpre = lpre or [0.0 for _ in range(e)]

    # --------------------------------------------------------------------------
    # initial values
    # --------------------------------------------------------------------------

    densities = [1.0 for _ in range(e)]
    lengths = [sum((X[i][axis] - X[j][axis]) ** 2 for axis in (0, 1, 2)) ** 0.5 for i, j in iter(edges)]
    forces = [q * length for q, length in zip(densities, lengths)]
    masses = [sum(0.5 * dt**2 * densities[ij_e[(i, j)]] for j in i_nbrs[i]) for i in range(n)]
    V = [[0.0, 0.0, 0.0] for _ in range(n)]
    R = [[0.0, 0.0, 0.0] for _ in range(n)]
    dX = [[0.0, 0.0, 0.0] for _ in range(n)]

    # --------------------------------------------------------------------------
    # helpers
    # --------------------------------------------------------------------------

    def update_R() -> None:
        for i in range(n):
            x = X[i][0]
            y = X[i][1]
            z = X[i][2]
            f = [0.0, 0.0, 0.0]
            for j in i_nbrs[i]:
                q = densities[ij_e[(i, j)]]
                f[0] += q * (X[j][0] - x)
                f[1] += q * (X[j][1] - y)
                f[2] += q * (X[j][2] - z)
            R[i] = [P[i][axis] + f[axis] for axis in (0, 1, 2)]

    def rk(X0: list[list[float]], V0: list[list[float]], steps: Literal[2, 4] = 2) -> list[list[float]]:
        def a(t: float, V: list[list[float]]) -> list[list[float]]:
            dX = [[V[i][axis] * t for axis in (0, 1, 2)] for i in range(n)]
            for i in free:
                X[i] = [X0[i][axis] + dX[i][axis] for axis in (0, 1, 2)]
            update_R()
            return [[cb * R[i][axis] / masses[i] for axis in (0, 1, 2)] for i in range(n)]

        if steps == 2:
            weights = [0.0, 1.0]
            a0 = a(K[0][0] * dt, V0)
            k0 = [[dt * a0[i][axis] for axis in (0, 1, 2)] for i in range(n)]
            a1 = a(
                K[1][0] * dt,
                [[V0[i][axis] + K[1][1] * k0[i][axis] for axis in (0, 1, 2)] for i in range(n)],
            )
            k1 = [[dt * a1[i][axis] for axis in (0, 1, 2)] for i in range(n)]
            return [[weights[0] * k0[i][axis] + weights[1] * k1[i][axis] for axis in (0, 1, 2)] for i in range(n)]

        if steps == 4:
            weights = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
            a0 = a(K[0][0] * dt, V0)
            k0 = [[dt * a0[i][axis] for axis in (0, 1, 2)] for i in range(n)]
            a1 = a(
                K[1][0] * dt,
                [[V0[i][axis] + K[1][1] * k0[i][axis] for axis in (0, 1, 2)] for i in range(n)],
            )
            k1 = [[dt * a1[i][axis] for axis in (0, 1, 2)] for i in range(n)]
            a2 = a(
                K[2][0] * dt,
                [[V0[i][axis] + K[2][1] * k0[i][axis] + K[2][2] * k1[i][axis] for axis in (0, 1, 2)] for i in range(n)],
            )
            k2 = [[dt * a2[i][axis] for axis in (0, 1, 2)] for i in range(n)]
            a3 = a(
                K[3][0] * dt,
                [[V0[i][axis] + K[3][1] * k0[i][axis] + K[3][2] * k1[i][axis] + K[3][3] * k2[i][axis] for axis in (0, 1, 2)] for i in range(n)],
            )
            k3 = [[dt * a3[i][axis] for axis in (0, 1, 2)] for i in range(n)]
            return [[weights[0] * k0[i][axis] + weights[1] * k1[i][axis] + weights[2] * k2[i][axis] + weights[3] * k3[i][axis] for axis in (0, 1, 2)] for i in range(n)]

        raise NotImplementedError

    # --------------------------------------------------------------------------
    # start iterating
    # --------------------------------------------------------------------------

    for k in range(kmax):
        Qfpre = [a / b if b else 0 for a, b in zip(Fpre, lengths)]
        Qlpre = [a / b if b else 0 for a, b in zip(forces, Lpre)]

        densities = [a + b + c for a, b, c in zip(Qpre, Qfpre, Qlpre)]
        masses = [sum(0.5 * dt**2 * densities[ij_e[(i, j)]] for j in i_nbrs[i]) for i in range(n)]

        X0 = [_[:] for _ in X]
        V0 = [[ca * V[i][axis] for axis in (0, 1, 2)] for i in range(n)]

        # RK
        dV = rk(X0, V0, steps=4)

        # update
        for i in free:
            V[i] = [V0[i][axis] + dV[i][axis] for axis in (0, 1, 2)]
            dX[i] = [V[i][axis] * dt for axis in (0, 1, 2)]
            X[i] = [X0[i][axis] + dX[i][axis] for axis in (0, 1, 2)]

        lengths = [sum((X[i][axis] - X[j][axis]) ** 2 for axis in (0, 1, 2)) ** 0.5 for i, j in iter(edges)]
        forces = [q * length for q, length in zip(densities, lengths)]

        update_R()

        # crits
        crit1 = max(norm_vectors([R[i] for i in free]))
        crit2 = max(norm_vectors([dX[i] for i in free]))

        # callback
        if callback:
            callback(k, X, (crit1, crit2), callback_args)

        # convergence
        if crit1 < tol1:
            break
        if crit2 < tol2:
            break

    # --------------------------------------------------------------------------
    # update
    # --------------------------------------------------------------------------

    update_R()

    return X, densities, forces, lengths, R
