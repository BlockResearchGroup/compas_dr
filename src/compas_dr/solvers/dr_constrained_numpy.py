from typing import Callable
from typing import Literal
from typing import Sequence

import numpy
import scipy.sparse  # noqa: F401
from compas.linalg import normrow
from numpy import isinf
from numpy import isnan
from scipy.linalg import norm
from scipy.sparse import diags

import compas_dr.numdata
from compas_dr.constraints import Constraint
from compas_dr.numdata import ResultData

old_settings = numpy.seterr(divide="ignore")


K = [
    [0.0],
    [0.5, 0.5],
    [0.5, 0.0, 0.5],
    [1.0, 0.0, 0.0, 1.0],
]


class Coeff:
    def __init__(self, c):
        self.c = c
        self.a = (1 - c * 0.5) / (1 + c * 0.5)
        self.b = 0.5 * (1 + self.a)


def dr_constrained_numpy(
    *,
    indata: compas_dr.numdata.InputData,
    constraints: Sequence[Constraint],
    kmax: int = 10000,
    dt: float = 1.0,
    tol1: float = 1e-3,
    tol2: float = 1e-6,
    c: float = 0.1,
    rk_steps: Literal[1, 2, 4] = 2,
    callback: Callable = None,
    callback_args: list = None,
) -> compas_dr.numdata.ResultData:
    """Implementation of the dynamic relaxation method for form finding and analysis
    of articulated networks of axial-force members.

    Parameters
    ----------
    indata : :class:`compas_dr.numdata.InputData`
        An input data object.
    constraints : list[:class:`~compas_dr.constraints.Constraint`]
        Vertex constraints.
    kmax : int, optional
        The maximum number of iterations.
    dt : float, optional
        The time step for the integration scheme.
    tol1 : float, optional
        Tolerance for the sum of the length of all residual force vectors.
    tol2 : float, optional
        Tolerance for the sum of the length of all displacement vectors.
    c : float, optional
        Value used to calculate coefficients "a" and "b", with
        "a" used as a multiplication factor for the starting velocity for the RK integration at every iteration, and
        "b" used as a multiplication factor for the acceleration used during RK integration.
    rk_steps : {1, 2, 4}, optional
        The number of Runge Kutta integration steps.
    callback : callable, optional
        User-defined function that is called at every iteration.
        If provided, the callback will be called at every iteration with the following arguments

        * `k`: the number of the current iteration
        * `x`: the current vertex coordinates
        * `crit1`: the norm of the residual forces
        * `crit2`: the norm of the displacement vectors
        * `callback_args`: optional additional arguments

    callback_args : tuple, optional
        Additional arguments passed to the callback.

    Returns
    -------
    :class:`compas_dr.numdata.ResultData`
        A result data object.

    Raises
    ------
    ValueError
        If a callback function is provided that is not callable.

    Notes
    -----
    For more info, see [1]_.

    References
    ----------
    .. [1] De Laet L., Veenendaal D., Van Mele T., Mollaert M. and Block P.,
           *Bending incorporated: designing tension structures by integrating bending-active elements*,
           Proceedings of Tensinet Symposium 2013,Istanbul, Turkey, 2013.

    Examples
    --------
    >>>

    """
    # --------------------------------------------------------------------------
    # callback
    # --------------------------------------------------------------------------

    if callback:
        if not callable(callback):
            raise ValueError("The provided callback is not callable.")

    # --------------------------------------------------------------------------
    # configuration
    # --------------------------------------------------------------------------

    coeff = Coeff(c)
    ca = coeff.a
    cb = coeff.b

    # --------------------------------------------------------------------------
    # numdata
    # --------------------------------------------------------------------------

    x = indata.vertices  # m
    p = indata.loads  # kN
    free = indata.free
    qpre = indata.qpre
    lpre = indata.lpre  # kN
    fpre = indata.fpre  # m
    linit = indata.linit  # m
    E = indata.E  # kN/mm2 => GPa
    radius = indata.radius  # mm

    C = indata.C  # type: scipy.sparse.csr_matrix
    Ct = C.transpose()
    Ci = C[:, free]
    Cit = Ci.transpose()
    Ct2 = Ct.copy()
    Ct2.data **= 2

    A = 3.14159 * radius**2  # mm2
    EA = E * A  # kN

    # --------------------------------------------------------------------------
    # initial values
    # --------------------------------------------------------------------------
    # if none of the initial lengths are set,
    # set the initial lengths to the current lengths
    # --------------------------------------------------------------------------

    q = indata.q0
    l = indata.l0  # noqa: E741
    f = q * l
    v = indata.v0
    r = indata.r0

    if all(linit == 0):
        linit = indata.l0

    # --------------------------------------------------------------------------
    # helpers
    # --------------------------------------------------------------------------

    def rk(x0, v0, steps=2):
        def acceleration(t, v):
            dx = v * t
            x[free] = x0[free] + dx[free]
            r[free] = p[free] - D.dot(x)
            return cb * r / mass

        if steps == 1:
            return acceleration(dt, v0)

        if steps == 2:
            B = [0.0, 1.0]
            K0 = dt * acceleration(K[0][0] * dt, v0)
            K1 = dt * acceleration(K[1][0] * dt, v0 + K[1][1] * K0)
            dv = B[0] * K0 + B[1] * K1
            return dv

        if steps == 4:
            B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
            K0 = dt * acceleration(K[0][0] * dt, v0)
            K1 = dt * acceleration(K[1][0] * dt, v0 + K[1][1] * K0)
            K2 = dt * acceleration(K[2][0] * dt, v0 + K[2][1] * K0 + K[2][2] * K1)
            K3 = dt * acceleration(K[3][0] * dt, v0 + K[3][1] * K0 + K[3][2] * K1 + K[3][3] * K2)
            dv = B[0] * K0 + B[1] * K1 + B[2] * K2 + B[3] * K3
            return dv

        raise NotImplementedError

    # --------------------------------------------------------------------------
    # start iterating
    # --------------------------------------------------------------------------

    for k in range(kmax):
        q_fpre = fpre / l
        q_lpre = f / lpre
        q_EA = EA * (l - linit) / (linit * l)
        q_lpre[isinf(q_lpre)] = 0
        q_lpre[isnan(q_lpre)] = 0
        q_EA[isinf(q_EA)] = 0
        q_EA[isnan(q_EA)] = 0

        q = qpre + q_fpre + q_lpre + q_EA
        Q = diags([q[:, 0]], [0])
        D = Cit.dot(Q).dot(C)
        mass = 0.5 * dt**2 * Ct2.dot(qpre + q_fpre + q_lpre + EA / linit)

        # RK

        x0 = x.copy()
        v0 = ca * v.copy()
        dv = rk(x0, v0, steps=rk_steps)
        v[free] = v0[free] + dv[free]
        dx = v * dt
        x[free] = x0[free] + dx[free]

        # update

        u = C.dot(x)
        l = normrow(u)  # noqa: E741
        f = q * l
        r = p - Ct.dot(Q).dot(u)

        # update constraints

        for vertex, constraint in enumerate(constraints):
            if not constraint:
                continue
            constraint.location = x[vertex]
            constraint.residual = r[vertex]
            constraint.update(damping=c)
            x[vertex] = constraint.location
            r[vertex] = constraint.residual

        # crits

        crit1 = norm(r[free])
        crit2 = norm(dx[free])

        # callback

        if callback:
            callback(k, x, crit1, crit2, callback_args)

        # convergence

        if crit1 < tol1:
            break
        if crit2 < tol2:
            break

    # --------------------------------------------------------------------------
    # result
    # --------------------------------------------------------------------------

    return ResultData(xyz=x, q=q, forces=f, lengths=l, residuals=r)
