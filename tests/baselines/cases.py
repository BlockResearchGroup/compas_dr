"""Baseline cases for the numerical routines of ``compas_dr``.

This module is the single definition of the inputs and calls of the baseline
cases. It is used by ``record.py`` (COMPAS 2, legacy revision) to write the
fixtures in this directory and by ``tests/test_baselines.py`` to re-run the
cases and compare against them. It only uses API that exists unchanged in
COMPAS 2 and at the COMPAS 3 target.

Every case is a function without arguments that returns ``(inputs, outputs)``:
two dictionaries of NumPy arrays. Inputs are compared exactly, outputs within
the tolerance of the baseline policy. A case builds all of its input from
scratch on every call, because the solvers mutate their input.

Shared input "grid": ``Mesh.from_meshgrid(dx=10, nx=10)`` (121 vertices, 220
edges), the four corners fixed, force density 10 on boundary edges and 1
inside, load (0, 0, -0.5) per vertex. "Column" is the eleven vertices at
x = 5. In the constrained cases the column is fixed and every column vertex
has its own constraint object. There are no random numbers.

"""

import contextlib
import io

import numpy as np

LOAD = (0.0, 0.0, -0.5)
RHINO_GUID = "9d2b4c1e-6f3a-4b7d-8e21-5c0a7f3b1d42"

CASES = {}
"""Case identifier -> case function, in the order of definition."""

REQUIRES = {}
"""Case identifier -> name of the optional backend package the case needs."""


def case(identifier, requires=None):
    def decorator(function):
        CASES[identifier] = function
        if requires:
            REQUIRES[identifier] = requires
        return function

    return decorator


# =============================================================================
# Shared input
# =============================================================================


def floats(value):
    return np.array(value, dtype=np.float64)


def ints(value):
    return np.array(value, dtype=np.int64)


class Grid:
    """The shared grid as the plain lists the ``compas_dr`` functions take."""

    def __init__(self):
        from compas.datastructures import Mesh

        mesh = Mesh.from_meshgrid(dx=10, nx=10)
        vertex_index = {vertex: index for index, vertex in enumerate(mesh.vertices())}

        self.mesh = mesh
        self.vertices = mesh.vertices_attributes("xyz")
        self.edges = [(vertex_index[u], vertex_index[v]) for u, v in mesh.edges()]
        self.corners = [vertex_index[vertex] for vertex in mesh.vertices_where(vertex_degree=2)]
        self.column = [vertex_index[vertex] for vertex in mesh.vertices_where(x=5)]
        self.fixed = list(self.corners)
        self.loads = [list(LOAD) for _ in mesh.vertices()]
        self.qpre = [10.0 if mesh.is_edge_on_boundary(edge) else 1.0 for edge in mesh.edges()]

    def fix_column(self):
        for index in self.column:
            if index not in self.fixed:
                self.fixed.append(index)

    def inputs(self):
        return {
            "vertices": floats(self.vertices),
            "edges": ints(self.edges),
            "fixed": ints(self.fixed),
            "column": ints(self.column),
            "loads": floats(self.loads),
            "qpre": floats(self.qpre),
        }


class Trace:
    """Callback of the NumPy solvers that collects the criteria per iteration."""

    def __init__(self):
        self.crit1 = []
        self.crit2 = []

    def __call__(self, k, x, crit1, crit2, callback_args):
        self.crit1.append(float(crit1))
        self.crit2.append(float(crit2))

    def outputs(self):
        return {
            "crit1": floats(self.crit1),
            "crit2": floats(self.crit2),
            "iterations": ints(len(self.crit1)),
        }


def result_outputs(result):
    return {
        "xyz": floats(result.xyz),
        "q": floats(result.q),
        "forces": floats(result.forces),
        "lengths": floats(result.lengths),
        "residuals": floats(result.residuals),
    }


def options_inputs(options):
    inputs = {}
    for name, value in options.items():
        inputs[name] = ints(value) if isinstance(value, int) else floats(value)
    return inputs


# =============================================================================
# InputData
# =============================================================================


def inputdata_outputs(indata):
    return {
        "vertices": np.array(indata.vertices),
        "edges": np.array(indata.edges),
        "fixed": ints(indata.fixed),
        "free": ints(indata.free),
        "loads": np.array(indata.loads),
        "qpre": np.array(indata.qpre),
        "fpre": np.array(indata.fpre),
        "lpre": np.array(indata.lpre),
        "linit": np.array(indata.linit),
        "E": np.array(indata.E),
        "radius": np.array(indata.radius),
        "q0": np.array(indata.q0),
        "l0": np.array(indata.l0),
        "v0": np.array(indata.v0),
        "r0": np.array(indata.r0),
        "C": np.array(indata.C.toarray()),
    }


@case("dr-01")
def inputdata_from_mesh():
    """``InputData.from_mesh`` on the loaded grid, optional arguments omitted."""
    from compas_dr.numdata import InputData

    grid = Grid()
    inputs = grid.inputs()
    indata = InputData.from_mesh(grid.mesh, grid.fixed, grid.loads, grid.qpre)
    return inputs, inputdata_outputs(indata)


@case("dr-02")
def inputdata_with_optional_arguments():
    """``InputData`` on the loaded grid with all optional arguments given."""
    from compas_dr.numdata import InputData

    grid = Grid()
    count = len(grid.edges)
    optional = {
        "fpre": [0.5 + 0.25 * (index % 4) for index in range(count)],
        "lpre": [1.0 + 0.125 * (index % 3) for index in range(count)],
        "linit": [0.75 + 0.0625 * (index % 5) for index in range(count)],
        "E": [210.0 - 10.0 * (index % 2) for index in range(count)],
        "radius": [5.0 + 0.5 * (index % 6) for index in range(count)],
    }
    inputs = grid.inputs()
    inputs.update({name: floats(value) for name, value in optional.items()})
    indata = InputData(grid.vertices, grid.edges, grid.fixed, grid.loads, grid.qpre, **optional)
    return inputs, inputdata_outputs(indata)


# =============================================================================
# dr_numpy
# =============================================================================


def run_dr_numpy(options, **optional):
    from compas_dr.numdata import InputData
    from compas_dr.solvers import dr_numpy

    grid = Grid()
    inputs = grid.inputs()
    inputs.update({name: floats(value) for name, value in optional.items()})
    inputs.update(options_inputs(options))

    indata = InputData.from_mesh(grid.mesh, grid.fixed, grid.loads, grid.qpre, **optional)
    trace = Trace()
    result = dr_numpy(indata, callback=trace, **options)

    outputs = result_outputs(result)
    outputs.update(trace.outputs())
    return inputs, outputs


@case("dr-03")
def dr_numpy_rk2_fixed_iterations():
    """``dr_numpy``, ``rk_steps=2``, exactly 20 iterations."""
    return run_dr_numpy({"rk_steps": 2, "kmax": 20, "tol1": 0.0, "tol2": 0.0})


@case("dr-04")
def dr_numpy_rk4_fixed_iterations():
    """``dr_numpy``, ``rk_steps=4``, exactly 20 iterations."""
    return run_dr_numpy({"rk_steps": 4, "kmax": 20, "tol1": 0.0, "tol2": 0.0})


@case("dr-05")
def dr_numpy_converged():
    """``dr_numpy`` with its defaults, run to convergence."""
    return run_dr_numpy({})


@case("dr-06")
def dr_numpy_elastic():
    """``dr_numpy`` with ``E=210``, ``radius=5``, ``linit=0.95`` per edge, exactly 20 iterations."""
    count = len(Grid().edges)
    return run_dr_numpy(
        {"kmax": 20, "tol1": 0.0, "tol2": 0.0},
        E=[210.0] * count,
        radius=[5.0] * count,
        linit=[0.95] * count,
    )


# =============================================================================
# dr_constrained_numpy
# =============================================================================


def plane_geometry():
    from compas.geometry import Plane

    return Plane([5, 0, 0], [1, 0, 0])


def line_geometry():
    from compas.geometry import Line

    return Line([5, -1, 0], [5, 11, 0])


def curve_geometry():
    from compas.geometry import NurbsCurve

    return NurbsCurve.from_points([[5, 0, 0], [5, 5, 3], [5, 10, 0]])


def run_dr_constrained_numpy(geometry, options, params=False):
    from compas_dr.constraints import Constraint
    from compas_dr.numdata import InputData
    from compas_dr.solvers import dr_constrained_numpy

    grid = Grid()
    grid.fix_column()
    inputs = grid.inputs()
    inputs.update(options_inputs(options))

    constraints = [None] * len(grid.vertices)
    for index in grid.column:
        constraints[index] = Constraint(geometry())

    indata = InputData.from_mesh(grid.mesh, grid.fixed, grid.loads, grid.qpre)
    trace = Trace()
    # The solver prints the iteration number at every iteration.
    with contextlib.redirect_stdout(io.StringIO()):
        result = dr_constrained_numpy(indata=indata, constraints=constraints, callback=trace, **options)

    outputs = result_outputs(result)
    outputs.update(trace.outputs())
    outputs["locations"] = floats([list(constraints[index].location) for index in grid.column])
    if params:
        outputs["params"] = floats([constraints[index].param for index in grid.column])
    return inputs, outputs


@case("dr-07")
def dr_constrained_numpy_plane_fixed_iterations():
    """``dr_constrained_numpy``, column on a plane, exactly 20 iterations."""
    return run_dr_constrained_numpy(plane_geometry, {"kmax": 20, "tol1": 0.0, "tol2": 0.0})


@case("dr-08")
def dr_constrained_numpy_line_fixed_iterations():
    """``dr_constrained_numpy``, column on a line, exactly 20 iterations."""
    return run_dr_constrained_numpy(line_geometry, {"kmax": 20, "tol1": 0.0, "tol2": 0.0})


@case("dr-09")
def dr_constrained_numpy_plane_converged():
    """``dr_constrained_numpy``, column on a plane, defaults, run to convergence."""
    return run_dr_constrained_numpy(plane_geometry, {})


@case("dr-13", requires="compas_occ")
def dr_constrained_numpy_curve_fixed_iterations():
    """``dr_constrained_numpy``, column on a NURBS curve, exactly 20 iterations."""
    return run_dr_constrained_numpy(curve_geometry, {"kmax": 20, "tol1": 0.0, "tol2": 0.0}, params=True)


# =============================================================================
# dr
# =============================================================================


@case("dr-10")
def dr_defaults():
    """Pure-Python ``dr`` on the loaded grid with its defaults."""
    from compas_dr.solvers import dr

    grid = Grid()
    inputs = grid.inputs()
    X, Q, F, L, R = dr(grid.vertices, grid.edges, grid.fixed, grid.loads, grid.qpre)
    outputs = {"X": floats(X), "Q": floats(Q), "F": floats(F), "L": floats(L), "R": floats(R)}
    return inputs, outputs


# =============================================================================
# SelfweightCalculator
# =============================================================================


@case("dr-11")
def selfweight_on_converged_geometry():
    """``SelfweightCalculator`` (t = 0.1, density 22) on the geometry of dr-05."""
    from compas_dr.loads import SelfweightCalculator
    from compas_dr.numdata import InputData
    from compas_dr.solvers import dr_numpy

    grid = Grid()
    inputs = grid.inputs()
    inputs["thickness"] = floats(0.1)
    inputs["density"] = floats(22.0)

    indata = InputData.from_mesh(grid.mesh, grid.fixed, grid.loads, grid.qpre)
    xyz = floats(dr_numpy(indata).xyz)

    grid.mesh.update_default_vertex_attributes(t=0.1)
    calculator = SelfweightCalculator(grid.mesh, density=22.0)
    selfweight = calculator(xyz)

    outputs = {"xyz": xyz, "F": floats(calculator.F.toarray()), "selfweight": floats(selfweight)}
    return inputs, outputs


# =============================================================================
# Constraint data (dr-12)
# =============================================================================

CONSTRAINTS_CASE = "dr-12"


def constraints():
    """A line constraint and a plane constraint; the plane constraint has a ``rhino_guid``."""
    from compas_dr.constraints import Constraint

    line = Constraint(line_geometry())
    plane = Constraint(plane_geometry())
    plane._rhino_guid = RHINO_GUID
    return [line, plane]
