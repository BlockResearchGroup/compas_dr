import math

import numpy as np
import pytest
import scipy.sparse
from compas.datastructures import Mesh

import compas_dr.numdata
from compas_dr.numdata import InputData

VERTICES = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]
EDGES = [(0, 1), (1, 2)]
FIXED = [0]
LOADS = [[0.0, 0.0, -1.0]] * 3
QPRE = [1.0, 2.0]

N = len(VERTICES)
M = len(EDGES)

OPTIONAL = {
    "fpre": [1.0, 2.0],
    "lpre": [3.0, 4.0],
    "linit": [5.0, 6.0],
    "E": [7.0, 8.0],
    "radius": [9.0, 10.0],
}

SHAPES = {
    "vertices": ((N, 3), np.float64),
    "edges": ((M, 2), np.int32),
    "loads": ((N, 3), np.float64),
    "qpre": ((M, 1), np.float64),
    "fpre": ((M, 1), np.float64),
    "lpre": ((M, 1), np.float64),
    "linit": ((M, 1), np.float64),
    "E": ((M, 1), np.float64),
    "radius": ((M, 1), np.float64),
    "q0": ((M, 1), np.float64),
    "l0": ((M, 1), np.float64),
    "v0": ((N, 3), np.float64),
    "r0": ((N, 3), np.float64),
}


def test_no_numpy_fallback():
    assert not hasattr(compas_dr.numdata, "has_numpy")


@pytest.mark.parametrize("optional", [{}, OPTIONAL], ids=["optional omitted", "optional given"])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_property_is_array(name, optional):
    indata = InputData(VERTICES, EDGES, FIXED, LOADS, QPRE, **optional)
    shape, dtype = SHAPES[name]
    value = getattr(indata, name)

    assert isinstance(value, np.ndarray)
    assert value.shape == shape
    assert value.dtype == dtype


def test_values():
    indata = InputData(VERTICES, EDGES, FIXED, LOADS, QPRE)

    assert indata.vertices.tolist() == VERTICES
    assert indata.edges.tolist() == [[0, 1], [1, 2]]
    assert indata.loads.tolist() == LOADS
    assert indata.qpre.tolist() == [[1.0], [2.0]]
    assert indata.fixed == [0]
    assert indata.free == [1, 2]
    assert indata.q0.tolist() == [[1.0], [1.0]]
    assert indata.l0.tolist() == [[5.0], [12.0]]
    assert indata.v0.tolist() == [[0.0, 0.0, 0.0]] * N
    assert indata.r0.tolist() == [[0.0, 0.0, 0.0]] * N


def test_optional_default_to_zero():
    indata = InputData(VERTICES, EDGES, FIXED, LOADS, QPRE)

    for name in OPTIONAL:
        assert getattr(indata, name).tolist() == [[0.0]] * M


def test_optional_given():
    indata = InputData(VERTICES, EDGES, FIXED, LOADS, QPRE, **OPTIONAL)

    for name, values in OPTIONAL.items():
        assert getattr(indata, name).tolist() == [[value] for value in values]


def test_connectivity_matrix_is_sparse_csr():
    indata = InputData(VERTICES, EDGES, FIXED, LOADS, QPRE)
    C = indata.C

    assert scipy.sparse.issparse(C)
    assert C.format == "csr"
    assert C.shape == (M, N)
    assert C.toarray().tolist() == [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]


def test_from_mesh():
    mesh = Mesh.from_meshgrid(dx=2, nx=2)
    fixed = list(mesh.vertices_where(vertex_degree=2))
    loads = [[0.0, 0.0, 0.0]] * mesh.number_of_vertices()
    qpre = [1.0] * mesh.number_of_edges()

    indata = InputData.from_mesh(mesh, fixed, loads, qpre)

    assert indata.vertices.shape == (9, 3)
    assert indata.vertices.tolist() == mesh.vertices_attributes("xyz")
    assert indata.edges.shape == (12, 2)
    assert indata.edges.tolist() == [list(edge) for edge in mesh.edges()]
    assert indata.C.shape == (12, 9)
    assert indata.l0.tolist() == [[pytest.approx(1.0, abs=1e-12)]] * 12
    assert sorted(indata.fixed + indata.free) == list(range(9))
    assert math.isclose(float(indata.l0.sum()), 12.0)
