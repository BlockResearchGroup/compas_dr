import numpy as np
from compas.datastructures import Mesh

from compas_dr.loads import SelfweightCalculator
from compas_dr.numdata import InputData


def test_input_data_builds_numeric_arrays_in_input_order():
    data = InputData(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 2, 0]],
        edges=[(0, 1), (1, 2)],
        fixed=[0, 2],
        loads=[[0, 0, 0], [0, 0, -1], [0, 0, 0]],
        qpre=[1, 2],
    )
    restored = InputData.from_jsonstring(data.to_jsonstring(minimal=True))

    assert restored.free == [1]
    assert restored.vertices.dtype == np.float64
    assert restored.edges.dtype == np.int32
    assert restored.C.format == "csr"
    matrix = restored.C.toarray()  # pyright: ignore[reportUnknownMemberType]
    np.testing.assert_array_equal(matrix, [[-1, 1, 0], [0, -1, 1]])
    np.testing.assert_array_equal(restored.loads, [[0, 0, 0], [0, 0, -1], [0, 0, 0]])
    np.testing.assert_array_equal(restored.qpre, [[1], [2]])
    np.testing.assert_array_equal(restored.l0, [[1], [2]])


def test_selfweight_of_unit_square():
    mesh = Mesh.from_meshgrid(dx=1, nx=1)
    mesh.update_default_vertex_attributes(t=0.2)
    calculator = SelfweightCalculator(mesh, density=10)

    # A quarter of the unit area, times thickness and density, at each corner.
    np.testing.assert_allclose(calculator(mesh.vertices_attributes("xyz")), [[0.5]] * 4, rtol=0, atol=1e-12)
