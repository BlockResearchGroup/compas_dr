import numpy as np

from compas_dr.numdata import InputData
from compas_dr.solvers import dr_numpy

data = InputData(
    vertices=[[1, 0, 0], [0, 0, 0], [4, 0, 0]],
    edges=[(0, 1), (0, 2)],
    fixed=[1, 2],
    loads=[[0, 0, -4], [0, 0, 0], [0, 0, 0]],
    qpre=[1, 3],
)
with np.errstate(divide="ignore", invalid="ignore"):
    result = dr_numpy(data, tol1=1e-8, tol2=0)

np.testing.assert_allclose(result.xyz[0], [3, 0, -1], rtol=0, atol=1e-8)
print(result.xyz[0])
