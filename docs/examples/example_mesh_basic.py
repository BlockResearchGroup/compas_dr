from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Sphere
from compas.geometry import Vector
from compas_dr.numdata import InputData
from compas_dr.solvers import dr_numpy
from compas_view2.app import App

# =============================================================================
# Input
# =============================================================================

mesh = Mesh.from_meshgrid(dx=10, nx=10)

fixed = list(mesh.vertices_where(vertex_degree=2))
loads = [[0, 0, 0] for _ in range(mesh.number_of_vertices())]
qpre = [1.0] * mesh.number_of_edges()

for index, edge in enumerate(mesh.edges()):
    if mesh.is_edge_on_boundary(edge):
        qpre[index] = 10

indata = InputData.from_mesh(mesh, fixed, loads, qpre)

# =============================================================================
# Solve and Update
# =============================================================================

result = dr_numpy(indata=indata)

result.update_mesh(mesh)

# =============================================================================
# Visualize
# =============================================================================

forcecolor = Color.green().darkened(50)

viewer = App()
viewer.view.camera.position = [5, -5, 20]
viewer.view.camera.look_at([5, 5, 0])

viewer.add(mesh)

for vertex in fixed:
    point = Point(*mesh.vertex_coordinates(vertex))
    residual = Vector(*result.residuals[vertex]) * 0.1

    ball = Sphere(radius=0.1, point=point).to_brep()
    line = Line(point, point - residual)

    viewer.add(ball, facecolor=Color.red())
    viewer.add(line, linecolor=forcecolor, linewidth=3)

viewer.run()
