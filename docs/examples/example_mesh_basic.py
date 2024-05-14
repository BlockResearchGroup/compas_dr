from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Sphere
from compas.geometry import Vector
from compas_dr.numdata import InputData
from compas_dr.solvers import dr_numpy
from compas_viewer import Viewer

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

viewer = Viewer()
viewer.renderer.camera.target = [5, 5, 0]
viewer.renderer.camera.position = [5, 4, 20]

viewer.scene.add(mesh, show_points=False)

for vertex in fixed:
    point = Point(*mesh.vertex_coordinates(vertex))
    residual = Vector(*result.residuals[vertex]) * 0.1

    ball = Sphere(radius=0.1, point=point).to_brep()
    line = Line(point, point - residual)

    viewer.scene.add(ball, surfacecolor=Color.red())
    viewer.scene.add(line, linecolor=forcecolor, lineswidth=3, show_points=False)

viewer.show()
