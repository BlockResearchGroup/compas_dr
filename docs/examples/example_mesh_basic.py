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

qpre = []
for edge in mesh.edges():
    if mesh.is_edge_on_boundary(edge):
        qpre.append(10)
    else:
        qpre.append(1.0)

indata = InputData.from_mesh(mesh, fixed, loads, qpre)

# =============================================================================
# Solve
# =============================================================================

result = dr_numpy(indata=indata)

# =============================================================================
# Update
# =============================================================================

for vertex in mesh.vertices():
    mesh.vertex_attributes(vertex, "xyz", result.xyz[vertex])

# =============================================================================
# Visualize
# =============================================================================

viewer = App()
viewer.view.camera.position = [5, -5, 20]
viewer.view.camera.look_at([5, 5, 0])

viewer.add(mesh)

for vertex in fixed:
    point = Point(*mesh.vertex_coordinates(vertex))
    residual = Vector(*result.residuals[vertex])
    ball = Sphere(radius=0.1, point=point)

    viewer.add(ball.to_brep(), facecolor=Color.red())
    viewer.add(
        Line(point, point - residual * 0.1),
        linecolor=Color.green().darkened(50),
        linewidth=3,
    )

viewer.run()
