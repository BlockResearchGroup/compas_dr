from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Sphere
from compas.geometry import Vector
from compas_dr.dr_numpy import dr_numpy
from compas_dr.numdata import InputData
from compas_view2.app import App

# =============================================================================
# Input
# =============================================================================

mesh = Mesh.from_meshgrid(dx=10, nx=10)

fixed = list(mesh.vertices_where(vertex_degree=2))
loads = [[0, 0, 0]] * mesh.number_of_vertices()
qpre = [1.0] * mesh.number_of_edges()

for index, edge in enumerate(mesh.edges()):
    if mesh.is_edge_on_boundary(edge):
        qpre[index] = 10

inputdata = InputData.from_mesh(mesh, fixed, loads, qpre)

# =============================================================================
# Solve
# =============================================================================

result = dr_numpy(inputdata, kmax=1000, rk_steps=2)

# =============================================================================
# Update
# =============================================================================

for vertex in mesh.vertices():
    mesh.vertex_attributes(vertex, "xyz", result.xyz[vertex])

# =============================================================================
# Visualization
# =============================================================================

viewer = App()
viewer.add(mesh)

for vertex in mesh.vertices():
    point = Point(*mesh.vertex_coordinates(vertex))
    residual = Vector(*result.residuals[vertex])

    if vertex in fixed:
        ball = Sphere(radius=0.1, point=point)
        viewer.add(ball.to_brep(), facecolor=Color.red())

        viewer.add(
            Line(point, point - residual * 0.1),
            linecolor=Color.green().darkened(50),
            linewidth=3,
        )

viewer.view.camera.zoom_extents()
viewer.run()
