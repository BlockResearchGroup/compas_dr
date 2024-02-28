from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import NurbsCurve
from compas.geometry import Point
from compas.geometry import Sphere
from compas.geometry import Vector
from compas_dr.constraints import Constraint
from compas_dr.numdata import InputData
from compas_dr.solvers import dr_constrained_numpy
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

qpre = []
for edge in mesh.edges():
    if mesh.is_edge_on_boundary(edge):
        qpre.append(10)
    else:
        qpre.append(1.0)

inputdata = InputData.from_mesh(mesh, fixed, loads, qpre)

# constraints

arch = NurbsCurve.from_points([[5, 0, 0], [5, 5, 5], [5, 10, 0]])
constraint = Constraint(arch)

constraints = [None] * mesh.number_of_vertices()
for vertex in mesh.vertices_where(x=5):
    constraints[vertex] = constraint
    fixed.append(vertex)

# =============================================================================
# Solve and Update
# =============================================================================

result = dr_constrained_numpy(indata=inputdata, constraints=constraints)

for vertex in mesh.vertices():
    mesh.vertex_attributes(vertex, "xyz", result.xyz[vertex])

# =============================================================================
# Visualization
# =============================================================================

viewer = App()
viewer.view.camera.position = [-7, -10, 5]
viewer.view.camera.look_at([5, 5, 2])

viewer.add(mesh)

for vertex in mesh.vertices():
    point = Point(*mesh.vertex_coordinates(vertex))
    residual = Vector(*result.residuals[vertex])

    if vertex in fixed:
        ball = Sphere(radius=0.1, point=point)

        if constraints[vertex]:
            viewer.add(ball.to_brep(), facecolor=Color.blue())
        else:
            viewer.add(ball.to_brep(), facecolor=Color.red())

        viewer.add(
            Line(point, point - residual * 0.1),
            linecolor=Color.green().darkened(50),
            linewidth=3,
        )

viewer.add(arch.to_polyline(), linecolor=Color.cyan(), linewidth=3)

viewer.run()
