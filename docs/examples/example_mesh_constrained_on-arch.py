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
from compas_viewer import Viewer

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
# Constraints
# =============================================================================

arch = NurbsCurve.from_points([[5, 0, 0], [5, 5, 5], [5, 10, 0]])
constraint = Constraint(arch)

constraints = [None] * mesh.number_of_vertices()
for vertex in mesh.vertices_where(x=5):
    if vertex in fixed:
        continue
    constraints[vertex] = constraint
    fixed.append(vertex)

# =============================================================================
# Solve and Update
# =============================================================================

result = dr_constrained_numpy(indata=inputdata, constraints=constraints)

result.update_mesh(mesh)

# =============================================================================
# Visualization
# =============================================================================

forcecolor = Color.green().darkened(50)

viewer = Viewer()
viewer.renderer.camera.target = [5, 5, 2]
viewer.renderer.camera.position = [-7, -10, 5]

viewer.scene.add(mesh, show_points=False)
viewer.scene.add(arch.to_polyline(), linecolor=Color.cyan(), lineswidth=3, show_points=False)

for vertex in fixed:
    point = Point(*mesh.vertex_coordinates(vertex))
    residual = Vector(*result.residuals[vertex]) * 0.5

    ball = Sphere(radius=0.1, point=point).to_brep()
    line = Line(point, point - residual)
    ballcolor = Color.blue() if constraints[vertex] else Color.red()

    viewer.scene.add(ball, surfacecolor=ballcolor, show_points=False)
    viewer.scene.add(line, linecolor=forcecolor, lineswidth=3, show_points=False)

viewer.show()
