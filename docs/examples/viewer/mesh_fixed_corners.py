"""View a mesh with fixed corners."""

from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Vector
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_dr.numdata import InputData
from compas_dr.solvers import dr_numpy

# ==============================================================================
# Parameters
# ==============================================================================

size = 10.0
subdivisions = 10
boundary_force_density = 10.0
interior_force_density = 1.0

# ==============================================================================
# Mesh
# ==============================================================================

midspan = size / 2
mesh = Mesh.from_meshgrid(dx=size, nx=subdivisions)
fixed = list(mesh.vertices_where(vertex_degree=2))
q = [boundary_force_density if mesh.is_edge_on_boundary(edge) else interior_force_density for edge in mesh.edges()]

# ==============================================================================
# Solve
# ==============================================================================

loads = [[0.0, 0.0, 0.0] for _ in mesh.vertices()]
data = InputData.from_mesh(
    mesh,
    fixed,
    loads,
    q,
)
result = dr_numpy(data)
result.update_mesh(mesh)

# ==============================================================================
# Viewer
# ==============================================================================

config = Config()
config.window.title = "Dynamic relaxation | Mesh with fixed corners"
config.camera.target = [midspan, midspan, 0]
config.camera.position = [-6, -9, 17]
viewer = Viewer(config=config)
viewer.scene.add(
    mesh,
    name="Equilibrium mesh",
    show_points=False,
    show_lines=True,
    linecolor=Color(0.2, 0.26, 0.34),
    hide_coplanaredges=False,
)

for vertex in fixed:
    x, y, z = mesh.vertex_coordinates(vertex)
    point = Point(x, y, z)
    viewer.scene.add(
        point,
        name=f"Support {vertex}",
        pointcolor=Color.red(),
        pointsize=10,
    )

    rx, ry, rz = result.residuals[vertex]
    reaction = Vector(rx, ry, rz) * -0.1
    if reaction.length > 1e-6:
        viewer.scene.add(
            Line.from_point_and_vector(point, reaction),
            name=f"Reaction {vertex}",
            linecolor=Color.green(),
            linewidth=2,
        )

viewer.show()
