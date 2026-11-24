"""View a prestressed, double-layer spoked wheel with sliding rim supports."""

from compas.colors import Color
from compas.geometry import Cylinder
from compas.geometry import Ellipse
from compas.geometry import Line
from compas.geometry import NurbsCurve
from compas.geometry import Point
from compas.geometry import Vector
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_dr.constraints import Constraint
from compas_dr.numdata import InputData
from compas_dr.solvers import dr_constrained_numpy

# ==============================================================================
# Parameters
# ==============================================================================

rim_major, rim_minor = 20.0, 15.0
hub_major, hub_minor = 12.0, 9.0
hub_offset = 1.0
panels = 45
fixed_rim_nodes = (11, 33)
rim_force_density = 0.0
hub_force_density = 40.0
spoke_force_density = 1.5
strut_prestress = -2.0
load = Vector(0, 0, -1)
iteration_limit = 100

# ==============================================================================
# Geometry
# ==============================================================================

rim = NurbsCurve.from_ellipse(Ellipse(major=rim_major, minor=rim_minor))
hub = NurbsCurve.from_ellipse(Ellipse(major=hub_major, minor=hub_minor))
# The last point of each sampled ring repeats its first point.
rim_points = rim.to_polyline(panels).points[:-1]
hub_points = hub.to_polyline(panels).points[:-1]
count = len(rim_points)
bottom = count
top = 2 * count
points = rim_points + [point + Vector(0, 0, -hub_offset) for point in hub_points] + [point + Vector(0, 0, hub_offset) for point in hub_points]

# ==============================================================================
# Members and supports
# ==============================================================================

# Each member stores its node indices, prescribed force density and axial prestress.
members: list[tuple[tuple[int, int], float, float]] = []
for node in range(count):
    next_node = (node + 1) % count
    members.extend(
        [
            ((node, next_node), rim_force_density, 0),
            ((bottom + node, bottom + next_node), hub_force_density, 0),
            ((top + node, top + next_node), hub_force_density, 0),
            ((node, bottom + node), spoke_force_density, 0),
            ((node, top + node), spoke_force_density, 0),
            ((bottom + node, top + node), 0, strut_prestress),
        ]
    )

supports = list(range(count))
constraints: list[Constraint[NurbsCurve] | None] = [None] * len(points)
for node in supports:
    if node not in fixed_rim_nodes:
        constraints[node] = Constraint(rim)

# ==============================================================================
# Solve
# ==============================================================================

data = InputData(
    vertices=[[point.x, point.y, point.z] for point in points],
    edges=[edge for edge, _, _ in members],
    fixed=supports,
    loads=[[0, 0, load.z if bottom <= node < top else 0] for node in range(len(points))],
    qpre=[q for _, q, _ in members],
    fpre=[f for _, _, f in members],
)
# Show the relaxation after up to 100 iterations, as in the original example.
result = dr_constrained_numpy(
    indata=data,
    constraints=constraints,
    kmax=iteration_limit,
    tol1=1e-6,
    tol2=1e-9,
    rk_steps=4,
)

# ==============================================================================
# Viewer
# ==============================================================================

deformed = [Point(*xyz) for xyz in result.xyz]
config = Config()
config.window.title = "Dynamic relaxation | Spoked wheel"
config.camera.position = [25, -50, 40]
config.renderer.show_grid = False
viewer = Viewer(config=config)

tension = viewer.scene.add_group("Tension")
compression = viewer.scene.add_group("Compression")
for index, ((u, v), _, _) in enumerate(members):
    force = float(result.forces[index, 0])
    if abs(force) <= 1e-6:
        continue
    color = Color.red() if force > 0 else Color.blue()
    scale = 0.002 if force > 0 else 0.02
    cylinder = Cylinder.from_line_and_radius(Line(deformed[u], deformed[v]), abs(force) * scale)
    viewer.scene.add(
        cylinder,
        parent=tension if force > 0 else compression,
        facecolor=color,
        show_lines=False,
    )

support_group = viewer.scene.add_group("Rim supports")
reaction_group = viewer.scene.add_group("Support reactions")
load_group = viewer.scene.add_group("Applied loads")
for node in supports:
    point = deformed[node]
    color = Color.cyan() if constraints[node] is not None else Color.yellow()
    viewer.scene.add(
        point,
        parent=support_group,
        pointcolor=color,
        pointsize=7,
    )
    reaction = Vector(*result.residuals[node]) * -0.3
    if reaction.length > 1e-6:
        viewer.scene.add(
            Line.from_point_and_vector(point, reaction),
            parent=reaction_group,
            linecolor=Color.green(),
            linewidth=2,
        )

for point in deformed[bottom:top]:
    viewer.scene.add(
        Line.from_point_and_vector(point, load),
        parent=load_group,
        linecolor=Color.green().darkened(50),
        linewidth=2,
    )

viewer.scene.add(
    rim.to_polyline(89),
    name="Sliding support path",
    linecolor=Color.cyan(),
    linewidth=2,
)
viewer.show()
