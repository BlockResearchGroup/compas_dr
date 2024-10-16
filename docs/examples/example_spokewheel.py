from compas.colors import Color
from compas.datastructures import Graph
from compas.geometry import Cylinder
from compas.geometry import Ellipse
from compas.geometry import Line
from compas.geometry import NurbsCurve
from compas.geometry import Vector
from compas.itertools import pairwise
from compas.tolerance import TOL
from compas_dr.constraints import Constraint
from compas_dr.numdata import InputData
from compas_dr.solvers import dr_constrained_numpy
from compas_viewer import Viewer
from compas_viewer.config import Config

OUTER = 0
INNER = 40
SPOKE = 1.5
STRUT = -2
LOAD = -1

SLIDE = True

# =============================================================================
# Geometry
# =============================================================================

outer: NurbsCurve = NurbsCurve.from_ellipse(Ellipse(major=20, minor=0.75 * 20))
inner: NurbsCurve = NurbsCurve.from_ellipse(Ellipse(major=12, minor=0.75 * 12))

outer_ring = outer.to_polyline(45)
top_ring = inner.translated([0, 0, 1]).to_polyline(45)
bottom_ring = inner.translated([0, 0, -1]).to_polyline(45)

# =============================================================================
# Graph
# =============================================================================

graph = Graph()

graph.update_default_edge_attributes(
    outer_ring=False,
    top_ring=False,
    bottom_ring=False,
    top_spoke=False,
    bottom_spoke=False,
    strut=False,
    force=0,
    qpre=0,
    fpre=0,
    lpre=0,
)
graph.update_default_node_attributes(
    outer_ring=False,
    top_ring=False,
    bottom_ring=False,
    panel=None,
    anchor=False,
    residual=None,
    load=None,
    constraint=None,
)

# =============================================================================
# Graph components
# =============================================================================

gkey_node = {}

for i, point in enumerate(outer_ring[:-1]):
    gkey = TOL.geometric_key(point)
    if gkey not in gkey_node:
        node = graph.add_node(x=point[0], y=point[1], z=point[2], outer_ring=True, panel=i)
        gkey_node[gkey] = node

for i, point in enumerate(bottom_ring[:-1]):
    gkey = TOL.geometric_key(point)
    if gkey not in gkey_node:
        node = graph.add_node(x=point[0], y=point[1], z=point[2], bottom_ring=True, panel=i)
        gkey_node[gkey] = node

for i, point in enumerate(top_ring[:-1]):
    gkey = TOL.geometric_key(point)
    if gkey not in gkey_node:
        node = graph.add_node(x=point[0], y=point[1], z=point[2], top_ring=True, panel=i)
        gkey_node[gkey] = node

for (a, b), (aa, bb) in zip(pairwise(outer_ring), pairwise(bottom_ring)):
    u = gkey_node[TOL.geometric_key(a)]
    v = gkey_node[TOL.geometric_key(b)]
    uu = gkey_node[TOL.geometric_key(aa)]
    vv = gkey_node[TOL.geometric_key(bb)]

    graph.add_edge(u, v, outer_ring=True)
    graph.add_edge(uu, vv, bottom_ring=True)
    graph.add_edge(u, uu, bottom_spoke=True)

for (a, b), (aa, bb) in zip(pairwise(outer_ring), pairwise(top_ring)):
    u = gkey_node[TOL.geometric_key(a)]
    v = gkey_node[TOL.geometric_key(b)]
    uu = gkey_node[TOL.geometric_key(aa)]
    vv = gkey_node[TOL.geometric_key(bb)]

    graph.add_edge(uu, vv, top_ring=True)
    graph.add_edge(u, uu, top_spoke=True)

for a, b in zip(bottom_ring[:-1], top_ring[:-1]):
    u = gkey_node[TOL.geometric_key(a)]
    v = gkey_node[TOL.geometric_key(b)]

    graph.add_edge(u, v, strut=True)

# =============================================================================
# Constraints
# =============================================================================

constraint = Constraint(outer)

graph.nodes_attribute(
    name="anchor",
    value=True,
    keys=graph.nodes_where(outer_ring=True),
)

graph.nodes_attribute(
    name="constraint",
    value=constraint,
    keys=graph.nodes_where(outer_ring=True),
)

# for node in [0, 11, 21, 22, 32]:
#     graph.unset_node_attribute(node, "constraint")

for node in [11, 33]:
    graph.unset_node_attribute(node, "constraint")

# =============================================================================
# Force densities
# =============================================================================

graph.edges_attribute(name="qpre", value=OUTER, keys=graph.edges_where(outer_ring=True))

graph.edges_attribute(name="qpre", value=INNER, keys=graph.edges_where(bottom_ring=True))
graph.edges_attribute(name="qpre", value=INNER, keys=graph.edges_where(top_ring=True))

graph.edges_attribute(name="qpre", value=SPOKE, keys=graph.edges_where(bottom_spoke=True))
graph.edges_attribute(name="qpre", value=SPOKE, keys=graph.edges_where(top_spoke=True))

graph.edges_attribute(name="fpre", value=STRUT, keys=graph.edges_where(strut=True))

graph.nodes_attribute(name="load", value=Vector(0, 0, LOAD), keys=graph.nodes_where(bottom_ring=True))

# =============================================================================
# FormFinding
# =============================================================================

points = graph.nodes_attributes(names="xyz")
edges = list(graph.edges())
fixed = list(graph.nodes_where(anchor=True))
qpre = list(graph.edges_attribute(name="qpre"))
fpre = list(graph.edges_attribute(name="fpre"))
loads = [graph.node_attribute(node, name="load") or Vector(0, 0, 0) for node in graph.nodes()]

if not SLIDE:
    constraints = [None] * graph.number_of_nodes()
else:
    constraints = graph.nodes_attribute(name="constraint")

indata = InputData(
    vertices=points,
    edges=edges,
    fixed=fixed,
    loads=loads,
    qpre=qpre,
    fpre=fpre,
)

result = dr_constrained_numpy(
    indata=indata,
    constraints=constraints,
    tol1=1e-6,
    tol2=1e-9,
    rk_steps=4,
    kmax=100,
    callback=lambda k, x, crit1, crit2, callback_args: print(k),
)

for node in graph.nodes():
    graph.node_attributes(node, "xyz", result.xyz[node])
    graph.node_attribute(node, "residual", Vector(*result.residuals[node]))

for index, edge in enumerate(graph.edges()):
    graph.edge_attribute(edge, "force", result.forces[index, 0])

# =============================================================================
# Pre-process visualisation
# =============================================================================

scale_tension = 0.002
scale_compression = 0.02

compression = []
tension = []

for edge in graph.edges():
    force = graph.edge_attribute(edge, name="force")

    if force > 0:
        radius = +force * scale_tension
        tension.append(Cylinder.from_line_and_radius(graph.edge_line(edge), radius))

    if force < 0:
        radius = -force * scale_compression
        compression.append(Cylinder.from_line_and_radius(graph.edge_line(edge), radius))

tol = 0.01

reactions = []
residuals = []
tangents = []

for node in graph.nodes():
    if graph.node_attribute(node, name="anchor"):
        location = graph.node_point(node)
        residual = graph.node_attribute(node, name="residual")
        constraint = graph.node_attribute(node, name="constraint")
        if residual:
            reactions.append(Line.from_point_and_vector(location, residual * -0.3))
            if constraint:
                constraint.location = location
                constraint.residual = residual
                constraint.compute_tangent()
                tangents.append(Line.from_point_and_vector(location, constraint.tangent))

    elif graph.node_attribute(node, name="constraint"):
        residual: Vector = graph.node_attribute(node, name="residual")
        if residual and residual.length > tol:
            residuals.append(Line.from_point_and_vector(graph.node_point(node), residual))

loads = []

for node in graph.nodes():
    load: Vector = graph.node_attribute(node, name="load")
    if load and load.length:
        loads.append(Line.from_point_and_vector(graph.node_point(node), load))

supports = [graph.node_point(node) for node in graph.nodes_where(anchor=True)]

# =============================================================================
# Visualisation
# =============================================================================

loadcolor = Color.green().darkened(50)
reactioncolor = Color.green()
residualcolor = Color.cyan()
supportcolor = Color(0.2, 0.2, 0.2)
tensioncolor = Color.red()
compressioncolor = Color.blue()

config = Config()
config.camera.target = [0, 0, 0]
config.camera.position = [25, -40, 30]

viewer = Viewer(config=config)

viewer.scene.add(graph, show_points=True)

viewer.scene.add(
    supports[:10],
    name="Supports",
    pointcolor=supportcolor,
    pointsize=10,
)
viewer.scene.add(
    tension,
    name="Tension",
    facecolor=tensioncolor,
    linecolor=tensioncolor.contrast,
)
viewer.scene.add(
    compression,
    name="Compression",
    facecolor=Color.blue(),
    show_lines=False,
)
viewer.scene.add(
    reactions,
    name="Reactions",
    linecolor=reactioncolor,
    linewidth=3,
)
viewer.scene.add(
    residuals,
    name="Residuals",
    linecolor=residualcolor,
    linewidth=3,
)
viewer.scene.add(
    tangents,
    name="Tangents",
    linecolor=Color.magenta(),
    linewidth=3,
)
viewer.scene.add(
    loads,
    name="Loads",
    linecolor=loadcolor,
    linewidth=3,
)
viewer.scene.add(
    constraint.geometry,
    name="Constraints",
    linecolor=Color.cyan(),
    linewidth=2,
)

viewer.show()
