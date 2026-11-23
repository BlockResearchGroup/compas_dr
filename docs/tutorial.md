# Solvers and loads

Dynamic relaxation advances an auxiliary motion driven by force imbalance.
As damping reduces motion, the network approaches static equilibrium. The
NumPy implementations here use Runge–Kutta integration and viscous damping.
Veenendaal and Block compare viscous and kinetic damping in their
[2012 network form-finding study](https://doi.org/10.1016/j.ijsolstr.2012.08.008).

Build an `InputData` object with coordinates and applied loads in vertex-index
order, one force density per edge and a list of fixed vertex indices.
`dr_numpy` returns a `ResultData` object. Coordinates and residuals have shape
`(n, 3)`. Axial forces, lengths and force densities have shape `(m, 1)`
and follow the input edge order. The solvers update the cached coordinates
in the input object.

Use `dr_constrained_numpy` to model sliding supports. Supply one constraint
entry per vertex, with `None` for an unconstrained vertex. Include sliding
supports in `InputData.fixed` so the geometric constraint moves the support.
A compatible COMPAS geometry backend is required for NURBS operations.

`SelfweightCalculator` computes per-vertex self-weight from tributary area,
thickness and density. It returns one weight per vertex, in mesh vertex
order. Apply those weights to the vertical component of the DR load vectors
with the sign appropriate to your coordinate system.

For a small network with only prescribed force densities, forces and lengths,
the pure-Python `dr` solver takes plain lists and returns a tuple containing
the coordinates, densities, forces, lengths and residuals.
