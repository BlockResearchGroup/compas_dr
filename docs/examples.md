# Worked examples

## Loaded network

Two fixed vertices sit at `(0, 0, 0)` and `(4, 0, 0)`. Their edges to the free
vertex have force densities 1 and 3. Balancing a downward load of 4 gives the
free vertex at `(3, 0, -1)`.

```python
--8<-- "docs/examples/loaded_network.py"
```

## Mesh with fixed corners or an arch

Run the scripts from the repository root in the interactive COMPAS 3
environment with this package and `compas_viewer` installed. The arch example
also requires `compas_occ` and its native `pythonocc-core` dependency.

```sh
python docs/examples/viewer/mesh_fixed_corners.py
python docs/examples/viewer/mesh_on_arch.py
```

Both grids have fixed corners (red) and higher force density along the
boundary. In the second example, the middle row slides along a NURBS arch
(blue). Green lines show support reactions, scaled for display.

```python
--8<-- "docs/examples/viewer/mesh_fixed_corners.py"
```

```python
--8<-- "docs/examples/viewer/mesh_on_arch.py"
```

## Prestressed spoked wheel

The double-layer wheel uses prestressed struts, a downward load on the lower
inner ring, and supports that slide along the outer ellipse. Two rim nodes
are fixed. Red members carry tension and blue members carry compression. The
wheel also requires COMPAS OCC. The view shows the relaxation after up to
100 iterations, before the sliding nodes bunch up along the rim.

```sh
python docs/examples/viewer/spoked_wheel.py
```

```python
--8<-- "docs/examples/viewer/spoked_wheel.py"
```

## Historical illustrations

These illustrations are from earlier COMPAS 2 viewer runs. They show a mesh
with fixed corners and a mesh with sliding supports along a NURBS arch.

![Mesh with fixed corners](assets/images/example_mesh_basic.png)
![Mesh supported by an arch](assets/images/example_mesh_constrained_on-arch.png)
