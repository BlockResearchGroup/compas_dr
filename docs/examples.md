# Worked examples

## Loaded network

Two fixed vertices sit at `(0, 0, 0)` and `(4, 0, 0)`. Their edges to the free
vertex have force densities 1 and 3. Balancing a downward load of 4 gives the
free vertex at `(3, 0, -1)`.

```python
--8<-- "docs/examples/loaded_network.py"
```

## Mesh examples

The original [mesh grid script](examples/example_mesh_basic.py) fixes the
corners of a mesh. The [supported arch script](examples/example_mesh_constrained_on-arch.py)
adds sliding vertices along a NURBS curve. The [spoked wheel script](examples/example_spokewheel.py)
constructs a cable network. These COMPAS 2 examples are retained as references
for the viewer and geometry backend workflows. The loaded-network example
is exercised in COMPAS 3 CI.

![Mesh supported by an arch](_images/example_mesh_constrained_on-arch.png)
