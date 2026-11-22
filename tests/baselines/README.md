# Numerical baselines

The `*.npz` files, `constraints.json`, and `provenance.json` in this directory
are **read-only test fixtures**. They hold the results of `compas_dr` at its
last COMPAS 2 revision and are the reference for every later change.

- Never edit or regenerate them to make a test pass. A deviation beyond the
  tolerance is a finding that needs a decision.
- They change only by an explicit decision, in a commit of their own.
- Results of a later revision are never stored here.

## How they were produced

| | |
|---|---|
| Package revision | `compas_dr` `84a3b4b3b0017280da73054ac811aab020b43beb`, installed non-editable |
| Environment | Conda environment `brg-compas2`: `compas` 2.16.0, `compas_occ` 1.5.0, Python 3.12 |
| Command | `<python of brg-compas2> tests/baselines/record.py tests/baselines`, from the repository root |

The exact versions, platform, date, and the status of every case are in
`provenance.json`. `record.py` refuses to run with COMPAS 3, with a
`compas_dr` that is not the installed legacy revision, or into a directory
that already holds baselines.

## Files

| File | Content |
|---|---|
| `cases.py` | The single definition of the inputs and calls of all cases. |
| `record.py` | The recorder. Runs under COMPAS 2 only. |
| `<case>.npz` | Inputs (`in_*`) and outputs (`out_*`) of one case, loaded with `allow_pickle=False`. |
| `constraints.json` | COMPAS 2 JSON of a line and a plane constraint (case dr-12); the plane constraint has a `rhino_guid`. |
| `provenance.json` | Revision, versions, platform, command, date, status per case. |

## Cases

Shared input "grid": `Mesh.from_meshgrid(dx=10, nx=10)`, the four corners
fixed, force density 10 on boundary edges and 1 inside, load (0, 0, -0.5) per
vertex. "Column" is the eleven vertices at x = 5; in the constrained cases
the column is fixed as well and every column vertex has its own constraint.

| Case | Function | Input |
|---|---|---|
| dr-01 | `InputData.from_mesh` | grid, optional arguments omitted |
| dr-02 | `InputData` | grid, with `fpre`, `lpre`, `linit`, `E`, `radius` |
| dr-03, dr-04 | `dr_numpy` | grid, `rk_steps` 2 and 4, `kmax=20`, `tol1=tol2=0` |
| dr-05 | `dr_numpy` | grid, defaults, converged |
| dr-06 | `dr_numpy` | grid, `E=210`, `radius=5`, `linit=0.95` per edge, `kmax=20`, `tol1=tol2=0` |
| dr-07, dr-08 | `dr_constrained_numpy` | grid, column on a `Plane` and on a `Line`, `kmax=20`, `tol1=tol2=0` |
| dr-09 | `dr_constrained_numpy` | as dr-07 with defaults, converged |
| dr-10 | `dr` | grid, defaults |
| dr-11 | `SelfweightCalculator` | grid, t = 0.1, density 22, on the geometry of dr-05 |
| dr-12 | constraint data | `json_dumps` of a line and a plane constraint |
| dr-13 | `dr_constrained_numpy` | grid, column on a `NurbsCurve`, `kmax=20`, `tol1=tol2=0`; needs `compas_occ` |

The solver cases store the result arrays, the two criteria per iteration
(collected through the solver callback), and the iteration count. The
constrained cases also store the constraint locations, dr-13 the curve
parameters. The iteration numbers that `dr_constrained_numpy` prints are
captured and not stored.

## Comparison

`tests/test_baselines.py` re-runs every case. Regenerated inputs, integers,
shapes, and iteration counts must be equal. Float outputs must satisfy
`abs(new - base) <= 1e-9 + 1e-9 * abs(base)`. `pytest tests/test_baselines.py -rP`
prints the maximum absolute and relative deviation per output. In the
environment that shares the numerical stack of the baseline the expected
deviation is exactly 0; anything else is a finding even inside the tolerance.
