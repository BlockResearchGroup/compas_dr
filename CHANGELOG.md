# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

* Added numerical baselines recorded with COMPAS 2 (`tests/baselines`) and tests comparing the solvers, `InputData`, `SelfweightCalculator` and the constraint data against them.
* Added tests for the public module imports, the constraints and `compas_dr.numdata.InputData`.

### Changed

* Changed constraint location setter to not check for existence of attribute `projected`.
* Changed plane projection to use closest point method.
* Changed constraint update damping parameter to `damping` instead of `c`.
* Ported to COMPAS 3: `connectivity_matrix` and `face_matrix` from `compas.linalg` (was `compas.matrices`); `cross_vectors`, `length_vector` and `vector_component` from `compas.linalg` (was `compas.geometry`).
* Changed the requirements to `compas >=3.0.0,<4` and Python `>=3.12`. COMPAS 2 is no longer supported.
* Changed `compas_dr.numdata` to import NumPy unconditionally; the properties of `InputData` always return arrays.
* Changed `SurfaceConstraint.compute_normal` to use `NurbsSurface.normal_at`; `Surface.curvature_at` no longer exists in COMPAS 3.
* `CircleConstraint` now works on a `compas.geometry.Circle`: COMPAS 3 implements `Circle.closest_point`, which raised `NotImplementedError` in COMPAS 2.

### Removed

* Removed the `DATASCHEMA` class attributes of the constraints (COMPAS 3 no longer uses data schemas).
* Removed `compas_dr.install` and `compas_dr.__all_plugins__`: the Rhino installer plugin; COMPAS 3 has no `compas_rhino` and does not read `__all_plugins__`.
* Removed `compas_dr.numdata.has_numpy` and the incomplete list fallbacks of `InputData` for environments without NumPy.
* Removed the `from __future__` imports (Python 2).

### Not verified against COMPAS 2

* `SurfaceConstraint` and `CircleConstraint` on a `Circle` did not work with COMPAS 2 and have no baseline; they are verified against closed-form values with COMPAS 3 only.
* `CurveConstraint` and `SurfaceConstraint` need a NURBS backend such as `compas_occ`, which is not a requirement of `compas_dr`. `CurveConstraint` is compared against a COMPAS 2 baseline when `compas_occ` is installed. The build workflow does not install it, so the backend tests run locally only.
* Loading constraints from data is tested for `LineConstraint` and `PlaneConstraint` only. Loading a `CircleConstraint` from data returns an unusable `CurveConstraint`, as before the port. Loading `CurveConstraint` or `SurfaceConstraint` data written with COMPAS 2 is not tested.
* The examples and scripts are not ported to COMPAS 3.
* Python 3.13 and 3.14 are declared but have not been tested yet.


## [0.3.1] 2024-05-14

### Added

### Changed

### Removed


## [0.3.0] 2024-02-29

### Added

### Changed

### Removed


## [0.2.0] 2024-02-29

### Added

* Added `compas_dr.solvers.dr`.
* Added `compas_dr.solvers.dr_numpy`.
* Added `compas_dr.solvers.dr_constrained_numpy`.
* Added `compas_dr.types`.
* Added `compas_dr.numdata.InputData`.
* Added `compas_dr.numdata.ResultData`.
* Added `compas_dr.loads.SelfweightCalculator`.
* Added `compas_dr.constraints.Constraint`.
* Added `compas_dr.constraints.LineConstraint`.
* Added `compas_dr.constraints.PlaneConstraint`.
* Added `compas_dr.constraints.CircleConstraint`.
* Added `compas_dr.constraints.NurbsCurveConstraint`.
* Added `compas_dr.constraints.NurbsSurfaceConstraint`.

### Changed

### Removed
