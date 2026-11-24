# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

* Added independent equilibrium, numerical data, self-weight and plugin regression tests.
* Added COMPAS 3 Viewer examples for fixed and arch-supported meshes and a prestressed spoked wheel.

### Changed

* Aligned numerical imports, plugin discovery and constraint serialization with COMPAS 3.
* Updated the supported Python versions to 3.12–3.14 and added locked development tools.
* Migrated the documentation build to MkDocs and updated solver annotations and docstrings.
* Organised the documentation assets, archived COMPAS 2 viewer scripts, and added the maintained example to the Pyright check.
* Organised the maintained examples into named sections and formatted longer calls with one argument per line.
* Simplified the loaded-network example to demonstrate solver inputs and results.

* Changed constraint location setter to not check for existence of attribute `projected`.
* Changed plane projection to use closest point method.
* Changed constraint update damping parameter to `damping` instead of `c`.

### Removed

* Removed the legacy `DATASCHEMA` attributes from all constraints to align with COMPAS 3. Existing constraint JSON payloads remain supported.
* Removed the superseded reStructuredText pages and Sphinx configuration.
* Removed the legacy Viewer examples and the `scripts/` and `notebooks/` folders.
* Removed the unused `temp/` placeholder directory.
* Removed the obsolete Rhino install plugin and its package entry point.
* Removed obsolete Python 2.7 compatibility imports, modernised legacy `super` calls, and documented the Python 3.12+ convention.
* Removed the unconditional per-iteration print from `dr_constrained_numpy`. Use its callback to report progress.

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
