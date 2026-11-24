from importlib import import_module
from pkgutil import walk_packages

import compas_dr

from compas_dr.constraints import Constraint


def test_constraint_classes_have_no_legacy_data_schema() -> None:
    classes = {Constraint, *Constraint.GEOMETRY_CONSTRAINT.values()}
    assert all(not hasattr(cls, "DATASCHEMA") for cls in classes)


def test_all_package_modules_import() -> None:
    for module in walk_packages(compas_dr.__path__, prefix="compas_dr."):
        import_module(module.name)
