import importlib

import pytest

import compas_dr


@pytest.mark.parametrize(
    "name",
    [
        "compas_dr",
        "compas_dr.constraints",
        "compas_dr.loads",
        "compas_dr.numdata",
        "compas_dr.solvers",
        "compas_dr.types",
    ],
)
def test_public_module_imports(name):
    assert importlib.import_module(name).__name__ == name


def test_install_plugin_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("compas_dr.install")
    assert not hasattr(compas_dr, "__all_plugins__")
