from importlib import import_module
from pkgutil import walk_packages

import compas_dr


def test_all_package_modules_import() -> None:
    for module in walk_packages(compas_dr.__path__, prefix="compas_dr."):
        import_module(module.name)


def test_install_plugin_registration() -> None:
    from compas.plugins import ExtensionPoint
    from compas.plugins import PluginManager

    extension = ExtensionPoint("https://plugins.compas.dev/", "install", "installable_rhino_packages")
    (implementation,) = [plugin for plugin in PluginManager().collect(extension) if plugin.plugin.__name__ == "compas_dr.install"]

    assert implementation.priority == 1
    assert implementation.method() == ["compas_dr"]
