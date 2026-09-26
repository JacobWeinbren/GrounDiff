def classFactory(iface):
    from . import deps
    deps.activate()                    # packages installed by the plugin itself
    from .plugin import GrounDiffPlugin
    return GrounDiffPlugin(iface)
