def classFactory(iface):
    from .plugin import GrounDiffPlugin
    return GrounDiffPlugin(iface)
