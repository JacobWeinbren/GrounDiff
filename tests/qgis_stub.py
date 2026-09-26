"""A minimal stand-in for qgis.core, enough to run the plugin's algorithm
classes (initAlgorithm / processAlgorithm) in tests without QGIS."""
import sys
import types


_SETTINGS: dict = {}


def install(new_enums: bool = True):
    core = types.ModuleType("qgis.core")

    class _Param:
        Integer, Double = 0, 1
        FlagAdvanced = 2

        def __init__(self, name, description="", *a, **kw):
            self.name, self.kw, self._flags = name, kw, 0

        def flags(self):
            return self._flags

        def setFlags(self, f):
            self._flags = f

    class QgsProcessingParameterNumber(_Param):
        pass

    names = ["QgsProcessingParameterBoolean", "QgsProcessingParameterEnum", "QgsProcessingParameterFile",
             "QgsProcessingParameterFolderDestination", "QgsProcessingParameterMultipleLayers",
             "QgsProcessingParameterRasterDestination", "QgsProcessingParameterRasterLayer"]
    for n in names:
        setattr(core, n, type(n, (_Param,), {}))
    core.QgsProcessingParameterNumber = QgsProcessingParameterNumber

    class QgsProcessing:
        TypeFile = 99
        TypePointCloud = 98
        TEMPORARY_OUTPUT = "TEMPORARY_OUTPUT"
    core.QgsProcessing = QgsProcessing
    core.QgsProcessingParameterDefinition = _Param

    class QgsSettings:
        store = _SETTINGS                  # shared across re-installs, like QGIS's settings file

        def value(self, k, default=None):
            return self.store.get(k, default)

        def setValue(self, k, v):
            self.store[k] = v
    core.QgsSettings = QgsSettings

    if new_enums:
        class Qgis:
            class ProcessingNumberParameterType:
                Integer, Double = 0, 1

            class ProcessingSourceType:
                File = 99
                PointCloud = 98

            class ProcessingParameterFlag:
                Advanced = 2
        core.Qgis = Qgis
    else:                         # QGIS 3.22-3.34: Qgis exists but without these enums
        core.Qgis = type("Qgis", (), {})

    class QgsProcessingException(Exception):
        pass
    core.QgsProcessingException = QgsProcessingException

    class QgsProcessingLayerPostProcessorInterface:
        def __init__(self):
            pass
    core.QgsProcessingLayerPostProcessorInterface = QgsProcessingLayerPostProcessorInterface

    class _LayerDetails:
        def __init__(self, name, project, output_name=""):
            self.name, self.post = name, None

        def setPostProcessor(self, pp):
            self.post = pp

    class QgsProcessingContext:
        LayerDetails = _LayerDetails

        def __init__(self):
            self.to_load = {}

        def project(self):
            return None

        def addLayerToLoadOnCompletion(self, path, details):
            self.to_load[path] = details

        def layerToLoadOnCompletionDetails(self, path):
            return self.to_load[path]

        def willLoadLayerOnCompletion(self, path):
            return path in self.to_load
    core.QgsProcessingContext = QgsProcessingContext

    class _Layer:
        def __init__(self, src):
            self._src = src

        def source(self):
            return self._src

        def name(self):
            return self._src.split("/")[-1]

    class QgsProcessingAlgorithm:
        def __init__(self):
            self.params = {}

        def addParameter(self, p):
            self.params[p.name] = p

        def _get(self, parameters, name):
            v = parameters.get(name)
            if v is None and name in self.params:
                v = self.params[name].kw.get("defaultValue")
            return v

        def parameterAsString(self, parameters, name, context):
            return str(self._get(parameters, name) or "")

        parameterAsFile = parameterAsString

        def parameterAsFileList(self, parameters, name, context):
            return list(self._get(parameters, name) or [])

        def parameterAsLayerList(self, parameters, name, context):
            return [_Layer(p) for p in (self._get(parameters, name) or [])]

        def parameterAsDouble(self, parameters, name, context):
            return float(self._get(parameters, name))

        def parameterAsInt(self, parameters, name, context):
            return int(self._get(parameters, name))

        def parameterAsEnum(self, parameters, name, context):
            return int(self._get(parameters, name) or 0)

        def parameterAsBool(self, parameters, name, context):
            return bool(self._get(parameters, name))

        def parameterAsRasterLayer(self, parameters, name, context):
            v = parameters.get(name)
            return _Layer(v) if v else None

        def parameterAsOutputLayer(self, parameters, name, context):
            path = parameters.get(name)
            if path:
                context.addLayerToLoadOnCompletion(path, _LayerDetails(name, None))
            return path
    core.QgsProcessingAlgorithm = QgsProcessingAlgorithm

    class QgsProcessingProvider:
        def addAlgorithm(self, a):
            pass
    core.QgsProcessingProvider = QgsProcessingProvider

    class QgsApplication:
        pass
    core.QgsApplication = QgsApplication

    qgis = types.ModuleType("qgis")
    qgis.core = core
    sys.modules["qgis"] = qgis
    sys.modules["qgis.core"] = core
    return core


class Feedback:
    def __init__(self, cancel_after=None):
        self.info, self.warnings, self.errors, self.progress = [], [], [], []
        self.cancel_after = cancel_after
        self.texts = []

    def setProgressText(self, t):
        self.texts.append(t)

    def setProgress(self, p):
        self.progress.append(p)

    def pushInfo(self, m):
        self.info.append(m)

    def pushWarning(self, m):
        self.warnings.append(m)

    def reportError(self, m, fatalError=False):
        self.errors.append(m)

    def isCanceled(self):
        return self.cancel_after is not None and len(self.progress) >= self.cancel_after
