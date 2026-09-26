import os

from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingContext, QgsProcessingException,
                       QgsProcessingLayerPostProcessorInterface, QgsProcessingParameterBoolean,
                       QgsProcessingParameterEnum, QgsProcessingParameterFile,
                       QgsProcessingParameterFolderDestination, QgsProcessingParameterMultipleLayers,
                       QgsProcessingParameterNumber, QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterRasterLayer)

# QGIS 3.36+ / 4.x moved these enums to Qgis.*; the old names are gone in 4.x.
try:
    from qgis.core import Qgis
    NUM_INT = Qgis.ProcessingNumberParameterType.Integer
    NUM_DOUBLE = Qgis.ProcessingNumberParameterType.Double
    SOURCE_FILE = Qgis.ProcessingSourceType.File
    SOURCE_POINTCLOUD = Qgis.ProcessingSourceType.PointCloud
    FLAG_ADVANCED = Qgis.ProcessingParameterFlag.Advanced
except (ImportError, AttributeError):                    # QGIS 3.22 - 3.34
    NUM_INT = QgsProcessingParameterNumber.Integer
    NUM_DOUBLE = QgsProcessingParameterNumber.Double
    SOURCE_FILE = QgsProcessing.TypeFile
    SOURCE_POINTCLOUD = getattr(QgsProcessing, "TypePointCloud", None)
    FLAG_ADVANCED = getattr(__import__("qgis.core", fromlist=["QgsProcessingParameterDefinition"])
                            .QgsProcessingParameterDefinition, "FlagAdvanced", None)

SETTINGS_MODEL = "groundiff/model_path"


def _settings():
    from qgis.core import QgsSettings
    return QgsSettings()


def _advanced(p):
    """Put a parameter under the dialog's 'Advanced' section."""
    if FLAG_ADVANCED is not None:
        try:
            p.setFlags(p.flags() | FLAG_ADVANCED)
        except Exception:
            pass
    return p


def _model_path(alg, parameters, context) -> str:
    """The model chosen in the dialog, else the one used last time (remembered)."""
    m = alg.parameterAsFile(parameters, "MODEL", context)
    s = _settings()
    if m:
        s.setValue(SETTINGS_MODEL, m)
        return m
    m = s.value(SETTINGS_MODEL, "")
    if not m or not os.path.exists(m):
        raise QgsProcessingException("Choose the model file (.onnx) once; it is remembered afterwards.")
    return m

LAS_FILTER = "LAS/LAZ (*.laz *.las *.LAZ *.LAS)"
CHANNELS = [
    ("dsm_max", "DSM, highest return per cell"),
    ("dsm_min", "DSM, lowest return per cell"),
    ("dsm_last", "DSM, lowest last return per cell"),
    ("dtm_before", "lasground_new DTM (before editing; also gives the predicted edit for DSM-only models)"),
    ("sem_ground", "lasground_new ground share raster"),
    ("sem_nonground", "lasground_new non-ground share raster"),
    ("density", "Return density (per m²)"),
    ("z_std", "Height standard deviation per cell"),
    ("echoes", "Mean number of returns per cell"),
    ("has_return", "1 where the cell has a return"),
    ("in_survey", "1 inside the LiDAR coverage"),
]
PROVIDERS = [
    ("Auto (best available)", None),
    ("DirectML (any Windows GPU)", ["DmlExecutionProvider", "CPUExecutionProvider"]),
    ("NVIDIA CUDA (needs onnxruntime-gpu with matching CUDA/cuDNN)", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ("CoreML (Mac)", ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
    ("CPU", ["CPUExecutionProvider"]),
]
BLENDS = ["linear", "min", "mean"]
PRIORS = ["auto", "global", "channel", "none"]
OUTPUTS = [("dtm", "Predicted DTM"), ("p_edit", "Edit probability"),
           ("dz_before", "Predicted edit vs lasground_new (m)"), ("std", "Uncertainty (m)"),
           ("p_ground", "Probability the DSM is ground (DSM-only models)")]

_KEEP = []          # post-processors must outlive processAlgorithm


class _Style(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, qml):
        super().__init__()
        self.qml = qml

    def postProcessLayer(self, layer, context, feedback):
        if layer is not None and os.path.exists(self.qml):
            layer.loadNamedStyle(self.qml)
            layer.triggerRepaint()


def _load(context, path, name, qml=None):
    details = QgsProcessingContext.LayerDetails(name, context.project(), name)
    context.addLayerToLoadOnCompletion(path, details)
    if qml and os.path.exists(qml):
        pp = _Style(qml)
        _KEEP.append(pp)
        context.layerToLoadOnCompletionDetails(path).setPostProcessor(pp)


def _model_params(alg):
    try:
        last = _settings().value(SETTINGS_MODEL, "") or None
    except Exception:
        last = None
    alg.addParameter(QgsProcessingParameterFile("MODEL", "Model (.onnx; remembered after the first run)",
                                                extension="onnx", optional=True, defaultValue=last))


def _run_params(alg):
    add = lambda p: alg.addParameter(_advanced(p))
    add(QgsProcessingParameterEnum("BACKEND", "Compute device", options=[p[0] for p in PROVIDERS], defaultValue=0))
    add(QgsProcessingParameterEnum("BLEND", "Tile blending", options=BLENDS, defaultValue=0))
    add(QgsProcessingParameterEnum("PRIOR", "Prior (PrioStitch)", options=PRIORS, defaultValue=0))
    add(QgsProcessingParameterNumber(
        "SAMPLES", "Diffusion samples (more = smoother edit probability and an uncertainty map; time x samples)",
        type=NUM_INT, defaultValue=4, minValue=1, maxValue=16))
    add(QgsProcessingParameterBoolean("TTA", "Average 8 flips/rotations", defaultValue=False))
    add(QgsProcessingParameterNumber("BATCH", "Network tiles per batch", type=NUM_INT,
                                     defaultValue=8, minValue=1, maxValue=128))
    add(QgsProcessingParameterBoolean(
        "OVERLAYS", "Also write coloured overlays for LP360 (RGBA + RGB GeoTIFF)", defaultValue=True))


def _run_kwargs(alg, parameters, context):
    return {"blend": BLENDS[alg.parameterAsEnum(parameters, "BLEND", context)],
            "prior": PRIORS[alg.parameterAsEnum(parameters, "PRIOR", context)],
            "n_samples": alg.parameterAsInt(parameters, "SAMPLES", context),
            "tta": alg.parameterAsBool(parameters, "TTA", context),
            "batch_size": alg.parameterAsInt(parameters, "BATCH", context)}


INSTALL_HINT = ("Use Plugins > GrounDiff > Install / check GrounDiff components (one click; it installs "
                "ONNX Runtime and laspy into your QGIS profile), then run the tool again.")


def _pipeline():
    try:
        from . import pipeline
        return pipeline
    except ImportError as e:
        raise QgsProcessingException(f"{e}. {INSTALL_HINT}")


class PredictTilesAlgorithm(QgsProcessingAlgorithm):
    """Many LAS/LAZ tiles -> one seamless set of rasters in an output folder."""

    def name(self):
        return "predict_dtm_tiles"

    def displayName(self):
        return "Find edits in point clouds"

    def shortHelpString(self):
        return ("Pick the lasground_new tiles: point-cloud layers already in the project, and/or LAS/LAZ files "
                "(… button, 'Add File(s)' or 'Add Directory'). Choose the model once; it is remembered. Press "
                "Run: the edit probability, the predicted change to the lasground_new DTM and the ranked "
                "priority blocks are added to the map. Results go to a temporary folder unless you choose one "
                "(for LP360: the *_overlay.tif files and priority.shp there). Settings under 'Advanced' can "
                "normally be left alone.")

    def createInstance(self):
        return PredictTilesAlgorithm()

    def initAlgorithm(self, config=None):
        if SOURCE_POINTCLOUD is not None:
            self.addParameter(QgsProcessingParameterMultipleLayers(
                "LAYERS", "Point-cloud layers in the project (lasground_new tiles)", layerType=SOURCE_POINTCLOUD,
                optional=True))
        self.addParameter(QgsProcessingParameterMultipleLayers(
            "TILES", "and/or LAS/LAZ files", layerType=SOURCE_FILE, optional=True))
        _model_params(self)
        add = lambda p: self.addParameter(_advanced(p))
        add(QgsProcessingParameterNumber("GSD", "Cell size (m, 0 = as trained)", type=NUM_DOUBLE,
                                         defaultValue=0.0, minValue=0.0))
        add(QgsProcessingParameterNumber("BUFFER", "Neighbour buffer (m, -1 = automatic)",
                                         type=NUM_DOUBLE, defaultValue=-1.0, minValue=-1.0))
        add(QgsProcessingParameterNumber("WORKERS", "Tiles prepared in parallel (each needs a few GB RAM)",
                                         type=NUM_INT, defaultValue=1, minValue=1, maxValue=32))
        add(QgsProcessingParameterNumber("BLOCK", "Priority block size (m)", type=NUM_DOUBLE,
                                         defaultValue=100.0, minValue=5.0))
        add(QgsProcessingParameterBoolean("DROP_OVERLAP", "Drop overlap points (flag / class 12)", defaultValue=False))
        add(QgsProcessingParameterBoolean("DROP_SYNTHETIC", "Drop synthetic points", defaultValue=False))
        _run_params(self)
        add(QgsProcessingParameterBoolean("LOAD", "Add the results to the map", defaultValue=True))
        self.addParameter(QgsProcessingParameterFolderDestination(
            "OUTPUT_FOLDER", "Output folder (optional)", defaultValue=QgsProcessing.TEMPORARY_OUTPUT))

    def processAlgorithm(self, parameters, context, feedback):
        pipe = _pipeline()
        from .core.batch import Cancelled
        out = self.parameterAsString(parameters, "OUTPUT_FOLDER", context)
        if not out or out == QgsProcessing.TEMPORARY_OUTPUT:
            import tempfile
            out = tempfile.mkdtemp(prefix="groundiff_")
        tiles = []
        if SOURCE_POINTCLOUD is not None:
            for lyr in self.parameterAsLayerList(parameters, "LAYERS", context) or []:
                src = lyr.source().split("|")[0]
                if not src.lower().endswith((".las", ".laz")):
                    raise QgsProcessingException(f"{lyr.name()}: not a LAS/LAZ file ({src})")
                tiles.append(src)
        tiles += self.parameterAsFileList(parameters, "TILES", context) or []
        if not tiles:
            raise QgsProcessingException("Select point-cloud layers or LAS/LAZ files.")
        model = _model_path(self, parameters, context)
        gsd = self.parameterAsDouble(parameters, "GSD", context)
        buf = self.parameterAsDouble(parameters, "BUFFER", context)
        ro = {}
        if self.parameterAsBool(parameters, "DROP_OVERLAP", context):
            ro["drop_overlap"] = True
        if self.parameterAsBool(parameters, "DROP_SYNTHETIC", context):
            ro["drop_synthetic"] = True
        try:
            summary = pipe.run_tiles(
                model, tiles, out,
                providers=PROVIDERS[self.parameterAsEnum(parameters, "BACKEND", context)][1],
                gsd=gsd if gsd > 0 else None, buffer_m=buf if buf >= 0 else None,
                workers=self.parameterAsInt(parameters, "WORKERS", context), read_opts=ro,
                overlays=self.parameterAsBool(parameters, "OVERLAYS", context),
                predict_kwargs=_run_kwargs(self, parameters, context),
                block_m=self.parameterAsDouble(parameters, "BLOCK", context),
                progress=lambda f: feedback.setProgress(int(100 * f)),
                log=feedback.pushInfo, cancelled=feedback.isCanceled)
        except Cancelled:
            raise QgsProcessingException("Cancelled")
        except ImportError as e:
            raise QgsProcessingException(f"{e}. {INSTALL_HINT}")
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
            raise QgsProcessingException(str(e))
        for f in summary.get("failed", []):
            feedback.reportError(f"{f.get('job', f['file'])}: {f['error']}", fatalError=False)
        outputs = summary.get("outputs", {})
        if self.parameterAsBool(parameters, "LOAD", context):
            for key, name in (("dtm", "GrounDiff DTM"), ("dz_before", "Predicted edit (m)"),
                              ("p_edit", "Edit probability")):
                if key in outputs:
                    _load(context, outputs[key], name, os.path.splitext(outputs[key])[0] + ".qml")
            pr = os.path.join(out, "priority.geojson")
            if os.path.exists(pr):
                _load(context, pr, "Edit priority blocks", os.path.join(out, "priority.qml"))
        for k, v in outputs.items():
            feedback.pushInfo(f"{k}: {v}")
        return {"OUTPUT_FOLDER": out}


class PredictRastersAlgorithm(QgsProcessingAlgorithm):
    def name(self):
        return "predict_dtm"

    def displayName(self):
        return "Predict DTM from rasters"

    def shortHelpString(self):
        return ("Runs the model on input rasters that are already on one grid (one GeoTIFF per channel the "
                "model was trained on, e.g. from `python -m groundiff.data.preprocess --geotiff`). For point "
                "clouds use 'Predict DTM and edit priorities from point-cloud tiles'. Outputs the model cannot "
                "produce are skipped with a warning.")

    def createInstance(self):
        return PredictRastersAlgorithm()

    def initAlgorithm(self, config=None):
        _model_params(self)
        for key, label in CHANNELS:
            self.addParameter(QgsProcessingParameterRasterLayer(key.upper(), label, optional=True))
        _run_params(self)
        for key, label in OUTPUTS:
            self.addParameter(QgsProcessingParameterRasterDestination(
                key.upper(), label, optional=key != "dtm", createByDefault=key == "dtm"))

    def processAlgorithm(self, parameters, context, feedback):
        pipe = _pipeline()
        paths = {}
        for key, _ in CHANNELS:
            layer = self.parameterAsRasterLayer(parameters, key.upper(), context)
            if layer is not None:
                paths[key] = layer.source()
        if not paths:
            raise QgsProcessingException("Give the input rasters the model needs.")
        model = _model_path(self, parameters, context)
        kw = _run_kwargs(self, parameters, context)
        try:
            spec = pipe.load_spec(model)
            can = pipe.producible(spec, kw["n_samples"], kw["tta"], has_before="dtm_before" in paths)
            outputs = {}
            for key, _ in OUTPUTS:
                requested = parameters.get(key.upper())
                if key in can:
                    outputs[key] = self.parameterAsOutputLayer(parameters, key.upper(), context)
                elif requested:
                    feedback.pushWarning(f"{key}: this model does not produce it; skipped")
            arrs, info = pipe.rasters_from_files(paths)

            def progress(f):
                if feedback.isCanceled():
                    raise QgsProcessingException("Cancelled")
                feedback.setProgress(int(100 * f))

            written = pipe.run(model, arrs, info, outputs,
                               providers=PROVIDERS[self.parameterAsEnum(parameters, "BACKEND", context)][1],
                               blend=kw["blend"], prior=kw["prior"], n_samples=kw["n_samples"], tta=kw["tta"],
                               batch_size=kw["batch_size"],
                               overlays=self.parameterAsBool(parameters, "OVERLAYS", context), progress=progress)
        except ImportError as e:
            raise QgsProcessingException(f"{e}. {INSTALL_HINT}")
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
            raise QgsProcessingException(str(e))
        feedback.pushInfo(f"ONNX Runtime providers used: {written.pop('providers')}")
        warn = written.pop("warning", None)
        if warn:
            feedback.pushWarning(warn)
        for key in [k for k in written if k.endswith("_overlays")]:
            feedback.pushInfo(f"LP360 overlays: {', '.join(written.pop(key))}")
        for key, path in written.items():
            qml = os.path.splitext(path)[0] + ".qml"
            if os.path.exists(qml) and context.willLoadLayerOnCompletion(path):
                pp = _Style(qml)
                _KEEP.append(pp)
                context.layerToLoadOnCompletionDetails(path).setPostProcessor(pp)
        return {key.upper(): path for key, path in written.items()}


class InspectLasAlgorithm(QgsProcessingAlgorithm):
    def name(self):
        return "inspect_las"

    def displayName(self):
        return "Inspect LAS/LAZ file"

    def shortHelpString(self):
        return ("Reports header, CRS, flags, classes and return statistics, and warns about things that "
                "make other software (e.g. QGIS point-cloud layers via PDAL) reject a file. Optionally "
                "compares two files, e.g. an EA delivery and the same tile saved by LP360.")

    def createInstance(self):
        return InspectLasAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile("FILE", "LAS/LAZ file", fileFilter=LAS_FILTER))
        self.addParameter(QgsProcessingParameterFile("COMPARE", "Compare with (optional)", optional=True,
                                                     fileFilter=LAS_FILTER))

    def processAlgorithm(self, parameters, context, feedback):
        pipe = _pipeline()
        try:
            report = pipe.inspect_report(self.parameterAsFile(parameters, "FILE", context),
                                         self.parameterAsFile(parameters, "COMPARE", context) or None)
        except ImportError as e:
            raise QgsProcessingException(f"{e}. {INSTALL_HINT}")
        except (ValueError, RuntimeError, OSError) as e:
            raise QgsProcessingException(str(e))
        for line in report.splitlines():
            feedback.pushInfo(line)
        return {}
