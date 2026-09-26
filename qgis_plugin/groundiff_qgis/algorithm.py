import os

from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingContext, QgsProcessingException,
                       QgsProcessingParameterBoolean, QgsProcessingParameterEnum, QgsProcessingParameterFile,
                       QgsProcessingParameterFolderDestination, QgsProcessingParameterMultipleLayers,
                       QgsProcessingParameterNumber, QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterRasterLayer)

CHANNELS = [
    ("dsm_max", "DSM, highest return per cell"),
    ("dsm_min", "DSM, lowest return per cell"),
    ("dsm_last", "DSM, lowest last return per cell"),
    ("dtm_before", "lasground_new DTM (before editing)"),
    ("sem_ground", "lasground_new ground share raster"),
    ("sem_nonground", "lasground_new non-ground share raster"),
    ("density", "Return density (per m²)"),
    ("z_std", "Height standard deviation per cell"),
    ("echoes", "Mean number of returns per cell"),
]
PROVIDERS = [
    ("Auto (best available)", None),
    ("NVIDIA CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ("DirectML (any Windows GPU)", ["DmlExecutionProvider", "CPUExecutionProvider"]),
    ("CoreML (Mac)", ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
    ("CPU", ["CPUExecutionProvider"]),
]
BLENDS = ["min", "linear", "mean"]
PRIORS = ["auto", "global", "channel", "none"]
OUTPUTS = [("dtm", "Predicted DTM"), ("p_ground", "Confidence that the gate surface is right"),
           ("p_edit", "Edit probability (before->after models)"),
           ("std", "Uncertainty (m)"), ("dz_before", "Predicted edit vs lasground_new (m)")]

INSTALL_HINT = (
    "Install the missing package into QGIS's Python. On Windows open the OSGeo4W Shell and run "
    "`python -m pip install onnxruntime-directml` (any GPU) or `onnxruntime-gpu` (NVIDIA; needs a "
    "matching CUDA/cuDNN) or `onnxruntime` (CPU), plus `laspy[lazrs]` for point-cloud input. "
    "On macOS: `/Applications/QGIS.app/Contents/MacOS/bin/python3 -m pip install onnxruntime laspy[lazrs]`.")


def _model_params(alg):
    alg.addParameter(QgsProcessingParameterFile("MODEL", "Model (.onnx, with its .json next to it)", extension="onnx"))


def _run_params(alg):
    alg.addParameter(QgsProcessingParameterEnum("BACKEND", "Compute device",
                                                options=[p[0] for p in PROVIDERS], defaultValue=0))
    alg.addParameter(QgsProcessingParameterEnum("BLEND", "Tile blending", options=BLENDS, defaultValue=0))
    alg.addParameter(QgsProcessingParameterEnum("PRIOR", "Prior (PrioStitch)", options=PRIORS, defaultValue=0))
    alg.addParameter(QgsProcessingParameterNumber("SAMPLES", "Diffusion samples (uncertainty if > 1)",
                                                  type=QgsProcessingParameterNumber.Integer,
                                                  defaultValue=1, minValue=1, maxValue=16))
    alg.addParameter(QgsProcessingParameterBoolean("TTA", "Average 8 flips/rotations", defaultValue=False))
    alg.addParameter(QgsProcessingParameterNumber("BATCH", "Network tiles per batch",
                                                  type=QgsProcessingParameterNumber.Integer,
                                                  defaultValue=8, minValue=1, maxValue=128))
    alg.addParameter(QgsProcessingParameterBoolean(
        "OVERLAYS", "Also write coloured overlays for LP360 (RGBA + RGB GeoTIFF)", defaultValue=True))


def _run_kwargs(alg, parameters, context):
    return {"blend": BLENDS[alg.parameterAsEnum(parameters, "BLEND", context)],
            "prior": PRIORS[alg.parameterAsEnum(parameters, "PRIOR", context)],
            "n_samples": alg.parameterAsInt(parameters, "SAMPLES", context),
            "tta": alg.parameterAsBool(parameters, "TTA", context),
            "batch_size": alg.parameterAsInt(parameters, "BATCH", context)}


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
        return "Predict DTM from point-cloud tiles"

    def shortHelpString(self):
        return ("Select any number of LAS/LAZ tiles (use the … button, then 'Add File(s)…'), and for "
                "before->after models the same tiles classified by lasground_new (paired by file name). "
                "Each tile is processed with a buffer of neighbouring points so the combined rasters have no "
                "seams. Several tiles are read and rasterised at once while the GPU works on the previous "
                "one. Results (dtm.tif, p_edit.tif, dz_before.tif, … and *_overlay.tif for LP360) are written "
                "to the output folder; per-tile pieces go to its 'tiles' subfolder.")

    def createInstance(self):
        return PredictTilesAlgorithm()

    def initAlgorithm(self, config=None):
        _model_params(self)
        self.addParameter(QgsProcessingParameterMultipleLayers(
            "AFTER_FILES", "Point-cloud tiles (LAS/LAZ)", layerType=QgsProcessing.TypeFile))
        self.addParameter(QgsProcessingParameterMultipleLayers(
            "BEFORE_FILES", "Same tiles classified by lasground_new (before->after models)",
            layerType=QgsProcessing.TypeFile, optional=True))
        self.addParameter(QgsProcessingParameterNumber("GSD", "Cell size (m)", type=QgsProcessingParameterNumber.Double,
                                                       defaultValue=0.5, minValue=0.05))
        self.addParameter(QgsProcessingParameterNumber("BUFFER", "Neighbour buffer (m)",
                                                       type=QgsProcessingParameterNumber.Double,
                                                       defaultValue=64.0, minValue=0.0))
        self.addParameter(QgsProcessingParameterNumber("WORKERS", "Tiles prepared in parallel",
                                                       type=QgsProcessingParameterNumber.Integer,
                                                       defaultValue=max(1, min(4, (os.cpu_count() or 2) - 1)),
                                                       minValue=1, maxValue=32))
        self.addParameter(QgsProcessingParameterBoolean("DROP_OVERLAP", "Drop overlap points (flag / class 12)",
                                                        defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean("DROP_SYNTHETIC", "Drop synthetic points",
                                                        defaultValue=False))
        _run_params(self)
        self.addParameter(QgsProcessingParameterBoolean("LOAD", "Add the DTM and overlay to the map",
                                                        defaultValue=True))
        self.addParameter(QgsProcessingParameterFolderDestination("OUTPUT_FOLDER", "Output folder"))

    def processAlgorithm(self, parameters, context, feedback):
        pipe = _pipeline()
        out = self.parameterAsString(parameters, "OUTPUT_FOLDER", context)
        try:
            summary = pipe.run_tiles(
                self.parameterAsFile(parameters, "MODEL", context),
                self.parameterAsFileList(parameters, "AFTER_FILES", context), out,
                before_files=self.parameterAsFileList(parameters, "BEFORE_FILES", context),
                providers=PROVIDERS[self.parameterAsEnum(parameters, "BACKEND", context)][1],
                gsd=self.parameterAsDouble(parameters, "GSD", context),
                buffer_m=self.parameterAsDouble(parameters, "BUFFER", context),
                workers=self.parameterAsInt(parameters, "WORKERS", context),
                read_opts={"drop_overlap": self.parameterAsBool(parameters, "DROP_OVERLAP", context),
                           "drop_synthetic": self.parameterAsBool(parameters, "DROP_SYNTHETIC", context)},
                overlays=self.parameterAsBool(parameters, "OVERLAYS", context),
                predict_kwargs=_run_kwargs(self, parameters, context),
                progress=lambda f: feedback.setProgress(int(100 * f)),
                log=feedback.pushInfo, cancelled=feedback.isCanceled)
        except ImportError as e:
            raise QgsProcessingException(f"{e}. {INSTALL_HINT}")
        except (ValueError, KeyError, FileNotFoundError) as e:
            raise QgsProcessingException(str(e))
        outputs = summary.get("outputs", {})
        if self.parameterAsBool(parameters, "LOAD", context):
            for key, name in (("dtm", "GrounDiff DTM"), ("p_edit_overlay", "Edit probability (overlay)"),
                              ("p_ground_overlay", "Low confidence (overlay)")):
                if key in outputs:
                    context.addLayerToLoadOnCompletion(
                        outputs[key], QgsProcessingContext.LayerDetails(name, context.project(), key))
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
                "model was trained on). For point clouds use 'Predict DTM from point-cloud tiles'.")

    def createInstance(self):
        return PredictRastersAlgorithm()

    def initAlgorithm(self, config=None):
        _model_params(self)
        for key, label in CHANNELS:
            self.addParameter(QgsProcessingParameterRasterLayer(key.upper(), label, optional=True))
        _run_params(self)
        for key, label in OUTPUTS:
            self.addParameter(QgsProcessingParameterRasterDestination(
                key.upper(), label, optional=key != "dtm", createByDefault=key in ("dtm", "p_edit")))

    def processAlgorithm(self, parameters, context, feedback):
        pipe = _pipeline()
        paths = {}
        for key, _ in CHANNELS:
            layer = self.parameterAsRasterLayer(parameters, key.upper(), context)
            if layer is not None:
                paths[key] = layer.source()
        if not paths:
            raise QgsProcessingException("Give the input rasters the model needs.")
        try:
            arrs, info = pipe.rasters_from_files(paths)
            outputs = {key: self.parameterAsOutputLayer(parameters, key.upper(), context) for key, _ in OUTPUTS}

            def progress(f):
                if feedback.isCanceled():
                    raise QgsProcessingException("Cancelled")
                feedback.setProgress(int(100 * f))

            kw = _run_kwargs(self, parameters, context)
            written = pipe.run(self.parameterAsFile(parameters, "MODEL", context), arrs, info, outputs,
                               providers=PROVIDERS[self.parameterAsEnum(parameters, "BACKEND", context)][1],
                               blend=kw["blend"], prior=kw["prior"], n_samples=kw["n_samples"], tta=kw["tta"],
                               batch_size=kw["batch_size"],
                               overlays=self.parameterAsBool(parameters, "OVERLAYS", context), progress=progress)
        except ImportError as e:
            raise QgsProcessingException(f"{e}. {INSTALL_HINT}")
        except (ValueError, KeyError, FileNotFoundError) as e:
            raise QgsProcessingException(str(e))
        feedback.pushInfo(f"ONNX Runtime providers used: {written.pop('providers')}")
        for key in [k for k in written if k.endswith("_overlays")]:
            feedback.pushInfo(f"LP360 overlays: {', '.join(written.pop(key))}")
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
        self.addParameter(QgsProcessingParameterFile("FILE", "LAS/LAZ file", fileFilter="LAS/LAZ (*.laz *.las)"))
        self.addParameter(QgsProcessingParameterFile("COMPARE", "Compare with (optional)", optional=True,
                                                     fileFilter="LAS/LAZ (*.laz *.las)"))

    def processAlgorithm(self, parameters, context, feedback):
        pipe = _pipeline()
        try:
            report = pipe.inspect_report(self.parameterAsFile(parameters, "FILE", context),
                                         self.parameterAsFile(parameters, "COMPARE", context) or None)
        except ImportError as e:
            raise QgsProcessingException(f"{e}. {INSTALL_HINT}")
        for line in report.splitlines():
            feedback.pushInfo(line)
        return {}
