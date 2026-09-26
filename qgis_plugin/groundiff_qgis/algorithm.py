from qgis.core import (QgsProcessingAlgorithm, QgsProcessingException, QgsProcessingParameterBoolean,
                       QgsProcessingParameterEnum, QgsProcessingParameterFile, QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterDestination, QgsProcessingParameterRasterLayer)

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


class PredictDtmAlgorithm(QgsProcessingAlgorithm):
    def name(self):
        return "predict_dtm"

    def displayName(self):
        return "Predict DTM (GrounDiff)"

    def shortHelpString(self):
        return ("Runs an ONNX model exported with `python -m groundiff.export` (the .json file with the "
                "same name must sit next to the .onnx). Give EITHER the input rasters the model was "
                "trained on (all on one grid) OR the EA point cloud tile plus, for before->after models, "
                "the same tile re-classified by lasground_new.")

    def createInstance(self):
        return PredictDtmAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile("MODEL", "Model (.onnx)", extension="onnx"))
        self.addParameter(QgsProcessingParameterFile("AFTER_LAZ", "Point cloud tile (LAZ/LAS)",
                                                     optional=True, fileFilter="LAS/LAZ (*.laz *.las)"))
        self.addParameter(QgsProcessingParameterFile("BEFORE_LAZ", "Same tile classified by lasground_new",
                                                     optional=True, fileFilter="LAS/LAZ (*.laz *.las)"))
        self.addParameter(QgsProcessingParameterNumber("GSD", "Cell size for point-cloud input (m)",
                                                       type=QgsProcessingParameterNumber.Double,
                                                       defaultValue=0.5, minValue=0.05))
        for key, label in CHANNELS:
            self.addParameter(QgsProcessingParameterRasterLayer(key.upper(), label, optional=True))
        self.addParameter(QgsProcessingParameterEnum("BACKEND", "Compute device",
                                                     options=[p[0] for p in PROVIDERS], defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum("BLEND", "Tile blending", options=BLENDS, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum("PRIOR", "Prior (PrioStitch)", options=PRIORS, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber("SAMPLES", "Diffusion samples (uncertainty if > 1)",
                                                       type=QgsProcessingParameterNumber.Integer,
                                                       defaultValue=1, minValue=1, maxValue=16))
        self.addParameter(QgsProcessingParameterBoolean("TTA", "Average 8 flips/rotations", defaultValue=False))
        self.addParameter(QgsProcessingParameterNumber("BATCH", "Tiles per batch",
                                                       type=QgsProcessingParameterNumber.Integer,
                                                       defaultValue=8, minValue=1, maxValue=128))
        for key, label in OUTPUTS:
            self.addParameter(QgsProcessingParameterRasterDestination(
                key.upper(), label, optional=key != "dtm", createByDefault=key in ("dtm", "p_ground")))

    def processAlgorithm(self, parameters, context, feedback):
        try:
            from . import pipeline
        except ImportError as e:
            raise QgsProcessingException(f"{e}. {INSTALL_HINT}")
        model = self.parameterAsFile(parameters, "MODEL", context)
        after = self.parameterAsFile(parameters, "AFTER_LAZ", context)
        before = self.parameterAsFile(parameters, "BEFORE_LAZ", context)
        try:
            if after:
                feedback.pushInfo("Rasterising point cloud(s)...")
                arrs, info = pipeline.rasters_from_points(after, before or None,
                                                          self.parameterAsDouble(parameters, "GSD", context))
            else:
                paths = {}
                for key, _ in CHANNELS:
                    layer = self.parameterAsRasterLayer(parameters, key.upper(), context)
                    if layer is not None:
                        paths[key] = layer.source()
                if not paths:
                    raise QgsProcessingException("Give either a point cloud tile or the input rasters.")
                arrs, info = pipeline.rasters_from_files(paths)
            outputs = {key: self.parameterAsOutputLayer(parameters, key.upper(), context) for key, _ in OUTPUTS}

            def progress(f):
                if feedback.isCanceled():
                    raise QgsProcessingException("Cancelled")
                feedback.setProgress(int(100 * f))

            written = pipeline.run(
                model, arrs, info, outputs,
                providers=PROVIDERS[self.parameterAsEnum(parameters, "BACKEND", context)][1],
                blend=BLENDS[self.parameterAsEnum(parameters, "BLEND", context)],
                prior=PRIORS[self.parameterAsEnum(parameters, "PRIOR", context)],
                n_samples=self.parameterAsInt(parameters, "SAMPLES", context),
                tta=self.parameterAsBool(parameters, "TTA", context),
                batch_size=self.parameterAsInt(parameters, "BATCH", context),
                progress=progress)
        except ImportError as e:
            raise QgsProcessingException(f"{e}. {INSTALL_HINT}")
        except (ValueError, KeyError, FileNotFoundError) as e:
            raise QgsProcessingException(str(e))
        feedback.pushInfo(f"ONNX Runtime providers used: {written.pop('providers')}")
        return {key.upper(): path for key, path in written.items()}
