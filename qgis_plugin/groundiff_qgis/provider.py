from qgis.core import QgsProcessingProvider

from .algorithm import InspectLasAlgorithm, PredictRastersAlgorithm, PredictTilesAlgorithm


class GrounDiffProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(PredictTilesAlgorithm())
        self.addAlgorithm(PredictRastersAlgorithm())
        self.addAlgorithm(InspectLasAlgorithm())

    def id(self):
        return "groundiff"

    def name(self):
        return "GrounDiff"

    def longName(self):
        return "GrounDiff DTM"
