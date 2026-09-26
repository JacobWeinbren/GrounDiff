from qgis.core import QgsProcessingProvider

from .algorithm import PredictDtmAlgorithm


class GrounDiffProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(PredictDtmAlgorithm())

    def id(self):
        return "groundiff"

    def name(self):
        return "GrounDiff"

    def longName(self):
        return "GrounDiff DTM"
