from qgis.core import Qgis, QgsApplication

from .provider import GrounDiffProvider

ALG_ID = "groundiff:predict_dtm_tiles"


class GrounDiffPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.actions = []

    def initProcessing(self):
        self.provider = GrounDiffProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        from qgis.PyQt.QtWidgets import QAction

        self.initProcessing()
        run = QAction("Find edits in point clouds (GrounDiff)", self.iface.mainWindow())
        run.triggered.connect(self.open_tool)
        inst = QAction("Install / check GrounDiff components", self.iface.mainWindow())
        inst.triggered.connect(lambda: self.install(ask=False))
        self.iface.addToolBarIcon(run)
        for a in (run, inst):
            self.iface.addPluginToMenu("GrounDiff", a)
        self.actions = [run, inst]
        from . import deps
        if deps.missing():
            self._offer_install()

    def unload(self):
        for a in self.actions:
            self.iface.removePluginMenu("GrounDiff", a)
            self.iface.removeToolBarIcon(a)
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)

    def open_tool(self):
        from . import deps
        if deps.missing() and not self.install(ask=True):
            return
        import processing
        processing.execAlgorithmDialog(ALG_ID, {})

    def _offer_install(self):
        from qgis.PyQt.QtWidgets import QPushButton
        bar = self.iface.messageBar()
        msg = bar.createMessage("GrounDiff", "needs a few components (ONNX Runtime, laspy) before first use.")
        btn = QPushButton("Install now")
        btn.clicked.connect(lambda: (bar.popWidget(msg), self.install(ask=False)))
        msg.layout().addWidget(btn)
        bar.pushWidget(msg, Qgis.Warning)

    def install(self, ask: bool = True) -> bool:
        """Install missing packages with QGIS's own Python; True when all are present."""
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication, QMessageBox

        from . import deps
        todo = deps.missing()
        win = self.iface.mainWindow()
        if not todo:
            QMessageBox.information(win, "GrounDiff", "All components are installed.")
            return True
        if ask and QMessageBox.question(
                win, "GrounDiff", "GrounDiff needs these components, installed into your QGIS profile "
                "(about 100 MB, a minute or two):\n\n  " + "\n  ".join(todo) + "\n\nInstall now?") \
                != QMessageBox.Yes:
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            deps.install(todo, log=lambda m: QgsApplication.messageLog().logMessage(m, "GrounDiff"))
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(win, "GrounDiff", f"Installing failed:\n\n{e}\n\nSee View > Panels > Log "
                                 "Messages (GrounDiff) for details.")
            return False
        QApplication.restoreOverrideCursor()
        left = deps.missing()
        if left:
            QMessageBox.warning(win, "GrounDiff", "Installed, but these still cannot be loaded: " + ", ".join(left)
                                + ". Restart QGIS and try again.")
            return False
        self.iface.messageBar().pushSuccess("GrounDiff", "components installed")
        return True
