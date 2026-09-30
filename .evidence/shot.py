import os, sys
os.environ["QT_QPA_PLATFORM"]="offscreen"
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox
import run_ui
app=QApplication([])
def shot():
    for w in app.topLevelWidgets():
        if isinstance(w,QMessageBox): w.grab().save(".evidence/error-dialog.png"); w.close()
def crit(p,t,m):
    b=QMessageBox(QMessageBox.Icon.Critical,t,m); QTimer.singleShot(100,shot); b.exec()
QMessageBox.critical=crit
sys.excepthook=run_ui.show_unhandled_exception
def boom(): raise RuntimeError("slot failed")
QTimer.singleShot(0,boom); QTimer.singleShot(1000,app.quit); app.exec()
