from PyQt6.QtWidgets import QApplication, QFileDialog
import sys
import numpy as np
import pyqtgraph as pg

app = QApplication(sys.argv)

p = pg.plot()
p.showGrid(x=True, y=True)

# fname = QFileDialog.getOpenFileName()[0]
ch1 = np.fromfile("ch1.bin", dtype=np.float32)
ch2 = np.fromfile("ch2.bin", dtype=np.float32)
ch3 = np.fromfile("ch3.bin", dtype=np.float32)
fch1 = np.fromfile("fch1.bin", dtype=np.float32)
fch2 = np.fromfile("fch2.bin", dtype=np.float32)
fch3 = np.fromfile("fch3.bin", dtype=np.float32)

p.plot(ch1[:500000], pen='g')
p.plot(fch1[:500000]-3, pen='y')


sys.exit(app.exec())