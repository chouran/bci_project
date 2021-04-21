from canvas_video_game import Canvas
from pylsl import StreamInlet, resolve_stream

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
import pickle
import numpy as np

'''
The two following classes handle the multithreading.
The GUI and data reception run on two separate threads. 
'''

class WorkerSignals(QObject):
    '''
    Inherits from QObject to handle signals to update the Qt GUI.
    '''
    data = pyqtSignal(tuple)

class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    '''

    def __init__(self, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        #self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Receive the EEG data from Neuropype and transmit them
        to the QT Gui
        '''
        print('thread 1')

        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])
        while True:
            #print(inlet.pull_sample())
            y = inlet.pull_sample()
            #print(len(y[0]))
            self.signals.data.emit(y)



class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Project GUI")
        self._canvas = Canvas()

        filename = 'lr_eeg_subject_1.sav'
        self.loaded_model = pickle.load(open(filename, 'rb'))
        self.setCentralWidget(self._canvas.native)

        self.threadpool = QThreadPool()
        self.test_thread()

        self.received_data = 0

    def test_data(self, y):
        data = y[0]
        self.received_data += 1
        data_array = np.array(data).reshape((1, 61))
        #print(data_array.shape)
        y_pred = self.loaded_model.predict(data_array)
        print(self.received_data)

        # print(data)
        # print(data_array.shape)
        # print(y_pred)

        if self.received_data % 200 == 0:
            self._canvas.update_camera(y_pred)

    def test_thread(self):
        worker = Worker()
        worker.signals.data.connect(self.test_data)

        self.threadpool.start(worker)



app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
