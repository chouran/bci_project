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
The GUI and data reception/processing run on two separate threads. 
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
        # self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Receive the EEG and ground truth eye gaze
        data from Neuropype and send them
        to the QT Gui
        '''

        print('Second thread start ')
        eeg_streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(eeg_streams[0])
        eye_stream = resolve_stream('type', 'Gaze')
        eye_inlet = StreamInlet(eye_stream[0])
        while True:
            # print(inlet.pull_sample())
            y_eeg = inlet.pull_sample()
            y_eye = eye_inlet.pull_sample()
            data_to_send = (y_eeg, y_eye)
            self.signals.data.emit(data_to_send)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load and bound the video game canvas
        # To the QT GUI
        self.setWindowTitle("Project GUI")
        self._canvas = Canvas()
        self.setCentralWidget(self._canvas.native)

        # Load the regression model for prediction
        filename = 'lr_eeg_subject_1.sav'
        self.loaded_model = pickle.load(open(filename, 'rb'))

        # Start the data thread
        self.threadpool = QThreadPool()
        self.data_thread()

        # Params
        self.nb_buffer = 0
        self.time_eeg = 0
        self.time_eye = 0

        # Control the timestamps to send the eye gaze
        # ground truths and prediction to the video game
        self.wait = 200

    def test_data(self, y):
        eeg, eye = y[0], y[1]
        eeg_data, self.time_eeg = eeg[0], eeg[1]
        eye_data, self.time_eye = eye[0], eye[1]

        # print to see if timestamps are synchronized (they are)
        # print(self.time_eeg, self.time_eeg)
        # print(self.nb_buffer)

        self.nb_buffer += 1
        eeg_array = np.array(eeg_data).reshape((1, 61))
        y_pred = self.loaded_model.predict(eeg_array)

        #print(y_pred)
        if self.nb_buffer % self.wait == 0:
            self._canvas.get_pred(y_pred)
            self._canvas.display_gt(eye_data)

    def data_thread(self):
        # Start second data thread
        worker = Worker()
        worker.signals.data.connect(self.test_data)
        self.threadpool.start(worker)


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
