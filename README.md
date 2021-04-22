# bci_project
A Brain-Computer Interface pipeline for video game camera control with EEG signals

### Python environment ##
- python 3.6 //
- numpy 1.19.2
- pyqt 5.9.2
- qt 5.9.7 (normally installed with pyqt)
- scikit learn 0.24.1
- vispy 0.5.3
- pylsl (install with pip, conda install not available)
- pickle

How to run
==
Setup the python environment <br />
Specify the CSV file path for the desired subject on neuropype node " import CSV " <br />
Run the gui_launch.py file  <br />
==> Launch the "video game" GUI, the camera can be controlled with 
the mouse pointer position. Use mouse click or mouse wheel to reset the
camera to the initial position 

### Run the neuropype pipeline ###
==> EEG and ground truth eye gaze data will be streamed to the python program <br />
The GUI will display a red square for the prediction, and a green one for the ground truth. <br />
Camera position will be automatically change according to the predicted eye gaze location. <br />
Camera can still be moved by the mouse. <br />
When prediction starts being display, you should reset the camera with a mouse click first. <br />

### Code description ###
The ML model trained on the jupyter notebook were saved and are loaded on the gui_launch.py file. <br />
The canvas_video_game.py file contains the video game canvas built with OpenGL. <br />
The gui_launch.py file contains the QT main GUI, and is used to handle multithreading : 
one for data reception and one for the GUI.
Once the streaming data are received, they are sent to the video game canvas that update 
the GUI camera and the predicted and ground truth eye-gazed positions.

The eye gaze positions are sent to the video game GUI approximately every 2 seconds, <br />
to smooth the visualization.
It is controlled by the self.wait attribute in the gui_launch.py file that is changable. <br />
If you want to try another subject or model change the filename variable in the gui_launch.py file 
and the file path on neuropype
