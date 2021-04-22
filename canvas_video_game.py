""" Visualization of traveling through space.
"""

import time
import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective

# Vertex shader for the stars
vertex = """
#version 120
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_time_offset;
uniform float u_pixel_scale;
attribute vec3  a_position;
attribute float a_offset;
varying float v_pointsize;
void main (void) {
    vec3 pos = a_position;
    pos.z = pos.z - a_offset - u_time_offset;
    vec4 v_eye_position = u_view * u_model * vec4(pos, 1.0);
    gl_Position = u_projection * v_eye_position;
    // stackoverflow.com/questions/8608844/...
    //  ... resizing-point-sprites-based-on-distance-from-the-camera
    float radius = 1;
    vec4 corner = vec4(radius, radius, v_eye_position.z, v_eye_position.w);
    vec4  proj_corner = u_projection * corner;
    gl_PointSize = 100.0 * u_pixel_scale * proj_corner.x / proj_corner.w;
    v_pointsize = gl_PointSize;
}
"""

# Fragment shader for the stars
fragment = """
#version 120
varying float v_pointsize;
void main()
{
    float x = 2.0*gl_PointCoord.x - 1.0;
    float y = 2.0*gl_PointCoord.y - 1.0;
    float a = 0.9 - (x*x + y*y);
    a = a * min(1.0, v_pointsize/1.5);
    gl_FragColor = vec4(1.0, 1.0, 1.0, a);
}
"""

# vertex shader for the predicted eye gaze
prediction_vertex = """
attribute vec2 a_position ; 
void main() {
    vec2 pos = a_position ;
    gl_Position = vec4(a_position, 0, 1.0) ;
    gl_PointSize = 10.0 ;
}
"""

# vertex shader for the ground truth eye gaze
ground_truth_vertex = """
attribute vec2 a_position ; 
void main() {
    vec2 pos = a_position ;
    gl_Position = vec4(a_position, 0, 1.0) ;
    gl_PointSize = 10.0 ;
}
"""

# Fragment shader for the predicted eye_gaze
prediction_fragment = """
void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0) ;
}
"""

# Fragment shader for the ground truth eye_gaze
ground_truth_fragment = """
void main() {
    gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0) ;
}
"""

N = 10000  # Number of stars
SIZE = 10
SPEED = 1.0  # time in seconds to go through one block
NBLOCKS = 10


class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, title='Spacy', keys='interactive',
                            size=(800, 600))

        # Init the window size
        self.width, self.height = 800, 600

        # Load the shader programs into the GPU
        self.game_program = gloo.Program(vertex, fragment)
        self.eye_predic_program = gloo.Program(prediction_vertex,
                                               prediction_fragment)
        self.ground_truth_program = gloo.Program(ground_truth_vertex,
                                                 ground_truth_fragment)

        self.view = 1 * np.eye(4, dtype=np.float32)
        self.model = 1 * np.eye(4, dtype=np.float32)
        self.translate = 1

        #  Camera parameters
        self.sensitivity_gaze = 1000
        self.sensitivity = 0.1
        self.x_pred, self.y_pred = 0, 0
        self.last_x, self.last_y = 0, 0
        self.x_offset, self.y_offset = 0, 0
        self.yaw = 0
        self.pitch = 0
        self.rot_mat_x = np.zeros((4, 4), dtype=np.float32)
        self.rot_mat_y = np.zeros((4, 4), dtype=np.float32)
        self.rot_mat_y[1][1], self.rot_mat_y[3][3] = 1, 1
        self.rot_mat_x[0][0], self.rot_mat_x[3][3] = 1, 1

        self.activate_zoom()

        self.timer = app.Timer('auto', connect=self.update, start=True)

        # Set uniforms (some are set later)
        self.game_program['u_model'] = self.model
        self.game_program['u_view'] = self.view
        self.game_program['u_pixel_scale'] = self.pixel_scale

        # Set attributes
        self.game_program['a_position'] = np.zeros((N, 3), np.float32)
        self.game_program['a_offset'] = np.zeros((N, 1), np.float32)
        self.eye_predic_program['a_position'] = np.zeros((1, 2), np.float32)
        self.ground_truth_program['a_position'] = np.zeros((1, 2), np.float32)

        # Init
        self._timeout = 0
        self._active_block = 0
        for i in range(NBLOCKS):
            self._generate_stars()
        self._timeout = time.time() + SPEED

        gloo.set_state(clear_color='black', depth_test=False,
                       blend=True, blend_equation='func_add',
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

    def on_resize(self, event):
        self.activate_zoom()

    def activate_zoom(self):
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)
        far = SIZE * (NBLOCKS - 2)
        self.projection = perspective(25.0, self.width / float(self.height), 1.0, far)
        self.game_program['u_projection'] = self.projection

        self.update()

    def on_draw(self, event):
        # Set time offset. Factor runs from 1 to 0
        # the time offset goes from 0 to size
        factor = (self._timeout - time.time()) / SPEED
        self.game_program['u_time_offset'] = -(1 - factor) * SIZE

        # Draw
        gloo.clear()
        self.game_program.draw('points')
        self.eye_predic_program.draw('points')
        self.ground_truth_program.draw('points')

        # Build new starts if the first block is fully behind us
        if factor < 0:
            self._generate_stars()

    def on_close(self, event):
        self.timer.stop()

    def on_mouse_press(self, event):
        """
        Reset the camera position when mouse buttons are triggered
        """
        self.on_mouse_wheel(event)

    def on_mouse_wheel(self, event):
        """
        Reset the camera position when mouse wheel is triggered
        """
        self.translate -= event.delta[1]
        self.game_program['u_view'] = self.view

        self.yaw, self.pitch = 0, 0

        self.rot_y(self.yaw * np.pi / 180)
        self.rot_x(self.pitch * np.pi / 180)

        self.view = np.dot(self.rot_mat_y, self.rot_mat_x)
        self.game_program['u_view'] = self.view

        self.update()

    def get_pred(self, pred):
        """
        Receive the eye gaze prediction and update the camera orientation
        """
        self.x_pred, self.y_pred = pred[0, 0], pred[0, 1]
        x_eye, y_eye = self.x_pred * 2 - 1, self.y_pred * 2 - 1
        self.display_pred(x_eye, y_eye)
        # x /= 1563
        # y /= 1093
        self.x_pred *= self.width
        self.y_pred *= self.height

        self.update_camera()

        # self.x_offset, self.y_offset = self.x_pred - self.last_x, - (self.y_pred - self.last_y)
        # self.last_x, self.last_y = self.x_pred, self.y_pred
        # self.x_offset *= self.sensitivity
        # self.y_offset *= self.sensitivity
        # self.yaw, self.pitch = self.yaw - self.x_offset, self.pitch + self.y_offset

        # Update the rotation and the View matrices
        # self.rot_y(self.yaw * np.pi / 180)
        # self.rot_x(self.pitch * np.pi / 180)
        # self.view = np.dot(self.rot_mat_y, self.rot_mat_x)
        # self.game_program['u_view'] = self.view.astype(np.float32)

        # self.update()

    def update_camera(self):
        x_list = np.linspace(self.last_x, self.x_pred, num=self.sensitivity_gaze)
        y_list = np.linspace(self.last_y, self.y_pred, num=self.sensitivity_gaze)

        for new_x, new_y in zip(x_list, y_list):
            self.x_offset, self.y_offset = new_x - self.last_x, - (new_y - self.last_y)
            self.last_x, self.last_y = new_x, new_y
            #print(self.last_x, self.last_y)
            self.x_offset *= self.sensitivity
            self.y_offset *= self.sensitivity
            self.yaw, self.pitch = self.yaw - self.x_offset, self.pitch + self.y_offset

            # Update the rotation and the View matrices
            self.rot_y(self.yaw * np.pi / 180)
            self.rot_x(self.pitch * np.pi / 180)
            self.view = np.dot(self.rot_mat_y, self.rot_mat_x)
            self.game_program['u_view'] = self.view.astype(np.float32)

            self.update()

    def display_pred(self, x, y):
        # Display the eye gaze position
        pred = np.array([[x, y]], np.float32)
        self.eye_predic_program['a_position'] = pred
        self.update()

    def display_gt(self, gt_eye):
        x_gt, y_gt = gt_eye[0]/1563, gt_eye[1]/1093
        x_gt, y_gt = x_gt * 2 - 1, y_gt * 2 - 1
        gt = np.array([[x_gt, y_gt]], np.float32)

        self.ground_truth_program['a_position'] = gt
        self.update()

    # Camera rotation with mouse movements
    def on_mouse_move(self, event):
        """
        Receive the eye gaze prediction and update the camera orientation
        """

        # self.view = 1 * np.eye(4, dtype=np.float32)
        # self.model = 1 * np.eye(4, dtype=np.float32)

        # self.translate -= event.delta[1]
        # self.translate = max(-1, self.translate)
        # print(event.delta[1])
        # print(self.translate)
        # self.view = translate((0, 0, -self.translate))
        # self.game_program['u_view'] = self.view
        # self.game_program['u_size'] = 5 / self.translate
        # self.view = (0.1*self.translate*np.eye(4, dtype=np.float32)) + self.view
        # self.model = (0.1*self.translate*np.eye(4, dtype=np.float32)) + self.model
        # print(self.view)

        # self.game_program['u_model'] = self.model
        # self.game_program['u_view'] = self.view

        x, y = event.pos
        #print(x, y)
        self.x_offset, self.y_offset = x - self.last_x, - (y - self.last_y)
        self.last_x, self.last_y = x, y
        self.x_offset *= self.sensitivity
        self.y_offset *= self.sensitivity

        self.yaw, self.pitch = self.yaw - self.x_offset, self.pitch + self.y_offset
        self.rot_y(self.yaw * np.pi / 180)
        self.rot_x(self.pitch * np.pi / 180)

        self.view = np.dot(self.rot_mat_y, self.rot_mat_x)
        self.game_program['u_view'] = self.view

        self.update()

    def _generate_stars(self):

        # Get number of stars in each block
        blocksize = N // NBLOCKS

        # Update active block
        self._active_block += 1
        if self._active_block >= NBLOCKS:
            self._active_block = 0

        # Create new position data for the active block
        pos = np.zeros((blocksize, 3), 'float32')
        pos[:, :2] = np.random.normal(0.0, SIZE / 2., (blocksize, 2))  # x-y
        pos[:, 2] = np.random.uniform(0, SIZE, (blocksize,))  # z
        start_index = self._active_block * blocksize
        self.game_program['a_position'].set_subdata(pos, offset=start_index)

        # print(start_index)

        # Set offsets - active block gets offset 0
        for i in range(NBLOCKS):
            val = i - self._active_block
            if val < 0:
                val += NBLOCKS
            values = np.ones((blocksize, 1), 'float32') * val * SIZE
            start_index = i * blocksize
            self.game_program['a_offset'].set_subdata(values, offset=start_index)

        # Reset timer
        self._timeout += SPEED

    def rot_y(self, theta):
        self.rot_mat_y[0][0] = self.rot_mat_y[2][2] = np.cos(theta)
        self.rot_mat_y[0][2], self.rot_mat_y[2][0] = np.sin(theta), - np.sin(theta)

    def rot_x(self, theta):
        self.rot_mat_x[1][1] = self.rot_mat_x[2][2] = np.cos(theta)
        self.rot_mat_x[2][1], self.rot_mat_x[1][2] = np.sin(theta), - np.sin(theta)


if __name__ == '__main__':
    c = Canvas()
    app.run()
