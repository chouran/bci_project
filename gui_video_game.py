""" Visualization of traveling through space.
"""

import time

import numpy as np

from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective


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

eye_vertex = """
attribute vec2 a_position ; 
void main {
vec2 pos = a_position ;
gl_PointSize = 1000.0 ;
gl_Position = (pos, 0, 0) ;
}
"""

eye_fragment = """
void main {
gl_FragColor = vec4(1.0, 0.0, 0.0, 0.0) ;
}
"""

N = 10000  # Number of stars
SIZE = 100
SPEED = 1.0  # time in seconds to go through one block
NBLOCKS = 10


class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, title='Spacy', keys='interactive',
                            size=(800, 600))

        self.program = gloo.Program(vertex, fragment)
        self.eye_program = gloo.Program(eye_vertex, eye_fragment)

        self.view = 1*np.eye(4, dtype=np.float32)
        self.model = 1*np.eye(4, dtype=np.float32)
        self.translate = 1

        # Rotation Camera
        self.sensitivity = 0.1
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
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_pixel_scale'] = self.pixel_scale

        # Set attributes
        self.program['a_position'] = np.zeros((N, 3), np.float32)
        self.program['a_offset'] = np.zeros((N, 1), np.float32)

        self.eye_program['a_position'] = (0, 0)

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
        width, height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)
        far = SIZE*(NBLOCKS-2)
        self.projection = perspective(25.0, width / float(height), 1.0, far)
        self.program['u_projection'] = self.projection

    def on_draw(self, event):
        # Set time offset. Factor runs from 1 to 0
        # the time offset goes from 0 to size
        factor = (self._timeout - time.time()) / SPEED
        self.program['u_time_offset'] = -(1-factor) * SIZE

        # Draw
        gloo.clear()
        self.program.draw('points')
        self.eye_program.draw('points')

        # Build new starts if the first block is fully behind us
        if factor < 0:
            self._generate_stars()

    def on_close(self, event):
        self.timer.stop()

    def on_mouse_wheel(self, event):
        self.translate -= event.delta[1]
        self.program['u_view'] = self.view

        self.yaw, self.pitch = 0, 0

        self.rot_y(self.yaw * np.pi / 180)
        self.rot_x(self.pitch * np.pi / 180)

        self.view = np.dot(self.rot_mat_y, self.rot_mat_x)
        self.program['u_view'] = self.view

        self.update()

    def update_camera(self, y_pred):
        x, y = y_pred[0, 0], y_pred[0, 1]
        x /= 1563
        y /= 1093
        x *= 800
        y *= 600
        print(x, y)

        #self.eye_program['a_position'] = (x/800, y/800)
        self.x_offset, self.y_offset = x - self.last_x, - (y - self.last_y)
        self.last_x, self.last_y = x, y
        self.x_offset *= self.sensitivity
        self.y_offset *= self.sensitivity

        self.yaw, self.pitch = self.yaw - self.x_offset, self.pitch + self.y_offset

        self.rot_y(self.yaw * np.pi / 180)
        self.rot_x(self.pitch * np.pi / 180)

        self.view = np.dot(self.rot_mat_y, self.rot_mat_x)
        # print(self.view)
        self.program['u_view'] = self.view

        self.update()

    # Camera rotation
    def on_mouse_move(self, event):
        #self.view = 1 * np.eye(4, dtype=np.float32)
        #self.model = 1 * np.eye(4, dtype=np.float32)

        #self.translate -= event.delta[1]
        #self.translate = max(-1, self.translate)
        #print(event.delta[1])
        #print(self.translate)
        #self.view = translate((0, 0, -self.translate))
        #self.program['u_view'] = self.view
        #self.program['u_size'] = 5 / self.translate
        #self.view = (0.1*self.translate*np.eye(4, dtype=np.float32)) + self.view
        # self.model = (0.1*self.translate*np.eye(4, dtype=np.float32)) + self.model
        #print(self.view)

        #self.program['u_model'] = self.model
        #self.program['u_view'] = self.view

        x, y = event.pos
        print(x, y)
        self.x_offset, self.y_offset = x - self.last_x, - (y - self.last_y)
        self.last_x, self.last_y = x, y
        self.x_offset *= self.sensitivity
        self.y_offset *= self.sensitivity

        self.yaw, self.pitch = self.yaw - self.x_offset, self.pitch + self.y_offset

        self.rot_y(self.yaw * np.pi / 180)
        self.rot_x(self.pitch * np.pi / 180)

        self.view = np.dot(self.rot_mat_y, self.rot_mat_x)
        #print(self.view)
        self.program['u_view'] = self.view

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
        pos[:, :2] = np.random.normal(0.0, SIZE/2., (blocksize, 2))  # x-y
        pos[:, 2] = np.random.uniform(0, SIZE, (blocksize,))  # z
        start_index = self._active_block * blocksize
        self.program['a_position'].set_subdata(pos, offset=start_index)

        #print(start_index)

        # Set offsets - active block gets offset 0
        for i in range(NBLOCKS):
            val = i - self._active_block
            if val < 0:
                val += NBLOCKS
            values = np.ones((blocksize, 1), 'float32') * val * SIZE
            start_index = i*blocksize
            self.program['a_offset'].set_subdata(values, offset=start_index)

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