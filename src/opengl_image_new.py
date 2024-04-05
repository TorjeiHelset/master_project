from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import time as time
import numpy as np

# Global constants - texture, w1 and h1 are updated when the background is being created
texture = None
w1 = 0
h1 = 0
shift_const = 0.008 # relative to the [0,1]x[0,1] coordinate system
color_gradient = 0
arrow_length = 5

def reshape(w, h):
    # Global to update w1 and h1
    global w1
    global h1
    w1 = w
    h1 = h
    glViewport(0, 0, w, h)

def orthogonalStart():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-w1/2, w1/2, -h1/2, h1/2)
    glMatrixMode(GL_MODELVIEW)

def orthogonalEnd():
    glMatrixMode(GL_PROJECTION)
    glMatrixMode(GL_MODELVIEW)

def background():
    glColor3f(1.0, 1.0, 1.0)
    # Setting the global texture
    glPushMatrix()
    glEnable( GL_TEXTURE_2D )
    glBindTexture( GL_TEXTURE_2D, texture)
    orthogonalStart()
    # Adding the texture to the viewport
    iw = 800
    ih = 600
    glTranslatef( -iw/2, -ih/2, 0 )
    glBegin(GL_QUADS)
    glTexCoord2f(0,0)
    glVertex2f(0, 0)
    glTexCoord2f(1,0)
    glVertex2f(iw, 0)
    glTexCoord2f(1,1)
    glVertex2f(iw, ih)
    glTexCoord2f(0,1)
    glVertex2f(0, ih)
    glEnd()

    orthogonalEnd()
    glDisable( GL_TEXTURE_2D )
    glPopMatrix()

def draw_on_top(x, y):
    # Test function for drawing on top of background
    glPushMatrix()
    orthogonalStart()
    iw = 800
    ih = 600
    glTranslatef( -iw/2, -ih/2, 0 )

    # map 0,1 to 0,iw:
    x = iw * x
    y = ih * y
    x_shift = 0.05 * iw
    y_shift = 0.1 * iw

    glBegin(GL_LINES)
    glColor3f(0.0, 0.5, 0.5)
    glVertex2f(x, y)            # Arrow base
    glVertex2f(x + x_shift, y + y_shift) # Arrow tip
    glVertex2f(x, y)            # Arrow base
    glVertex2f(x - x_shift, y + y_shift) # Arrow tip
    glEnd()

    orthogonalEnd()
    glPopMatrix()

def display_fnc():
    # Test function for displaying only 
    glClearColor (0.0,0.0,0.0,0.0)
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    background()

    x_1, y_1 = [0.098, 0.92]
    x_2, y_2 = [0.015, 0.22]
    x_3, y_3 = [0.97, 0.22]
    x_4, y_4 = [0.97, 0.92]
    draw_on_top(x_1, y_1)
    draw_on_top(x_2, y_2)
    draw_on_top(x_3, y_3)
    draw_on_top(x_4, y_4)

    gluLookAt (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    glutSwapBuffers()

def load_texture(filename):
    img = Image.open(filename)
    img_data = img.tobytes("raw", "RGB", 0, -1)
    # print(img_data)
    # Testing with dummy data:
    # img_data = { 255,0,0, 0,255,0, 0,0,255, 255,255,255 }
    width, height = img.size

    texture = glGenTextures(1) # Extra argument here?
    glBindTexture( GL_TEXTURE_2D, texture )
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 )

    glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE )
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR) # Linear?
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR)
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,GL_RGB, GL_UNSIGNED_BYTE, img_data)
    return texture

def find_points_of_busses(network, busses, bus_lengths):
    # Get positions of busses at all times of the simulation
    # First find the position in the [-1,7]x[0,9] coordinate system
    try:
        x_shift = [[0 for _ in range(len(bus_lengths[i]))] for i in range(len(busses))]
        y_shift = [[0 for _ in range(len(bus_lengths[i]))] for i in range(len(busses))]
        positions = [[(None, None) for _ in range(len(bus_lengths[i]))] for i in range(len(busses))]
    except:
        x_shift = [[0 for _ in range(len(bus_lengths[str(i)]))] for i in range(len(busses))]
        y_shift = [[0 for _ in range(len(bus_lengths[str(i)]))] for i in range(len(busses))]
        positions = [[(None, None) for _ in range(len(bus_lengths[str(i)]))] for i in range(len(busses))]
    for i, lengths in enumerate(bus_lengths):
        # print(len(lengths))
        for j, length in enumerate(lengths):
            # Find the road the bus is on
            road_id, length_travelled = busses[i].get_road_id_at_length(float(length))
            # print(road_id, length_travelled)
            if road_id == "":
                # Bus has reached the end of the route -> don't draw the bus
                # The bus will not reenter the route, so the for loop can be broken
                break
            
            # Find the road the bus is on
            _, road = network.get_road(road_id)
            left = road.left_pos
            right = road.right_pos

            if road_id[-2:] == 'fw':
                if left[0] < right[0]:
                    # Road goes from left to right
                    # Shift road down
                    y_shift[i][j] = - shift_const
                else:
                    # Road goes from top to bottom
                    # Shift road to the left
                    x_shift[i][j] = - shift_const
            elif road_id[-2:] == 'bw':
                # print(f"road id: {road_id}")
                if left[0] > right[0]:
                    # Road goes from right to the left
                    # Shift road up
                    y_shift[i][j] = shift_const
                else:
                    # Road goes from bottom to up
                    # Shift road to the right
                    x_shift[i][j] = shift_const
                # print(f"Shift on road: {x_shift[i][j], y_shift[i][j]}")

            # The bus has travelled length/road.L of the road
            relative_length = length_travelled / road.L # Going from 0 to b
            # 0 -> left, b -> right
            x = 1/road.b * ((road.b - relative_length) * left[0] +  relative_length * right[0])
            y = 1/road.b * ((road.b - relative_length) * left[1] +  relative_length * right[1])
            positions[i][j] = (x, y)

    return positions, x_shift, y_shift

def convert_bus_positions(network, bus_positions, x_shift, y_shift):
    # Bus positions are now in the [-1,7]x[0,9] coordinate system
    # Change these to the [0,1]x[0,1] coordinate system

    # x = -0.5 should map to x = 0.01
    # x = 7 should map to x = 0.95
    # y = 0 should map to y = 0.92
    # y = 9 should map to y = 0.15

    # -0.5*a_x + b_x = 0.01 -> a_x = 2*b_x - 2*0.01
    # 7 * a_x + b_x = 0.95 -> 15 * b_x - 14*0.01 = 0.95
    # -> b_x = (0.95 + 14*0.01) / 15
    # -> a_x = 2*(0.95 + 14*0.01) / 15 - 2*0.01
    left_x = 0.025
    right_x = 0.99
    b_x = (right_x + 14*left_x) / 15
    a_x = 2*(right_x + 14*left_x) / 15 - 2*left_x

    # b_y = 0.92
    # 9 * a_y + b_y = 0.15
    # -> a_y = (0.15 - b_y) / 9
    # -> a_y = (0.15 - 0.92) / 9
    top_y = 0.92
    bottom_y = 0.12
    b_y = top_y
    a_y = (bottom_y - top_y) / 9

    points = [[None, None] for i in range(len(network.roads))]
    for i, road in enumerate(network.roads):
        left = road.left_pos
        right = road.right_pos
        points[i][0] = (a_x * left[0] + b_x, a_y * left[1] + b_y)
        points[i][1] = (a_x * right[0] + b_x, a_y * right[1] + b_y)

    bus_points = [[[None, None] for i in range(len(bus_positions[i]))] for i in range(len(bus_positions))]
    for i, bus_pos in enumerate(bus_positions):
        for j, pos in enumerate(bus_pos):
            # pos is a tuple (x, y)
            # Do shifting here
            try:
                bus_points[i][j] = (a_x * pos[0] + b_x + x_shift[i][j], a_y * pos[1] + b_y + y_shift[i][j])
            except:
                bus_points[i][j] = (None, None)
    
    return points, bus_points

def map_density_to_color(value):
    # Colors: 
    # (1.0, 0.0, 0.0) : Red
    # (1.0, 1.0, 0.0) : Yellow
    # (0.0, 1.0, 0.0) : Green
    # (0.0, 1.0, 1.0) : Cyan
    # (0.0, 0.0, 1.0) : Red

    value = max(0.0, min(1.0, value)) # Ensure between 0 and 1, if density comes from proper simulation not needed
    match color_gradient:
        case 0: # Black/White
            # Black/white scale:
            r = 1.0 - value
            g = 1.0 - value
            b = 1.0 - value
        case 1: # Blue/red with intermediate steps
            # Interpolate between blue and red
            # 0.0 -> blue, 
            # 0.25 -> cyan
            # 0.5 -> green
            # 0.75 -> yellow, 
            # 1.0 -> red
            if value <= 0.25:
                # Somewhere between blue and cyan
                r = 0
                g = 4 * value
                b = 1
            elif value <= 0.5:
                # Between cyan and green
                r = 0
                g = 1
                b = -4 * value + 2
            elif value <= 0.75:
                # Between green and yellow
                r = 4 * value - 2
                g = 1
                b = 0
            else:
                # Between yellow and red
                r = 1
                g = -4 * value + 4
                b = 0

        case 2: # Blue Red
            # 0.0 -> blue, 
            # 0.25 -> (0, 0.5, 0.5)
            # 0.5 -> green
            # 0.75 -> (0.5, 0.5, 0) 
            # 1.0 -> red
            if value <= 0.5:
                r = 0.0
                g = 2.0 * value
                b = 1.0 - 2.0 * value
            # Interpolate between yellow (0.5) and red (1.0)
            else:
                r = 2.0 * (value - 0.5)
                g = 1.0 - 2.0 * (value - 0.5)
                b = 0.0

        case 3: # Green/red
            # Interpolate between green and red
            # 0.0 -> green, 0.5 -> yellow, 1.0 -> red
            if value <= 0.5:
                # Somewhere between green and yellow
                r = 2.0 * value
                g = 1
                b = 0
            else:
                # Somewhere between yellow and red
                r = 1
                g = 2 - 2 * value
                b = 0

    return (float(r), float(g), float(b))

def shift_single_point(id, left, right):
    shifted_left = None
    shifted_right = None
    if id[-2:] == 'fw':
        # Shift road left or down depening on the direction
        # Find direction from x-values
        if right[0] > left[0]:
            # Road goes from left to the right
            # Shift road down
            shifted_left = (left[0], left[1] - shift_const)
            shifted_right = (right[0], right[1] - shift_const)
        else:
            # Road goes from top to bottom
            # Shift road to the left
            shifted_left = (left[0] - shift_const, left[1])
            shifted_right = (right[0] - shift_const, right[1])
    elif id[-2:] == 'bw':
        # Shift road right or up depending on the direction
        # Find direction from x-values
        if right[0] < left[0]:
            # Road goes from right to left
            # Shift road up
            shifted_left = (left[0], left[1] + shift_const)
            shifted_right = (right[0], right[1] + shift_const)
        else:
            # Road goes from top to bottom
            # Shift road to the right
            shifted_left = (left[0] + shift_const, left[1])
            shifted_right = (right[0] + shift_const, right[1])
    else:
        shifted_left = left
        shifted_right = right
    return shifted_left, shifted_right


def create_density_points_from_road_points(network, densities,
                                           road_points):
    colors = []
    points = []
    try:
        times = list(densities[0].keys())
    except:
        times = list(densities['0'].keys())

    for t in times:
        # Get densities at time t
        try:
            density = [densities[i][t] for i in range(len(densities))]
        except:
            density = [densities[str(i)][t] for i in range(len(densities))]

        # Transform densities to colors
        colors.append([None] * len(density))
        points.append([None] * len(density))
        for i, d in enumerate(density):
            # d is an array of densities -> convert to colors
            colors[-1][i] = [map_density_to_color(rho) for rho in d]
            road = network.roads[i]
            # Shifting of left and right points
            left, right = shift_single_point(road.id, road_points[i][0], road_points[i][1])
            n = len(colors[-1][i])
            x = np.linspace(left[0], right[0], n)
            y = np.linspace(left[1], right[1], n)
            points[-1][i] = [(x[i], y[i]) for i in range(len(x))]
    return colors, points

def draw_arrows_with_orientation(start, end, iw, ih, line_width=1):
    # Draws two red arrows in the orientation of the road
    arrow_color = (0.3, 0.3, 0.3)
    if start[0] < end[0]:
        # Left to right
        x1 = (start[0] + 0.3*(end[0]-start[0])) * iw
        y1 = (start[1]) * ih
        x2 = (start[0] + 0.7*(end[0]-start[0])) * iw
        y2 = (start[1]) * ih
        glColor3f(*arrow_color)
        glLineWidth(1)
        glBegin(GL_LINES)
        glVertex2f(x1, y1) # base
        glVertex2f(x1 - arrow_length, y1 + arrow_length/2) # first tip
        glVertex2f(x1, y1) # base
        glVertex2f(x1 - arrow_length, y1 - arrow_length/2) # second tip
        glEnd()

        glBegin(GL_LINES)
        glVertex2f(x2, y2) # base
        glVertex2f(x2 - arrow_length, y2 + arrow_length/2) # first tip
        glVertex2f(x2, y2) # base
        glVertex2f(x2 - arrow_length, y2 - arrow_length/2) # second tip
        glEnd()

    elif start[0] > end[0]:
        # Right to left
        x1 = (start[0] + 0.3*(end[0]-start[0])) * iw
        y1 = (start[1]) * ih
        x2 = (start[0] + 0.7*(end[0]-start[0])) * iw
        y2 = (start[1]) * ih
        glColor3f(*arrow_color)
        glLineWidth(1)
        glBegin(GL_LINES)
        glVertex2f(x1, y1) # base
        glVertex2f(x1 + arrow_length, y1 + arrow_length/2) # first tip
        glVertex2f(x1, y1) # base
        glVertex2f(x1 + arrow_length, y1 - arrow_length/2) # second tip
        glEnd()

        glBegin(GL_LINES)
        glVertex2f(x2, y2) # base
        glVertex2f(x2 + arrow_length, y2 + arrow_length/2) # first tip
        glVertex2f(x2, y2) # base
        glVertex2f(x2 + arrow_length, y2 - arrow_length/2) # second tip
        glEnd()

    elif start[1] < end[1]:
        # down to up
        x1 = (start[0]) * iw
        y1 = (start[1] + 0.3*(end[1]-start[1])) * ih
        x2 = (start[0]) * iw
        y2 = (start[1] + 0.7*(end[1]-start[1])) * ih
        glColor3f(*arrow_color)
        glLineWidth(1)
        glBegin(GL_LINES)
        glVertex2f(x1, y1) # base
        glVertex2f(x1 - arrow_length/2, y1 - arrow_length) # first tip
        glVertex2f(x1, y1) # base
        glVertex2f(x1 + arrow_length/2, y1 - arrow_length) # second tip
        glEnd()

        glBegin(GL_LINES)
        glVertex2f(x2, y2) # base
        glVertex2f(x2 - arrow_length/2, y2 - arrow_length) # first tip
        glVertex2f(x2, y2) # base
        glVertex2f(x2 + arrow_length/2, y2 - arrow_length) # second tip
        glEnd()

    else:
        # up to down
        x1 = (start[0]) * iw
        y1 = (start[1] + 0.3*(end[1]-start[1])) * ih
        x2 = (start[0]) * iw
        y2 = (start[1] + 0.7*(end[1]-start[1])) * ih
        glColor3f(*arrow_color)
        glLineWidth(1)
        glBegin(GL_LINES)
        glVertex2f(x1, y1) # base
        glVertex2f(x1 - arrow_length/2, y1 + arrow_length) # first tip
        glVertex2f(x1, y1) # base
        glVertex2f(x1 + arrow_length/2, y1 + arrow_length) # second tip
        glEnd()

        glBegin(GL_LINES)
        glVertex2f(x2, y2) # base
        glVertex2f(x2 - arrow_length/2, y2 + arrow_length) # first tip
        glVertex2f(x2, y2) # base
        glVertex2f(x2 + arrow_length/2, y2 + arrow_length) # second tip
        glEnd()

def draw_line_with_colors(colors, points, line_width, iw, ih, arrow=True):
    # Draw black border aroung roads
    glLineWidth(line_width + 2) # +2 also changing with iw/ih?
    glBegin(GL_LINE_STRIP)
    for color, point in zip(colors, points):
        # print(f"Trying to add point {point}")
        # print(f"With color {color}")
        glColor3f(0,0,0)
        x,y = point
        glVertex2f(x*iw, y*ih)
    glEnd()
    
    # Draw densities
    glLineWidth(line_width)
    glBegin(GL_LINE_STRIP)
    
    for color, point in zip(colors, points):
        glColor3f(*color)
        x, y = point
        glVertex2f(x*iw, y*ih)
    glEnd()

    start = points[0]
    end = points[-1]
    if arrow:
        # Draw red arrows, oriented in the direction of the road
        # Assume road either goes from left to right, right to left, up to down or down to up
        draw_arrows_with_orientation(start, end, iw, ih, line_width)

def add_colorbar():
    # # Add colorbar
    # Change this to be a function being called
    # glBegin(GL_LINE_STRIP)
    # match color_gradient:
    #     case 1:
    #         color_map_colors = [(0,0,1), (0,1,1), (0,1,0),
    #                             (1,1,0), (1,0,0)]
    #         color_map_points = [(1.8, -0.5), (1.8, -0.25), (1.8, 0.0),
    #                             (1.8, 0.25), (1.8, 0.5)]

    #         for c, p in zip(color_map_colors, color_map_points):
    #             glColor3f(*c)
    #             glVertex2f(*p)
    #     case 2:
    #         color_map_colors = [(0,0,1), (0,0.5,0.5), (0,1,0),
    #                             (0.5,0.5,0), (1,0,0)]
    #         color_map_points = [(1.8, -0.5), (1.8, -0.25), (1.8, 0.0),
    #                             (1.8, 0.25), (1.8, 0.5)]
    #         for c, p in zip(color_map_colors, color_map_points):
    #             glColor3f(*c)
    #             glVertex2f(*p)
    #     case 3:
    #         color_map_colors = [(1,0,0), (1,1,0), (0,1,0)]
    #         color_map_points = [(1.2, 0.5), (1.8, 0), (1.8, -0.5)]

    #         for c, p in zip(color_map_colors, color_map_points):
    #             glColor3f(*c)
    #             glVertex2f(*p)
    # glEnd()
    pass

def draw_colored_line(colors, points):
    glPushMatrix()
    orthogonalStart()
    iw = 800
    ih = 600
    glTranslatef( -iw/2, -ih/2, 0 )
    line_width = 6.0  # Adjust the line width as needed
    for i in range(len(colors)):
        draw_line_with_colors(colors[i], points[i], line_width, iw, ih)
    orthogonalEnd()
    glPopMatrix()

def draw_busses(bus_positions):
    glPushMatrix()
    orthogonalStart()
    iw = 800
    ih = 600
    glTranslatef( -iw/2, -ih/2, 0 )
    line_width = 5.0
    # Draw the busses
    for bus_position in bus_positions:
        if bus_position != (None, None):
            glPointSize(7.0)
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.0, 0.0)
            glVertex2f(bus_position[0]*iw, bus_position[1]*ih)
            glEnd()
    orthogonalEnd()
    glPopMatrix()

class BusDensityRenderer:
    def __init__(self, colors, road_points, bus_points, interval_seconds, output_name):
        '''
        Road poitns is a list of points defining the network. This is fixed for all times
        Bus points is a list of points defining the position of the bus. This changes for each time step
        '''
        self.colors = colors
        self.road_points = road_points
        self.bus_points = bus_points
        self.current_idx = 0
        self.interval_seconds = interval_seconds
        self.last_update_time = time.time()
        self.is_rendering = True
        self.images = []
        assert type(output_name) == str
        if not output_name.endswith('.gif'):
            output_name += '.gif'
        self.output_name = output_name

    def display(self):
        # Clear window
        glClearColor (0.0,0.0,0.0,0.0)
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # Draw background
        background()
        # Draw densities on road
        draw_colored_line(self.colors[self.current_idx], self.road_points[self.current_idx])
        # Draw busses
        draw_busses([points[self.current_idx] for points in self.bus_points])
        gluLookAt (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glutSwapBuffers()

        # Read the content of the framebuffer and save as an image
        data = glReadPixels(0, 0, 800, 600, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (800, 600), data)
        self.images.append(image.transpose(Image.FLIP_TOP_BOTTOM))

    def timer(self, value):
        if self.current_idx >= len(self.colors)-1:
            print("End of simulation reached!")
            self.is_rendering = False
            self.save_gif()
            glutLeaveMainLoop()
            return
        # Update the current element index
        self.current_idx += 1
        # Redraw the scene
        glutPostRedisplay()
        # Set the timer for the next update - change this to take into account
        # that time intervals of the simulation might change
        glutTimerFunc(int(self.interval_seconds * 1000), self.timer, 0)
        # Update the last update time
        self.last_update_time = time.time()
    
    def save_gif(self):
        if len(self.images) == 0:
            print("No images to save")
            return
        print("Saving GIF as:", self.output_name)
        self.images[0].save(self.output_name, save_all=True, append_images=self.images[1:], optimize=False, duration=int(self.interval_seconds * 1000), loop=0) 
        print("GIF saved.")


def main():
    global texture
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE)
    glutInitWindowSize(1600, 1200)
    glutCreateWindow(b"OpenGL Window")

    glutDisplayFunc(display_fnc)
    glutReshapeFunc(reshape)
    # texture = load_texture("kvadraturen_sat2.png")
    texture = load_texture("kvadraturen_simple2.png")
    glutMainLoop()

def draw_busses_w_densities(bus_network, busses, bus_lengths, densities, output_name='animation.gif',
                            background_img = 'kvadraturen_simple2.png', interval_seconds=0.05):
    # Road positions go from -1 to 7 in x direction and 0 to 9 in y direction
    # These need to be correctly mapped to the display window
    # Map from [-1,7]x[0,9] to [0+left_margin,1-right_margin]x[0+bottom_margin,1-top_margin] 
    # x = -1 should map to x = 0.015
    # x = 7 should map to x = 0.92
    # y = 0 should map to y = 0.92
    # y = 9 should map to y = 0.22
    # note that a high low y value of road position should map to a high pixel value in animation

    # Converting points to correct format
    try:
        times = list(bus_lengths[0].keys())
        lengths = [[float(bus_lengths[i][t]) for t in times] for i in range(len(busses))]

    except:
        times = list(bus_lengths['0'].keys())
        # print("Times:", times)
        lengths = [[float(bus_lengths[str(i)][t]) for t in times] for i in range(len(busses))]

    positions, x_shift, y_shift = find_points_of_busses(bus_network, busses, lengths)
    # print(lengths)
    road_points, bus_points = convert_bus_positions(bus_network, positions, x_shift, y_shift)
    colors, points = create_density_points_from_road_points(bus_network, densities, road_points)
    # print(bus_points[0])
    # Create object for displaying the simulation
    renderer = BusDensityRenderer(colors, points, bus_points, interval_seconds, output_name)

    # Initializing the window
    global texture
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"OpenGL Window")

    # Setting up callback functions for the animation
    glutDisplayFunc(renderer.display)
    glutReshapeFunc(reshape)
    texture = load_texture(background_img)
    glutTimerFunc(1000, renderer.timer, 0)
    glutMainLoop()


if __name__ == "__main__":
    scenario = 3
    match scenario:
        case 0:
            import json
            import bus
            import network as nw
            import generate_kvadraturen as gk

            print("Loading results...")
            f = open("notebooks/kvadraturen_results_minimal_1000_variable.json")
            data = json.load(f)
            f.close()
            densities = data[0]
            queues = data[1]
            bus_lengths = data[2]
            bus_delays = data[3]

            print("Creating network...")
            T = 1000
            network = gk.generate_kvadraturen_minimal_junctions(T)

            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_2bw", "tollbod_1bw", "v_strand_5bw", 
                    "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"]            
            stops_bw = [("tollbod_2bw", 50), ("tollbod_1bw", 90), ("tollbod_1bw", 230), ("v_strand_1bw", 25)]
            times_bw = [40, 130, 190, 250]
            bus_bw = bus.Bus(ids_bw, stops_bw, times_bw, network, id = "2", start_time = 0.0)
            ids_fw = ["v_strand_1fw", "h_w_2", "festning_3fw", "festning_4fw", "tollbod_2fw",
                    "elvegata_fw", "lundsbro_fw"]
            stops_fw = [("h_w_2", 130), ("festning_4fw", 40), ("tollbod_2fw", 25), 
                        ("tollbod_2fw", 260)]
            times_fw = [30, 110, 130, 230]
            bus_fw = bus.Bus(ids_fw, stops_fw, times_fw, network, id = "1")
            times_bw_2 = [240, 330, 390, 450]
            times_fw_2 = [530, 610, 630, 830]
            bus_bw_2 = bus.Bus(ids_bw, stops_bw, times_bw, network, id = "3", start_time = 200.0)
            bus_fw_2 = bus.Bus(ids_fw, stops_fw, times_fw, network, id = "4", start_time = 500.0)

            roads = network.roads
            junctions = network.junctions
            T = network.T
            # Don't store the densities 
            # bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw], store_densities = False)
            bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw, bus_fw_2, bus_bw_2], store_densities = True)

            print("Creating animation...")
            draw_busses_w_densities(bus_network, [bus_fw, bus_bw, bus_fw_2, bus_bw_2], bus_lengths,
                                    densities, output_name="test_minimal_1000_variable_map.gif",
                                    background_img="background_imgs/kvadraturen_simple2.png")
            # print(bus_lengths.keys())        
        case 1:
            import generate_kvadraturen as gk
            import bus
            import network as nw
            T = 1000
            network = gk.generate_kvadraturen_minimal_junctions(T)

            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_2bw", "tollbod_1bw", "v_strand_5bw", 
                    "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"]            
            stops_bw = [("tollbod_2bw", 50), ("tollbod_1bw", 90), ("tollbod_1bw", 230), ("v_strand_1bw", 25)]
            times_bw = [40, 130, 190, 250]
            bus_bw = bus.Bus(ids_bw, stops_bw, times_bw, network, id = "2", start_time = 0.0)
            ids_fw = ["v_strand_1fw", "h_w_2", "festning_3fw", "festning_4fw", "tollbod_2fw",
                    "elvegata_fw", "lundsbro_fw"]
            stops_fw = [("h_w_2", 130), ("festning_4fw", 40), ("tollbod_2fw", 25), 
                        ("tollbod_2fw", 260)]
            times_fw = [30, 110, 130, 230]
            bus_fw = bus.Bus(ids_fw, stops_fw, times_fw, network, id = "1")
            times_bw_2 = [240, 330, 390, 450]
            times_fw_2 = [530, 610, 630, 830]
            bus_bw_2 = bus.Bus(ids_bw, stops_bw, times_bw, network, id = "3", start_time = 200.0)
            bus_fw_2 = bus.Bus(ids_fw, stops_fw, times_fw, network, id = "4", start_time = 500.0)

            roads = network.roads
            junctions = network.junctions
            T = network.T
            # Don't store the densities 
            # bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw], store_densities = False)
            bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw, bus_fw_2, bus_bw_2], store_densities = True)

            min_x = np.inf
            min_y = np.inf
            max_x = -np.inf
            max_y = -np.inf
            for road in bus_network.roads:
                min_x = min(min_x, min(road.left_pos[0], road.right_pos[0]))
                min_y = min(min_y, min(road.left_pos[1], road.right_pos[1]))
                max_x = max(max_x, max(road.left_pos[0], road.right_pos[0]))
                max_y = max(max_y, max(road.left_pos[1], road.right_pos[1]))
            
            print(min_x, max_x)
            print(min_y, max_y)

        case 2:
            main()
        case 3:
            import json
            import bus
            import network as nw
            import generate_kvadraturen as gk

            print("Loading results...")
            f = open("notebooks/kvadraturen_roundabout.json")
            data = json.load(f)
            f.close()
            densities = data[0]
            queues = data[1]
            bus_lengths = data[2]
            bus_delays = data[3]

            print("Creating network...")
            T = 1000
            network = gk.generate_kvadraturen_w_roundabout(T)

            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_2bw", "tollbod_1bw", "v_strand_5bw", 
                    "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"]            
            stops_bw = [("tollbod_2bw", 50), ("tollbod_1bw", 90), ("tollbod_1bw", 230), ("v_strand_1bw", 25)]
            times_bw = [40, 130, 190, 250]
            bus_bw = bus.Bus(ids_bw, stops_bw, times_bw, network, id = "2", start_time = 0.0)
            ids_fw = ["v_strand_1fw", "h_w_2", "festning_3fw", "festning_4fw", "tollbod_2fw",
                    "elvegata_fw", "lundsbro_fw"]
            stops_fw = [("h_w_2", 130), ("festning_4fw", 40), ("tollbod_2fw", 25), 
                        ("tollbod_2fw", 260)]
            times_fw = [30, 110, 130, 230]
            bus_fw = bus.Bus(ids_fw, stops_fw, times_fw, network, id = "1")
            times_bw_2 = [240, 330, 390, 450]
            times_fw_2 = [530, 610, 630, 830]
            bus_bw_2 = bus.Bus(ids_bw, stops_bw, times_bw, network, id = "3", start_time = 200.0)
            bus_fw_2 = bus.Bus(ids_fw, stops_fw, times_fw, network, id = "4", start_time = 500.0)

            roads = network.roads
            junctions = network.junctions
            T = network.T
            roundabouts = network.roundabouts
            # Don't store the densities 
            # bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw], store_densities = False)
            bus_network = nw.RoadNetwork(roads, junctions, T, busses=[bus_fw, bus_bw, bus_fw_2, bus_bw_2], roundabouts=roundabouts,
                                         store_densities = True)

            print("Creating animation...")
            draw_busses_w_densities(bus_network, [bus_fw, bus_bw, bus_fw_2, bus_bw_2], bus_lengths,
                                    densities, output_name="roundabout.gif",
                                    background_img="background_imgs/kvadraturen_simple2.png")

        case 4:
            import json
            print("Loading results...")
            f = open("notebooks/kvadraturen_results_minimal_1000_variable.json")
            data = json.load(f)
            f.close()
            densities = data[0]
            queues = data[1]
            bus_lengths = data[2]
            bus_delays = data[3]
            positions = [[(None, None) for _ in range(len(bus_lengths[str(i)]))] for i in range(4)]

            for i, lengths in enumerate(bus_lengths):
                for j, length in enumerate(lengths):
                    positions[i][j] = (2.2, 2.2)

            print([len(bus_lengths[str(i)]) for i in range(4)])
