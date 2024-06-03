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
y_shift_const = 0.008 # relative to the [0,1]x[0,1] coordinate system
x_shift_const = y_shift_const * 3/4
color_gradient = 1
arrow_length = 5
with_e18 = True

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
                    y_shift[i][j] = - y_shift_const
                else:
                    # Road goes from top to bottom
                    # Shift road to the left
                    x_shift[i][j] = - x_shift_const
            elif road_id[-2:] == 'bw':
                # print(f"road id: {road_id}")
                if left[0] > right[0]:
                    # Road goes from right to the left
                    # Shift road up
                    y_shift[i][j] = y_shift_const
                else:
                    # Road goes from bottom to up
                    # Shift road to the right
                    x_shift[i][j] = x_shift_const
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

    # b_y = 0.92
    # 9 * a_y + b_y = 0.15
    # -> a_y = (0.15 - b_y) / 9
    # -> a_y = (0.15 - 0.92) / 9

    if with_e18:
        left_x = 0.03
        right_x = 0.99
        top_y = 0.92
        bottom_y = 0.12
    else: 
        left_x = 0.175
        right_x = 0.99
        top_y = 0.78
        bottom_y = 0.1

    b_x = (right_x + 14*left_x) / 15
    a_x = 2*(right_x + 14*left_x) / 15 - 2*left_x
    
    
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
        case 1: # Red monochrome scale
            r = 1.0
            g = 1.0 - value
            b = 1.0 - value

        case 2: # Blue/red with intermediate steps
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

        case 3: # Blue Red
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

        case 4: # Green/red
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
            shifted_left = (left[0], left[1] - y_shift_const)
            shifted_right = (right[0], right[1] - y_shift_const)
        else:
            # Road goes from top to bottom
            # Shift road to the left
            shifted_left = (left[0] - x_shift_const, left[1])
            shifted_right = (right[0] - x_shift_const, right[1])
    elif id[-2:] == 'bw':
        # Shift road right or up depending on the direction
        # Find direction from x-values
        if right[0] < left[0]:
            # Road goes from right to left
            # Shift road up
            shifted_left = (left[0], left[1] + y_shift_const)
            shifted_right = (right[0], right[1] + y_shift_const)
        else:
            # Road goes from top to bottom
            # Shift road to the right
            shifted_left = (left[0] + x_shift_const, left[1])
            shifted_right = (right[0] + x_shift_const, right[1])
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

def create_density_points_from_road_points_w_arrows(network, densities,
                                           road_points):
    colors = []
    points = []
    arrows = []
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
            if t == times[0]:
                if road.id in ["h_w_1", "h_w_2", "tollbod_1bw"]:
                    arrows.append(True)
                else:
                    arrows.append(False)
            # Shifting of left and right points
            left, right = shift_single_point(road.id, road_points[i][0], road_points[i][1])

            n = len(colors[-1][i])
            x = np.linspace(left[0], right[0], n)
            y = np.linspace(left[1], right[1], n)
            points[-1][i] = [(x[i], y[i]) for i in range(len(x))]
    return colors, points, arrows

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
        # draw_line_with_colors(colors[i], points[i], line_width, iw, ih)
        draw_line_with_colors(colors[i], points[i], line_width, iw, ih,arrow=False)

    orthogonalEnd()
    glPopMatrix()

def draw_colored_line_w_arrows(colors, points, arrows):
    glPushMatrix()
    orthogonalStart()
    iw = 800
    ih = 600
    glTranslatef( -iw/2, -ih/2, 0 )
    line_width = 6.0  # Adjust the line width as needed
    for i in range(len(colors)):
        draw_line_with_colors(colors[i], points[i], line_width, iw, ih, arrow=arrows[i])
    orthogonalEnd()
    glPopMatrix()

def draw_busses(bus_positions, color = [1.0, 0.0, 0.0]):
    glPushMatrix()
    orthogonalStart()
    iw = 800
    ih = 600
    glTranslatef( -iw/2, -ih/2, 0 )
    line_width = 5.0
    # Draw the busses
    for bus_position in bus_positions:
        if bus_position != (None, None):
            glPointSize(9.0)
            glBegin(GL_POINTS)
            glColor3f(0.0, 0.0, 0.0)
            glVertex2f(bus_position[0]*iw, bus_position[1]*ih)
            glEnd()
            glPointSize(7.0)
            glBegin(GL_POINTS)
            glColor3f(*color)
            glVertex2f(bus_position[0]*iw, bus_position[1]*ih)
            glEnd()
    orthogonalEnd()
    glPopMatrix()

class BusDensityRenderer:
    def __init__(self, colors, road_points, bus_points, interval_seconds, output_name,
                 old_bus_points = [], arrows = None):
        '''
        Road poitns is a list of points defining the network. This is fixed for all times
        Bus points is a list of points defining the position of the bus. This changes for each time step
        '''
        self.colors = colors
        self.road_points = road_points
        self.bus_points = bus_points
        self.old_bus_points = old_bus_points
        self.current_idx = 0
        self.interval_seconds = interval_seconds
        self.last_update_time = time.time()
        self.is_rendering = True
        self.images = []
        assert type(output_name) == str
        if not output_name.endswith('.gif'):
            output_name += '.gif'
        self.output_name = output_name
        if arrows is None:
            arrows = [True for _ in range(len(road_points[0]))]
        self.arrows = arrows

    def display_comparing(self):
        # Clear window
        glClearColor (0.0,0.0,0.0,0.0)
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # Draw background
        background()
        # Draw densities on road
        draw_colored_line_w_arrows(self.colors[self.current_idx], self.road_points[self.current_idx],
                                   self.arrows)
        # Draw busses
        match color_gradient:
            case 0: # Black white scale
                # Pick red and blue busses
                color_1 = [0.0, 0.0, 1.0]
                color_2 = [1.0, 0.0, 0.0]
            case 1: # Red monochrome
                # yellow and blue busses
                color_1 = [0.0, 0.0, 1.0]
                color_2 = [0.0, 1.0, 0.0]
            case _: 
                # 2 gray scale busses
                color_1 = [0.2, 0.2, 0.2]
                color_2 = [0.8, 0.8, 0.8]

        draw_busses([points[self.current_idx] for points in self.bus_points], color_2)
        draw_busses([points[self.current_idx] for points in self.old_bus_points], color_1)

        gluLookAt (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glutSwapBuffers()

        # Read the content of the framebuffer and save as an image
        data = glReadPixels(0, 0, 800, 600, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (800, 600), data)
        self.images.append(image.transpose(Image.FLIP_TOP_BOTTOM))

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
        draw_busses([points[self.current_idx] for points in self.bus_points], [0.0, 0.0, 1.0])
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

def draw_busses_compare_w_opt(bus_network, busses, bus_lengths, densities, old_busses = [],
                              old_lengths = [], output_name='animation.gif',
                              background_img = 'kvadraturen_simple2.png',
                              interval_seconds=0.05):
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
        old_lengths = [[float(old_lengths[i][t]) for t in times] for i in range(len(old_busses))]

    except:
        times = list(bus_lengths['0'].keys())
        # print("Times:", times)
        lengths = [[float(bus_lengths[str(i)][t]) for t in times] for i in range(len(busses))]
        old_lengths = [[float(old_lengths[str(i)][t]) for t in times] for i in range(len(old_busses))]


    positions, x_shift, y_shift = find_points_of_busses(bus_network, busses, lengths)
    old_positions, old_x_shift, old_y_shift = find_points_of_busses(bus_network, old_busses, old_lengths)

    
    # print(lengths)
    road_points, bus_points = convert_bus_positions(bus_network, positions, x_shift, y_shift)
    _, old_bus_points = convert_bus_positions(bus_network, old_positions, old_x_shift, old_y_shift)
    # colors, points = create_density_points_from_road_points(bus_network, densities, road_points)
    colors, points, arrows = create_density_points_from_road_points_w_arrows(bus_network, densities, road_points)

    # print(bus_points[0])
    # Create object for displaying the simulation
    # renderer = BusDensityRenderer(colors, points, bus_points, interval_seconds, output_name,
    #                               old_bus_points)
    renderer = BusDensityRenderer(colors, points, bus_points, interval_seconds, output_name,
                                  old_bus_points, arrows)

    # Initializing the window
    global texture
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"OpenGL Window")

    # Setting up callback functions for the animation
    glutDisplayFunc(renderer.display_comparing)
    glutReshapeFunc(reshape)
    texture = load_texture(background_img)
    glutTimerFunc(1000, renderer.timer, 0)
    glutMainLoop()

def draw_densities(network, densities, output_name='animation.gif',
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
        times = list(densities[0].keys())

    except:
        times = list(densities['0'].keys())

    positions, x_shift, y_shift = find_points_of_busses(network, [], [])

    # print(lengths)
    road_points, bus_points = convert_bus_positions(network, positions, x_shift, y_shift)
    colors, points = create_density_points_from_road_points(network, densities, road_points)
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


def update_e18_bool():
    global with_e18
    with_e18 = False



n_speeds = []
last_speed_idx = 0
n_cycles = []
control_points = []
config = None

def update_nspeeds_ncycles_controls(speed_limits, cycle_times, new_control_points):
    global n_speeds
    global n_cycles
    global control_points
    global last_speed_idx

    n_speeds = []
    n_cycles = []
    speed_idx = 0
    for speeds in speed_limits:
        n_speeds.append(len(speeds))
        speed_idx += len(speeds)
    last_speed_idx = speed_idx

    for cycles in cycle_times:
        n_cycles.append(len(cycles))

    control_points = new_control_points

def update_config(config_data):
    global config
    config = config_data

def load_bus_network(network_file, config_file):
    '''
    Function for initializing a bus network modelling kvadraturen
    with initial speed limits and speed limits as specified in the file
    filename. The grid spacing is also specified in the file
    '''
    f = open(network_file)
    data = json.load(f)
    f.close()
    T = data["T"]
    N = data["N"]
    speed_limits = data["speed_limits"] # Nested list
    control_points = data["control_points"] # Nested list
    cycle_times = data["cycle_times"] # Nested list

    update_nspeeds_ncycles_controls(speed_limits, cycle_times, control_points)

    f = open(config_file)
    data = json.load(f)
    f.close()
    update_config(data)
    
    return T, N, speed_limits, cycle_times

def get_speeds_cycles_from_params(params):
    idx = 0
    speed_limits = []
    cycle_times = []

    for i in range(len(n_speeds)):
        speed_limits.append([])
        for j in range(n_speeds[i]):
            speed_limits[i].append(params[idx])
            idx += 1

    for i in range(len(n_cycles)):
        cycle_times.append([])
        for j in range(n_cycles[i]):
            cycle_times[i].append(params[idx])
            idx += 1

    return speed_limits, cycle_times
def create_network_from_params(T, N, params, track_grad = False):
    speed_limits, cycle_times = get_speeds_cycles_from_params(params)
    # bus_network = gk.generate_kvadraturen_roundabout_w_params(T, N, speed_limits, control_points, cycle_times,
    #                                                           track_grad=track_grad)
    bus_network = gk.generate_kvadraturen_from_config_e18(T, N, speed_limits, control_points,
                                                          cycle_times, config, track_grad=track_grad)
    return bus_network


if __name__ == "__main__":
    scenario = 20
    
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
                                    background_img="background_imgs/blurred_kvadraturen.png")
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
            f = open("notebooks/kvadraturen_roundabout_200_max_dens1.json")
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
                                    densities, output_name="roundabout_200_max_dens1.gif",
                                    background_img="background_imgs/blurred_kvadraturen.png")

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

        case 5:
            import json
            import bus
            import network as nw
            import generate_kvadraturen as gk

            print("Loading results...")
            f = open("results/kvadraturen_roundabout_750_internal_new.json")
            data = json.load(f)
            f.close()
            densities = data[0]
            queues = data[1]
            bus_lengths = data[2]
            bus_delays = data[3]
            T = 750 
            bus_network = gk.generate_kvadraturen_w_bus(T)

            draw_busses_w_densities(bus_network, bus_network.busses, bus_lengths,
                                    densities, output_name="roundabout_750_internal_new.gif",
                                    background_img="background_imgs/blurred_kvadraturen.png")
            
        case 6:
            import json
            import bus
            import network as nw
            import generate_kvadraturen as gk

            print("Loading results...")
            f = open("results/kvadraturen_500_temp_opt_internal.json")
            data = json.load(f)
            f.close()
            densities = data[0]
            queues = data[1]
            bus_lengths = data[2]
            bus_delays = data[3]

            bus_network = gk.generate_kvadraturen_w_bus(T = 500)
            bus_network.busses = [bus_network.busses[i] for i in range(3)]

            f = open("results/kvadraturen_500_orig_lengths.json")
            data = json.load(f)
            f.close()

            # bus_network_2 = gk.generate_kvadraturen_roundabout_w_params()
            bus_network_2 = gk.generate_kvadraturen_w_bus(T = 500)
            bus_network_2.busses = [bus_network_2.busses[i] for i in range(3)]


            old_busses = bus_network_2.busses
            old_lengths = data[0]


            # draw_busses_compare_w_opt(bus_network, [bus_network.busses[0]], [bus_lengths['0']],
            #                         densities, [old_busses[0]], [old_lengths['0']], output_name="comparing_500_w_stops_test.gif",
            #                         background_img="background_imgs/blurred_kvadraturen_w_stops.png")
            
            draw_busses_compare_w_opt(bus_network, bus_network.busses, bus_lengths,
                                    densities, old_busses, old_lengths, output_name="comparing_500_w_stops_test.gif",
                                    background_img="background_imgs/blurred_kvadraturen_w_stops.png")
            
        case 7:
            import json
            import bus
            import network as nw
            import generate_kvadraturen as gk

            # global with_e18

            # with_e18 = True
            update_e18_bool()

            print("Loading results...")
            f = open("results/test_w_e18.json")
            data = json.load(f)
            f.close()
            densities = data[0]
            queues = data[1]
            bus_lengths = data[2]
            bus_delays = data[3]
 
            bus_network = gk.generate_kvadraturen_w_e18_w_busses(T=500)

            draw_busses_w_densities(bus_network, bus_network.busses, bus_lengths,
                                    densities, output_name="test_e18.gif",
                                    background_img="background_imgs/background_e18_cropped.png")
            
        case 8:
            import json
            import road as rd
            import network as nw
            import traffic_lights as tl
            import junction as jn
            import initial_and_bc as ibc
            import torch

            print("Loading results...")
            f = open("examples_for_presentation/results/traffic_lights.json")
            data = json.load(f)
            f.close()
            densities = data

            L = 25
            N = 2
            b = 4
            init_fnc = lambda x : torch.ones_like(x) * 0.2
            boundary_fnc = ibc.boundary_conditions(1, max_dens=1, densities=torch.tensor([0.2]), time_jumps=[], 
                                                in_speed=torch.tensor(50.0/3.6), L = L)
            road_1 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=boundary_fnc,
                            left_pos=(0, 3), right_pos=(2.9, 3), id="road_1", max_dens=1)
            road_2 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=None,
                            left_pos=(3.1, 3), right_pos=(6, 3), id = "road_2", max_dens=1)

            roads = [road_1, road_2]

            # Creating the traffic light and the junction
            traffic_light = tl.TrafficLightContinous(True, [0], [1], [torch.tensor(30.0), torch.tensor(50.0)])
            junction = jn.Junction(roads, [0], [1], [[1.0]], [traffic_light], [])

            # Creating the network
            network = nw.RoadNetwork(roads, [junction], T = 100)

            draw_densities(network, densities, output_name="examples_for_presentation/gifs/traffic_lights.gif",
                           background_img="background_imgs/white_background.png", interval_seconds = 0.1)
            
        case 9:
            import json
            import road as rd
            import network as nw
            import traffic_lights as tl
            import junction as jn
            import initial_and_bc as ibc
            import torch

            print("Loading results...")
            f = open("examples_for_presentation/results/with_duty_to_gw.json")
            data = json.load(f)
            f.close()
            densities = data

            L = 25
            N = 2
            b = 4

            distribution = [[1.0, 0.0], [0.0, 1.0]]
            priorities = [[1,0],[1,0]]
            crossings = [[[], []],
                        [[], [(0,0)]]]

            init_fnc_1 = lambda x : torch.ones_like(x) * 0.6
            init_fnc_2 = lambda x : torch.ones_like(x) * 0.2

            boundary_fnc_1 = ibc.boundary_conditions(1, max_dens=1, densities=torch.tensor([0.6]), time_jumps=[], 
                                                    in_speed=torch.tensor(50.0/3.6), L = L)
            boundary_fnc_2 = ibc.boundary_conditions(1, max_dens=1, densities=torch.tensor([0.2]), time_jumps=[], 
                                                    in_speed=torch.tensor(50.0/3.6), L = L)

            # Main roads
            road_1 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=boundary_fnc_1,
                            left_pos=(0, 3), right_pos=(2.9, 3), id="road_1", max_dens=1)
            road_2 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=None,
                            left_pos=(3.1, 3), right_pos=(6, 3), id="road_2", max_dens=1)
            # Secondary roads
            road_3 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=boundary_fnc_2,
                            left_pos=(3, 0), right_pos=(3, 2.9), id="road_3", max_dens=1)
            road_4 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_2, boundary_fnc=None,
                            left_pos=(3, 3.1), right_pos=(3, 6), id="road_4", max_dens=1)
            roads = [road_1, road_2, road_3, road_4]

            # Creating the junction
            junction = jn.Junction(roads, [0,2], [1,3], distribution, [],  [], True, priorities, crossings)

            # Creating the network
            network = nw.RoadNetwork(roads, [junction], T = 100)

            draw_densities(network, densities, output_name="examples_for_presentation/gifs/with_duty_to_gw.gif",
                           background_img="background_imgs/white_background.png", interval_seconds = 0.1)
            
        case 10:
            import json
            import road as rd
            import network as nw
            import traffic_lights as tl
            import junction as jn
            import initial_and_bc as ibc
            import torch

            print("Loading results...")
            f = open("examples_for_presentation/results/without_duty_to_gw.json")
            data = json.load(f)
            f.close()
            densities = data

            L = 25
            N = 2
            b = 4

            distribution = [[1.0, 0.0], [0.0, 1.0]]
            priorities = [[1,0],[1,0]]
            crossings = [[[], []],
                        [[], [(0,0)]]]

            init_fnc_1 = lambda x : torch.ones_like(x) * 0.6
            init_fnc_2 = lambda x : torch.ones_like(x) * 0.6
            init_fnc_3 = lambda x : torch.ones_like(x) * 0.2

            boundary_fnc_1 = ibc.boundary_conditions(1, max_dens=1, densities=torch.tensor([0.6]), time_jumps=[], 
                                                    in_speed=torch.tensor(50.0/3.6), L = L)
            boundary_fnc_2 = ibc.boundary_conditions(1, max_dens=1, densities=torch.tensor([0.2]), time_jumps=[], 
                                                    in_speed=torch.tensor(50.0/3.6), L = L)

            # Main roads
            road_1 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=boundary_fnc_1,
                            left_pos=(0, 3), right_pos=(2.9, 3), id="road_1", max_dens=1)
            road_2 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=None,
                            left_pos=(3.1, 3), right_pos=(6, 3), id="road_2", max_dens=1)
            # Secondary roads
            road_3 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_2, boundary_fnc=boundary_fnc_2,
                            left_pos=(3, 0), right_pos=(3, 2.9), id="road_3", max_dens=1)
            road_4 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_3, boundary_fnc=None,
                            left_pos=(3, 3.1), right_pos=(3, 6), id="road_4", max_dens=1)
            roads = [road_1, road_2, road_3, road_4]

            # Creating the junction
            junction = jn.Junction(roads, [0,2], [1,3], distribution, [],  [], True, priorities, crossings)

            # Creating the network
            network = nw.RoadNetwork(roads, [junction], T = 100)

            draw_densities(network, densities, output_name="examples_for_presentation/gifs/without_duty_to_gw.gif",
                           background_img="background_imgs/white_background.png", interval_seconds = 0.1)

        case 11:
            import json
            import road as rd
            import network as nw
            import traffic_lights as tl
            import junction as jn
            import initial_and_bc as ibc
            import torch

            print("Loading results...")
            f = open("examples_for_presentation/results/with_and_without_duty_to_gw.json")
            data = json.load(f)
            f.close()
            densities = data

            L = 25
            N = 2
            b = 4

            distribution = [[1.0, 0.0], [0.0, 1.0]]
            priorities = [[1,0],[1,0]]
            crossings = [[[], []],
                        [[], [(0,0)]]]

            init_fnc_1 = lambda x : torch.ones_like(x) * 0.6
            init_fnc_2 = lambda x : torch.ones_like(x) * 0.2


            boundary_fnc_1 = ibc.boundary_conditions(1, max_dens=1, densities=torch.tensor([0.6]), time_jumps=[], 
                                                    in_speed=torch.tensor(50.0/3.6), L = L)
            boundary_fnc_2 = ibc.boundary_conditions(1, max_dens=1, densities=torch.tensor([0.2]), time_jumps=[], 
                                                    in_speed=torch.tensor(50.0/3.6), L = L)

            # Main roads
            road_1 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=boundary_fnc_1,
                            left_pos=(0, 3), right_pos=(1.3, 3), id="road_1", max_dens=1)
            road_2 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=None,
                            left_pos=(1.5, 3), right_pos=(2.8, 3), id="road_2", max_dens=1)
            # Secondary roads
            road_3 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=boundary_fnc_2,
                            left_pos=(1.4, 1), right_pos=(1.4, 2.9), id="road_3", max_dens=1)
            road_4 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_2, boundary_fnc=None,
                            left_pos=(1.4, 3.1), right_pos=(1.4, 5), id="road_4", max_dens=1)
            roads_1 = [road_1, road_2, road_3, road_4]

            # Creating the junction
            junction_1 = jn.Junction(roads_1, [0,2], [1,3], distribution, [],  [], True, priorities, crossings)


            # Main roads
            road_5 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=boundary_fnc_1,
                            left_pos=(3.2, 3), right_pos=(4.5, 3), id="road_5", max_dens=1)
            road_6 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=None,
                            left_pos=(4.7, 3), right_pos=(6, 3), id="road_6", max_dens=1)
            # Secondary roads
            road_7 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_1, boundary_fnc=boundary_fnc_2,
                            left_pos=(4.6, 1), right_pos=(4.6, 2.9), id="road_7", max_dens=1)
            road_8 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc_2, boundary_fnc=None,
                            left_pos=(4.6, 3.1), right_pos=(4.6, 5), id="road_8", max_dens=1)
            roads_2 = [road_5, road_6, road_7, road_8]

            # Creating the junction
            junction_2 = jn.Junction(roads_2, [0,2], [1,3], distribution, [],  [], False)

            # Creating the network
            network = nw.RoadNetwork(roads_1 + roads_2, [junction_1, junction_2], T = 100)

            draw_densities(network, densities, output_name="examples_for_presentation/gifs/with_and_without_duty_to_gw.gif",
                        background_img="background_imgs/white_background.png", interval_seconds = 0.1)
            
        case 12:
            import json
            import road as rd
            import network as nw
            import traffic_lights as tl
            import junction as jn
            import initial_and_bc as ibc
            import torch
            import bus

            print("Loading results...")
            f = open("examples_for_presentation/results/busses.json")
            data = json.load(f)
            f.close()
            densities = data[0]
            bus_lengths = data[1]

            L = 25
            N = 2
            b = 4
            init_fnc = lambda x : torch.ones_like(x) * 0.2
            boundary_fnc = ibc.boundary_conditions(1, max_dens=1, densities=torch.tensor([0.2]), time_jumps=[], 
                                                in_speed=torch.tensor(50.0/3.6), L = L)
            road_1 = rd.Road(b, L, N, [torch.tensor(40/3.6)], [], initial=init_fnc, boundary_fnc=boundary_fnc,
                            left_pos=(0, 3), right_pos=(2.9, 3), id="road_1", max_dens=1)
            road_2 = rd.Road(b, L, N, [torch.tensor(40/3.6)], [], initial=init_fnc, boundary_fnc=None,
                            left_pos=(3.1, 3), right_pos=(6, 3), id = "road_2", max_dens=1)

            road_3 = rd.Road(b, L, N, [torch.tensor(60/3.6)], [], initial=init_fnc, boundary_fnc=boundary_fnc,
                            left_pos=(0, 6), right_pos=(2.9, 6), id="road_3", max_dens=1)
            road_4 = rd.Road(b, L, N, [torch.tensor(60/3.6)], [], initial=init_fnc, boundary_fnc=None,
                            left_pos=(3.1, 6), right_pos=(6, 6), id = "road_4", max_dens=1)

            roads = [road_1, road_2, road_3, road_4]

            # Creating the traffic light and the junction
            junction_1 = jn.Junction([road_1, road_2], [0], [1], [[1.0]], [], [])
            junction_2 = jn.Junction([road_3, road_4], [0], [1], [[1.0]], [], [])


            # Creating the network
            network = nw.RoadNetwork(roads, [junction_1, junction_2], T = 100)

            # Creating the busses
            bus_1_ids = ["road_1", "road_2"]
            bus_1_stops = [("road_2", 50)]
            bus_1_times = [0]
            bus_1 = bus.Bus(bus_1_ids, bus_1_stops, bus_1_times, network, id = "bus_1")

            bus_2_ids = ["road_3", "road_4"]
            bus_2_stops = [("road_4", 50)]
            bus_2_times = [0]
            bus_2 = bus.Bus(bus_2_ids, bus_2_stops, bus_2_times, network, id = "bus_2")

            bus_network = nw.RoadNetwork(roads, [junction_1, junction_2], T=100, roundabouts=[], busses=[bus_1, bus_2])


            draw_busses_w_densities(bus_network, bus_network.busses, bus_lengths, densities,
                                    output_name="examples_for_presentation/gifs/busses.gif",
                                    background_img="background_imgs/white_background_w_stops.png",interval_seconds = 0.1)
        
        case 13:
            import json
            import road as rd
            import network as nw
            import traffic_lights as tl
            import junction as jn
            import initial_and_bc as ibc
            import torch
            import bus
            import roundabout as rb

            print("Loading results...")
            f = open("examples_for_presentation/results/roundabout.json")
            data = json.load(f)
            f.close()
            densities = data[0]
            bus_lengths = data[1]

            # Creating the roads
            L = 25
            N = 2
            b = 4
            init_fnc = lambda x : torch.ones_like(x) * 0.2
            boundary_fnc = ibc.boundary_conditions(1, max_dens=1, densities=torch.tensor([0.2]), time_jumps=[], 
                                                in_speed=torch.tensor(50.0/3.6), L = L)
            # Roads outside roundabout:
            hori_fw1 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=boundary_fnc,
                            left_pos=(0, 3), right_pos=(1.9, 3), id="hori_1fw", max_dens=1)
            hori_bw1 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=None,
                            left_pos=(1.9, 3), right_pos=(0, 3), id="hori_1bw", max_dens=1)
            hori_fw2 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=None,
                            left_pos=(4.1, 3), right_pos=(6, 3), id="hori_2fw", max_dens=1)
            hori_bw2 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=boundary_fnc,
                            left_pos=(6, 3), right_pos=(4.1, 3), id="hori_2bw", max_dens=1)

            vert_fw1 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=boundary_fnc,
                            left_pos=(3, -1), right_pos=(3, 1.4), id="vert_1fw", max_dens=1)
            vert_bw1 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=None,
                            left_pos=(3, 1.4), right_pos=(3, -1), id="vert_1bw", max_dens=1)
            vert_fw2 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=None,
                            left_pos=(3, 4.6), right_pos=(3, 7), id="vert_2fw", max_dens=1)
            vert_bw2 = rd.Road(b, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=boundary_fnc,
                            left_pos=(3, 7), right_pos=(3, 4.6), id="vert_2bw", max_dens=1)

            # Roundabout roads:
            mainline_1 = rd.Road(2, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=None,
                                left_pos=(2,3), right_pos=(3,4.5), id="mainline_1")
            mainline_2 = rd.Road(2, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=None,
                                left_pos=(3,4.5), right_pos=(4,3), id="mainline_2")
            mainline_3 = rd.Road(2, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=None,
                                left_pos=(4,3), right_pos=(3,1.5), id="mainline_3")
            mainline_4 = rd.Road(2, L, N, [torch.tensor(50/3.6)], [], initial=init_fnc, boundary_fnc=None,
                                left_pos=(3,1.5), right_pos=(2,3), id="mainline_4")

            # Roundabout junctions
            junction_1 = rb.RoundaboutJunction(mainline_4, mainline_1, 0.6, hori_fw1, hori_bw1, queue_junction=False)
            junction_2 = rb.RoundaboutJunction(mainline_1, mainline_2, 0.6, vert_bw2, vert_fw2, queue_junction=False)
            junction_3 = rb.RoundaboutJunction(mainline_2, mainline_3, 0.6, hori_bw2, hori_fw2, queue_junction=False)
            junction_4 = rb.RoundaboutJunction(mainline_3, mainline_4, 0.6, vert_fw1, vert_bw1, queue_junction=False)

            # Roundabout
            roundabout = rb.Roundabout([mainline_1, mainline_2, mainline_3, mainline_4],
                                    [hori_fw1, vert_bw2, hori_bw2, vert_fw1],
                                    [hori_bw1, vert_fw2, hori_fw2, vert_bw1],
                                    [junction_1, junction_2, junction_3, junction_4])

            # Create temp network
            roads = [hori_fw1, hori_bw1, hori_fw2, hori_bw2, vert_fw1, vert_bw1, vert_fw2, vert_bw2,
                    mainline_1, mainline_2, mainline_3, mainline_4]
            network = nw.RoadNetwork(roads, [], T = 100, roundabouts=[roundabout])

            # Busline
            bus_ids = ["hori_1fw", "mainline_1", "mainline_2", "hori_2fw"]
            bus_stops = [("hori_2fw", 50)]
            bus_times = [0]
            bus_1 = bus.Bus(bus_ids, bus_stops, bus_times, network, id = "bus_1")

            # Create full network
            bus_network = nw.RoadNetwork(roads, [], T=100, roundabouts=[roundabout], busses=[bus_1])

            draw_busses_w_densities(bus_network, bus_network.busses, bus_lengths, densities,
                                    output_name="examples_for_presentation/gifs/roundabout.gif",
                                    background_img="background_imgs/white_background.png",interval_seconds = 0.1)
            
        case 14:
            import json
            import generate_general_networks as generate

            network = generate.compare_grid_size_network(T = 100, N = 2)

            print("Loading results...")
            f = open("results/comparing_grids_N=2.json")
            data = json.load(f)
            f.close()
            densities = data[0]


            print("Creating animation...")
            draw_densities(network, densities, output_name="gifs/comparing_grids_N=2.gif", 
                           background_img="background_imgs/white_background.png", interval_seconds = 0.1)
            
        case 15:
            import json
            import json
            import road as rd
            import network as nw
            import traffic_lights as tl
            import junction as jn
            import initial_and_bc as ibc
            import torch
            import bus

            print("Loading results...")
            f = open("results/test_bus_stopping.json")
            data = json.load(f)
            f.close()
            densities = data[0]
            bus_lengths = data[2]

            T = 100
            road_1 = rd.Road(1, 50, 5, torch.tensor([50.0/3.6], requires_grad=True), [],
                        initial=lambda x : torch.ones_like(x) * 0.2,
                        left_pos=(-1, 2), right_pos=(2.9,2),
                        periodic=True, id = "road_1_fw")

            road_2 = rd.Road(1, 50, 5, torch.tensor([50.0/3.6], requires_grad=True), [],
                        initial=lambda x : torch.ones_like(x) * 0.2,
                        left_pos=(3.1, 2), right_pos=(7,2),
                        periodic=True, id = "road_2_fw")

            road_3 = rd.Road(1, 50, 5, torch.tensor([50.0/3.6], requires_grad=True), [],
                        initial=lambda x : torch.ones_like(x) * 0.4,
                        left_pos=(-1, 4), right_pos=(2.9,4),
                        periodic=True, id = "road_3_fw")

            road_4 = rd.Road(1, 50, 5, torch.tensor([50.0/3.6], requires_grad=True), [],
                        initial=lambda x : torch.ones_like(x) * 0.4,
                        left_pos=(3.1, 4), right_pos=(7,4),
                        periodic=True, id = "road_4_fw")

            road_5 = rd.Road(1, 50, 5, torch.tensor([50.0/3.6], requires_grad=True), [],
                        initial=lambda x : torch.ones_like(x) * 0.6,
                        left_pos=(-1, 6), right_pos=(2.9,6),
                        periodic=True, id = "road_5_fw")

            road_6 = rd.Road(1, 50, 5, torch.tensor([50.0/3.6], requires_grad=True), [],
                        initial=lambda x : torch.ones_like(x) * 0.2,
                        left_pos=(3.1, 6), right_pos=(7,6),
                        periodic=True, id = "road_6_fw")

            traffic_light_2 = tl.TrafficLightContinous(True, [0], [1], [torch.tensor(30.),torch.tensor(30.)])
            traffic_light_3 = tl.TrafficLightContinous(True, [0], [1], [torch.tensor(30.),torch.tensor(30.)])
            traffic_light_1 = tl.TrafficLightContinous(True, [0], [1], [torch.tensor(30.),torch.tensor(30.)])
            junction_1 = jn.Junction([road_1, road_2], [0], [1], [[1.0]], [traffic_light_1], [])
            junction_2 = jn.Junction([road_3, road_4], [0], [1], [[1.0]], [traffic_light_2], [])
            junction_3 = jn.Junction([road_5, road_6], [0], [1], [[1.0]], [traffic_light_3], [])


            roads = [road_1, road_2, road_3, road_4, road_5, road_6]
            junctions = [junction_1, junction_2, junction_3]
            network = nw.RoadNetwork(roads, [], T)

            ids = ["road_1_fw", "road_2_fw"]
            stops = [("road_2_fw", 40)]
            times = [60]
            bus_1 = bus.Bus(ids, stops, times, network, id="bus1")

            ids = ["road_3_fw", "road_4_fw"]
            stops = [("road_4_fw", 40)]
            times = [60]
            bus_2 = bus.Bus(ids, stops, times, network, id="bus2")

            ids = ["road_5_fw", "road_6_fw"]
            stops = [("road_6_fw", 40)]
            times = [60]
            bus_3 = bus.Bus(ids, stops, times, network, id="bus3")


            bus_network = nw.RoadNetwork(roads, junctions, T, busses=[bus_1, bus_2, bus_3])


            draw_busses_w_densities(bus_network, bus_network.busses, bus_lengths, densities,
                                    output_name="gifs/test_bus_stopping.gif",
                                    background_img="background_imgs/white_background.png",interval_seconds = 0.1)
            
        case 16:
            import json
            import torch
            import generate_general_networks as generate

            # Compare optimal and non-optimal solutions on a single lane
            f = open("optimization_results/general_optimization/single_lane.json")
            results = json.load(f)
            f.close()

            network_file = results['network_file']
            f = open(network_file)
            network_config = json.load(f)
            f.close()

            T = network_config['T']
            N = network_config['N']
            controls = network_config['control_points'][0]

            # Collecting the start and final parameters
            start = results['parameters'][0]
            end = results['parameters'][-1]

            # Create the networks
            start_speed = [torch.tensor(v) for v in start]
            end_speed = [torch.tensor(v) for v in end]

            start_network = generate.single_lane_network(T, N, start_speed, controls, track_grad=False)
            end_network = generate.single_lane_network(T, N, end_speed, controls, track_grad=False)

            # Update positions of roads:
            start_network.roads[0].left_pos = (-1, 3)
            start_network.roads[0].right_pos = (7, 3)

            end_network.roads[0].left_pos = (-1, 3)
            end_network.roads[0].right_pos = (7, 3)

            # Load densities
            f = open("general_densities/single_lane_start_opt_times.json")
            data = json.load(f)
            f.close()
            orig_densities = data[0]
            orig_lengths = data[1]

            f = open("general_densities/single_lane_optimal.json")
            data_opt = json.load(f)
            f.close()
            opt_densities = data_opt[0]
            opt_lengths = data_opt[1]

            # Create gif:
            draw_busses_compare_w_opt(end_network, end_network.busses, opt_lengths,
                                    opt_densities, start_network.busses, orig_lengths, output_name="general_densities/videos/single_lane.gif",
                                    background_img="background_imgs/white_background.png")

        case 17:
            # Compare optimal and non-optimal solutions on a single junction
            import json
            import torch
            import generate_general_networks as generate

            f = open("optimization_results/general_optimization/single_junction_1.json")
            results = json.load(f)
            f.close()

            network_file = results['network_file']
            f = open(network_file)
            network_config = json.load(f)
            f.close()

            T = network_config['T']
            N = network_config['N']
            controls = network_config['control_points']

            # Collecting the start and final parameters
            start = results['parameters'][0]
            opt = results['parameters'][-1]

            # Create the networks
            start_speed = [[torch.tensor(start[i])] for i in range(2)]
            opt_speed = [[torch.tensor(opt[i])] for i in range(2)]
            start_cycle = [torch.tensor(start[2]), torch.tensor(start[3])]
            opt_cycle = [torch.tensor(opt[2]), torch.tensor(opt[3])]

            start_network = generate.single_junction_network(T, N, start_speed, controls, start_cycle, track_grad=False)
            opt_network = generate.single_junction_network(T, N, opt_speed, controls, opt_cycle, track_grad=False)

            # Update positions of roads:
            start_network.roads[0].left_pos = (-1, 3)
            start_network.roads[0].right_pos = (2.9, 3)
            start_network.roads[1].left_pos = (3.1, 3)
            start_network.roads[1].right_pos = (7, 3)

            opt_network.roads[0].left_pos = (-1, 3)
            opt_network.roads[0].right_pos = (2.9, 3)
            opt_network.roads[1].left_pos = (3.1, 3)
            opt_network.roads[1].right_pos = (7, 3)

            # Load densities
            f = open("general_densities/single_junction_start_opt_times.json")
            data = json.load(f)
            f.close()
            orig_densities = data[0]
            orig_lengths = data[1]

            f = open("general_densities/single_junction_optimal.json")
            data_opt = json.load(f)
            f.close()
            opt_densities = data_opt[0]
            opt_lengths = data_opt[1]

            # Create gif:
            draw_busses_compare_w_opt(opt_network, opt_network.busses, opt_lengths,
                                    opt_densities, start_network.busses, orig_lengths, output_name="general_densities/videos/single_junction.gif",
                                    background_img="background_imgs/white_background.png")
        
        case 18:
            # Compare optimal and non-optimal solutions on a 2-2 junction
            import json
            import torch
            import generate_general_networks as generate

            f = open("optimization_results/general_optimization/two_two_junction.json")
            results = json.load(f)
            f.close()

            network_file = results['network_file']
            f = open(network_file)
            network_config = json.load(f)
            f.close()

            T = network_config['T']
            print(T)
            N = network_config['N']
            controls = network_config['control_points']

            # Collecting the start and final parameters
            start = results['parameters'][0]
            opt = results['parameters'][-1]

            # Create the networks
            start_speeds = [[torch.tensor(start[i])] for i in range(4)]
            opt_speeds = [[torch.tensor(opt[i])] for i in range(4)]

            start_cycle = [torch.tensor(start[4]), torch.tensor(start[5])]
            opt_cycle = [torch.tensor(opt[4]), torch.tensor(opt[5])]


            start_network = generate.two_two_junction(T, N, start_speeds, controls, start_cycle, track_grad=False)
            opt_network = generate.two_two_junction(T, N, opt_speeds, controls, opt_cycle, track_grad=False)

            # Update positions of roads:
            start_network.roads[0].left_pos = (0, 4)
            start_network.roads[0].right_pos = (2.9, 4)
            start_network.roads[1].left_pos = (3.1, 4)
            start_network.roads[1].right_pos = (6, 4)
            start_network.roads[2].left_pos = (3, 0)
            start_network.roads[2].right_pos = (3, 3.9)
            start_network.roads[3].left_pos = (3, 4.1)
            start_network.roads[3].right_pos = (3, 8)

            opt_network.roads[0].left_pos = (0, 4)
            opt_network.roads[0].right_pos = (2.9, 4)
            opt_network.roads[1].left_pos = (3.1, 4)
            opt_network.roads[1].right_pos = (6, 4)
            opt_network.roads[2].left_pos = (3, 0)
            opt_network.roads[2].right_pos = (3, 3.9)
            opt_network.roads[3].left_pos = (3, 4.1)
            opt_network.roads[3].right_pos = (3, 8)

            # Load densities
            f = open("general_densities/two_two_start_opt_times.json")
            data = json.load(f)
            f.close()
            orig_densities = data[0]
            orig_lengths = data[1]

            f = open("general_densities/two_two_optimal.json")
            data_opt = json.load(f)
            f.close()
            opt_densities = data_opt[0]
            opt_lengths = data_opt[1]

            # Create gif:
            draw_busses_compare_w_opt(opt_network, opt_network.busses, opt_lengths,
                                    opt_densities, start_network.busses, orig_lengths, output_name="general_densities/videos/two_two.gif",
                                    background_img="background_imgs/white_background.png")
        
        case 19:
            # Compare optimal and non-optimal solutions on the medium complex network
            # Compare optimal and non-optimal solutions on a single junction
            import json
            import torch
            import generate_general_networks as generate

            f = open("optimization_results/general_optimization/medium_complex_new.json")
            results = json.load(f)
            f.close()

            network_file = results['network_file']
            f = open(network_file)
            network_config = json.load(f)
            f.close()

            T = network_config['T']
            N = network_config['N']
            controls = network_config['control_points']

            # Collecting the start and final parameters
            start = results['parameters'][0]
            opt = results['parameters'][-1]

            # Create the networks
            start_speeds = [[torch.tensor(start[i])] for i in range(8)]
            opt_speeds = [[torch.tensor(50.0)] for i in range(8)]

            start_cycle = [torch.tensor(start[8]), torch.tensor(start[9])]
            opt_cycle = [torch.tensor(opt[8]), torch.tensor(opt[9])]

            start_network = generate.medium_complex_network(T, N, start_speeds, controls, [start_cycle], track_grad=False)
            opt_network = generate.medium_complex_network(T, N, opt_speeds, controls, [opt_cycle], track_grad=False)

            # Update positions of roads:
            # left_positions = [(0.5, 4.69), (2, 8.18), (2, 4.81), (4, 4.91),
            #                   (2, 4.81), (-0.5, 9), (4, 4.91), (7, 4.91),
            #                   (0.5, 1.856), (4, 0), (4, 0), (7, 0),
            #                   (4, 4.91), (-1, 3.27), (0.5, 4.69), (0.5, 1.856)]
            # right_positions = [(2, 4.81), (-0.5, 9), (4, 4.91), (7, 4.91),
            #                    (0.5, 4.69), (2, 9), (2, 9), (4, 4.91),
            #                    (4, 0), (7, 0), (0.5, 1.856), (4, 0),
            #                    (4, 0), (0.5, 4.69), (0.5, 1.856), (-1, 3.27)]
            left_positions = [(1.5 - 0.1, -0.866 - 0.05 + 0.01732), (2.9, -3.1), (3.25, -3.1), (5.1, -1.1),
                              (3, -3), (0.4, -3.7), (4.85, -1.1), (8, -1),
                              (1.5, 0.866), (5.1, 2), (4.9, 2.1), (8, 2.1),
                              (5, -0.9), (0, 0), (1.5, -0.866), (1.5, 0.866)]
            right_positions = [(3 -0.16, -3 + 0.05+0.01732), (0.5, -3.5), (4.9, -1.2), (8, -1.1),
                               (1.5, -0.866), (2.8, -3.3), (3.15, -3.1), (5.1, -1),
                               (4.9, 2), (8, 2), (1.5, 0.966), (5.1, 2.1),
                               (5, 1.9), (1.5, -0.866), (1.5, 0.866), (0, 0)]
            
            for i in range(16):
                y_min = 0.1
                y_max = 8.9
                a_y = (y_min - y_max) / 5.5
                b_y = y_min + 2 * (y_max - y_min) / 5.5

                x_min = 0.1
                x_max = 6.9
                a_x = (x_max - x_min) / 8
                b_x = x_min

                start_network.roads[i].left_pos = (left_positions[i][0] * a_x + b_x,
                                                   left_positions[i][1] * a_y + b_y)
                opt_network.roads[i].left_pos = (left_positions[i][0] * a_x + b_x,
                                                   left_positions[i][1] * a_y + b_y)
                start_network.roads[i].right_pos = (right_positions[i][0] * a_x + b_x,
                                                   right_positions[i][1] * a_y + b_y)
                opt_network.roads[i].right_pos = (right_positions[i][0] * a_x + b_x,
                                                   right_positions[i][1] * a_y + b_y)


            # Load densities
            f = open("general_densities/medium_complex_start_opt_times.json")
            data = json.load(f)
            f.close()
            orig_densities = data[0]
            orig_lengths = data[1]

            f = open("general_densities/medium_complex_optimal.json")
            data_opt = json.load(f)
            f.close()
            opt_densities = data_opt[0]
            opt_lengths = data_opt[1]

            # Create gif:
            draw_busses_compare_w_opt(opt_network, opt_network.busses, opt_lengths,
                                    opt_densities, start_network.busses, orig_lengths, output_name="general_densities/videos/medium_complex.gif",
                                    background_img="background_imgs/white_background.png", interval_seconds=0.1)

        case 20:
            # Compare optimal and non-optimal solutions of kvadraturen
            import json
            import torch
            import generate_kvadraturen as gk

            # Load results from optimization
            f = open("optimization_results/kvadraturen_optimization/network22_config22_fwd.json")
            results = json.load(f)
            f.close()

            # Collecting network configuration
            network_file = results['network_file']
            config_file = results['config_file']
            f = open(network_file)
            network_config = json.load(f)
            f.close()

            T = network_config['T']
            N = network_config['N']
            controls = network_config['control_points']

            # Collecting the start and final parameters
            start = results['parameters'][0]
            opt = results['parameters'][-1]

            T, N, speed_limits, cycle_times = load_bus_network(network_file, config_file)




            start_network = create_network_from_params(T, N, start, track_grad=False)
            opt_network = create_network_from_params(T, N, opt, track_grad=False)




            # Load densities
            f = open("general_densities/kvadraturen_start_opt_times.json")
            data = json.load(f)
            f.close()
            orig_densities = data[0]
            orig_lengths = data[1]

            f = open("general_densities/kvadraturen_optimal.json")
            data_opt = json.load(f)
            f.close()
            opt_densities = data_opt[0]
            opt_lengths = data_opt[1]

            update_e18_bool()
            # Create gif:
            draw_busses_compare_w_opt(opt_network, opt_network.busses, opt_lengths,
                                    opt_densities, start_network.busses, orig_lengths, output_name="general_densities/videos/kvadraturen.gif",
                                    background_img="background_imgs/background_e18_cropped.png")