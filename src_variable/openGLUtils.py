from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
import time as time
import loading_json as load


color_gradient = 2

def draw_line_with_colors(colors, points, line_width):
    glLineWidth(line_width)
    glBegin(GL_LINE_STRIP)
    
    for color, point in zip(colors, points):
        glColor3f(*color)
        glVertex2f(*point)
    glEnd()

def map_value_to_color(value):
    # Ensure the input value is within the valid range [0, 1]'
    # Should not be necessary

    # Colors: 
    # (1.0, 0.0, 0.0) : Red
    # (1.0, 1.0, 0.0) : Yellow
    # (0.0, 1.0, 0.0) : Green
    # (0.0, 1.0, 1.0) : Cyan
    # (0.0, 0.0, 1.0) : Red

    value = max(0.0, min(1.0, value))
    match color_gradient:
        case 0:
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
        case 1:
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

        case 2:
            # Interpolate between blue and red
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

    return (float(r), float(g), float(b))

def convert_to_points(road_positions, n):
    left_point = None
    left_found = False
    right_point = None
    right_found = False

    while not left_found or not right_found:
        # At most go through twice
        for key in road_positions.keys():
            if key[:4] == 'Left':
                left_point = road_positions[key]
                left_found = True
            elif key[:5] == 'Right':
                right_point = road_positions[key]
                right_found = True
            if not left_found and right_found and key[:1] == 'J':
                # Right edge already found - left edge at junction
                left_point = road_positions[key]
                left_found = True
            elif left_found and not right_found and key[:1] == 'J':
                right_point = road_positions[key]
                right_found = True
                
    x = np.linspace(left_point[0], right_point[0], n)
    y = np.linspace(left_point[1], right_point[1], n)

    points = [(x[i], y[i]) for i in range(len(x))]
    return points

def draw_colored_line(colors, points):
    glClear(GL_COLOR_BUFFER_BIT)

    # Set up the view and projection matrices
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-2, 2, -2, 2, -2, 2)  # Set the coordinate system to be [-2, 2]x[-2, 2]

    glMatrixMode(GL_MODELVIEW)
    
    glLoadIdentity()

    line_width = 10.0  # Adjust the line width as needed
    for i in range(len(colors)):
        draw_line_with_colors(colors[i], points[i], line_width)

    # Add colormap
    glBegin(GL_LINE_STRIP)
    match color_gradient:
        case 0:
            color_map_colors = [(1,0,0), (1,1,0), (0,1,0)]
            color_map_points = [(1.8, 0.5), (1.8, 0), (1.8, -0.5)]

            for c, p in zip(color_map_colors, color_map_points):
                glColor3f(*c)
                glVertex2f(*p)
        case 1:
            color_map_colors = [(0,0,1), (0,1,1), (0,1,0),
                                (1,1,0), (1,0,0)]
            color_map_points = [(1.8, -0.5), (1.8, -0.25), (1.8, 0.0),
                                (1.8, 0.25), (1.8, 0.5)]

            for c, p in zip(color_map_colors, color_map_points):
                glColor3f(*c)
                glVertex2f(*p)
        case 2:
            color_map_colors = [(0,0,1), (0,0.5,0.5), (0,1,0),
                                (0.5,0.5,0), (1,0,0)]
            color_map_points = [(1.8, -0.5), (1.8, -0.25), (1.8, 0.0),
                                (1.8, 0.25), (1.8, 0.5)]
            for c, p in zip(color_map_colors, color_map_points):
                glColor3f(*c)
                glVertex2f(*p)
    glEnd()
    
    glutSwapBuffers()

def draw_network(network, densities):
    
    # Instead of using networkx to get road positions - assume they are already specified
    # road_positions = network.get_node_pos()

    # densities are evaluated at a specific point in time
    colors = [None] * len(densities)
    points = [None] * len(densities)
    for i, d in enumerate(densities):
        # d is an array of densities -> convert to colors
        colors[i] = [map_value_to_color(rho) for rho in d]
        road = network.roads[i]
        left = road.left_pos
        right = road.right_pos
        n = len(colors[i])
        x = np.linspace(left[0], right[0], n)
        y = np.linspace(left[1], right[1], n)
        points[i] = [(x[i], y[i]) for i in range(len(x))]

    
    # Initialize display
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutCreateWindow(b"OpenGL Colored Line")

    # Set the window size
    glutInitWindowSize(800, 800)
    glutDisplayFunc(lambda: draw_colored_line(colors, points))
    glutIdleFunc(lambda: draw_colored_line(colors, points))

    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    glutMainLoop()

class DensityRenderer:
    def __init__(self, colors, points, interval_seconds):
        self.colors = colors
        self.points = points
        self.current_idx = 0
        self.interval_seconds = interval_seconds
        self.last_update_time = time.time()
        self.is_rendering = True

    def display(self):
        draw_colored_line(self.colors[self.current_idx], self.points[self.current_idx])

    def timer(self, value):

        if self.current_idx >= len(self.colors)-1:
            self.is_rendering = False
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
        
def create_density_container(network, densities):
    # Create a container for the densities
    # Every element should be the 
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
            colors[-1][i] = [map_value_to_color(rho) for rho in d]
            road = network.roads[i]
            left = road.left_pos
            right = road.right_pos
            n = len(colors[-1][i])
            x = np.linspace(left[0], right[0], n)
            y = np.linspace(left[1], right[1], n)
            points[-1][i] = [(x[i], y[i]) for i in range(len(x))]
    return colors, points

def draw_timed(network, densities, interval_seconds=0.05):
    colors, points = create_density_container(network, densities)
    renderer = DensityRenderer(colors, points, interval_seconds=interval_seconds)

    # Initialize display
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutCreateWindow(b"OpenGL Colored Line")

    #time.sleep(0.5)

    # Set up callback functions
    glutDisplayFunc(renderer.display)
    #glutTimerFunc(int(interval_seconds * 1000), renderer.timer, 0)
    glutTimerFunc(1000, renderer.timer, 0)

    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    glutMainLoop()

def draw_black_lines(points):
    glClear(GL_COLOR_BUFFER_BIT)

    # Set up the view and projection matrices
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-2, 2, -2, 2, -2, 2)  # Set the coordinate system to be [-2, 2]x[-2, 2]

    glMatrixMode(GL_MODELVIEW)
    
    glLoadIdentity()
    line_width = 5.0  # Adjust the line width as needed

    for point in points:
        # For now only black line
        colors_ = [(0,0,0), (0,0,0)]

        # Id is a string
        # point on the form [(x1, y1), (x2, y2)]

        # If forward/backward, shift line up/down or left/right
        # Add logic for shifting here...
        # This shifting actually only needs to be done once at the beginnning


        # Positions should now be correct
        # Draw a line from start to end

        glLineWidth(line_width)
        glBegin(GL_LINE_STRIP)
        for c, p in zip(colors_, point):
            glColor3f(*c)
            glVertex2f(*p)
        glEnd()
    
    glutSwapBuffers()

def get_road_positions(network):
    # Go through the network and find the smalles and largest x and y values
    min_x = 1e10
    max_x = -1e10
    min_y = -1e10
    max_y = 1e10

    for road in network.roads:
        left = road.left_pos
        right = road.right_pos
        min_x = min(min_x, left[0], right[0])
        max_x = max(max_x, left[0], right[0])
        max_y = min(max_y, left[1], right[1])
        min_y = max(min_y, left[1], right[1])

    a_x = 4 / (max_x - min_x)
    b_x = 2 -  4 * max_x / (max_x - min_x)

    a_y = 4 / (max_y - min_y)
    b_y = 2 -  4 * max_y / (max_y - min_y)

    points = [[None, None] for i in range(len(network.roads))]
    for i, road in enumerate(network.roads):
        left = road.left_pos
        right = road.right_pos
        points[i][0] = (a_x * left[0] + b_x, a_y * left[1] + b_y)
        points[i][1] = (a_x * right[0] + b_x, a_y * right[1] + b_y)
    
    return points

def get_positions_bus(network, bus_positions, x_shift, y_shift):
    # Go through the network and find the smalles and largest x and y values
    min_x = 1e10
    max_x = -1e10
    min_y = -1e10
    max_y = 1e10

    for road in network.roads:
        left = road.left_pos
        right = road.right_pos
        min_x = min(min_x, left[0], right[0])
        max_x = max(max_x, left[0], right[0])
        max_y = min(max_y, left[1], right[1])
        min_y = max(min_y, left[1], right[1])
    
    a_x = 4 / (max_x - min_x)
    b_x = 2 -  4 * max_x / (max_x - min_x)

    a_y = 4 / (max_y - min_y)
    b_y = 2 -  4 * max_y / (max_y - min_y)

    points = [[None, None] for i in range(len(network.roads))]
    for i, road in enumerate(network.roads):
        left = road.left_pos
        right = road.right_pos
        points[i][0] = (a_x * left[0] + b_x, a_y * left[1] + b_y)
        points[i][1] = (a_x * right[0] + b_x, a_y * right[1] + b_y)

    bus_points = [[None, None] for i in range(len(bus_positions))]
    for i, pos in enumerate(bus_positions):
        # pos is a tuple (x, y)
        # Do shifting here
        try:
            bus_points[i] = (a_x * pos[0] + b_x + x_shift[i], a_y * pos[1] + b_y + y_shift[i])
        except:
            bus_points[i] = (None, None)
        
    
    return points, bus_points

def get_positions_of_busses(network, bus_positions, x_shift, y_shift):
    # Go through the network and find the smalles and largest x and y values
    min_x = 1e10
    max_x = -1e10
    min_y = -1e10
    max_y = 1e10

    for road in network.roads:
        left = road.left_pos
        right = road.right_pos
        min_x = min(min_x, left[0], right[0])
        max_x = max(max_x, left[0], right[0])
        max_y = min(max_y, left[1], right[1])
        min_y = max(min_y, left[1], right[1])
    
    a_x = 4 / (max_x - min_x)
    b_x = 2 -  4 * max_x / (max_x - min_x)

    a_y = 4 / (max_y - min_y)
    b_y = 2 -  4 * max_y / (max_y - min_y)

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

def shift_points(ids, points):
    for id, point in zip(ids, points):
        if id[-2:] == 'fw':
            # Shift road left or down depening on the direction
            # Find direction from x-values
            if point[1][0] > point[0][0]:
                # Road goes from left to the right
                # Shift road down
                point[0] = (point[0][0], point[0][1] - 0.1)
                point[1] = (point[1][0], point[1][1] - 0.1)
            else:
                # Road goes from top to bottom
                # Shift road to the left
                point[0] = (point[0][0] - 0.05, point[0][1])
                point[1] = (point[1][0] - 0.05, point[1][1])


        elif id[-2:] == 'bw':
            # Shift road right or up depending on the direction
            # Find direction from x-values
            if point[1][0] > point[0][0]:
                # Road goes from left to the right
                # Shift road up
                point[0] = (point[0][0], point[0][1] + 0.1)
                point[1] = (point[1][0], point[1][1] + 0.1)
            else:
                # Road goes from top to bottom
                # Shift road to the right
                point[0] = (point[0][0] + 0.05, point[0][1])
                point[1] = (point[1][0] + 0.05, point[1][1])
    return points

# def shift_bus_points(ids, points):
#     for id in 

def draw_black_lines_container(ids, points):
    glClear(GL_COLOR_BUFFER_BIT)

    # Set up the view and projection matrices
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-2, 2, -2, 2, -2, 2)  # Set the coordinate system to be [-2, 2]x[-2, 2]

    glMatrixMode(GL_MODELVIEW)

     # Add colormap
    glLineWidth(5.0)
    glBegin(GL_LINE_STRIP)
    color_map_colors = [(0,0,0), (0,0,0)]
    color_map_points = [(1.8, 0.5),(1.8, -0.5)]

    for c, p in zip(color_map_colors, color_map_points):
        glColor3f(*c)
        glVertex2f(*p)
    glEnd()
    
    glutSwapBuffers()



def visualize_network(network):
    # For now, don't care about the densities
    
    # Get the road positions
    points = get_road_positions(network)
    # Get the road ids
    ids = [road.id for road in network.roads]
    
    # Shift the points
    points = shift_points(ids, points)
    # Initialize display
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutCreateWindow(b"OpenGL Colored Line")

    # Set the window size
    glutInitWindowSize(800, 800)
    glutDisplayFunc(lambda: draw_black_lines(points))
    glutIdleFunc(lambda: draw_black_lines(points))

    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    glutMainLoop()


def draw_bus_network(road_points, bus_position):
    glClear(GL_COLOR_BUFFER_BIT)

    # Set up the view and projection matrices
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-2, 2, -2, 2, -2, 2)  # Set the coordinate system to be [-2, 2]x[-2, 2]

    glMatrixMode(GL_MODELVIEW)
    
    glLoadIdentity()
    line_width = 5.0  # Adjust the line width as needed


    for point in road_points:
        # For now only black line
        colors_ = [(0,0,0), (0,0,0)]

        # Id is a string
        # point on the form [(x1, y1), (x2, y2)]

        # If forward/backward, shift line up/down or left/right
        # Add logic for shifting here...
        # This shifting actually only needs to be done once at the beginnning


        # Positions should now be correct
        # Draw a line from start to end

        glLineWidth(line_width)
        glBegin(GL_LINE_STRIP)
        for c, p in zip(colors_, point):
            glColor3f(*c)
            glVertex2f(*p)
        glEnd()

    # Draw the bus
    if bus_position != (None, None):
        glPointSize(7.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.0, 0.0)
        glVertex2f(bus_position[0], bus_position[1])
        glEnd()

    glutSwapBuffers()


def draw_busses_network(road_points, bus_positions):
    glClear(GL_COLOR_BUFFER_BIT)

    # Set up the view and projection matrices
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-2, 2, -2, 2, -2, 2)  # Set the coordinate system to be [-2, 2]x[-2, 2]

    glMatrixMode(GL_MODELVIEW)
    
    glLoadIdentity()
    line_width = 5.0  # Adjust the line width as needed


    for point in road_points:
        # For now only black line
        colors_ = [(0,0,0), (0,0,0)]

        # Id is a string
        # point on the form [(x1, y1), (x2, y2)]

        # If forward/backward, shift line up/down or left/right
        # Add logic for shifting here...
        # This shifting actually only needs to be done once at the beginnning


        # Positions should now be correct
        # Draw a line from start to end

        glLineWidth(line_width)
        glBegin(GL_LINE_STRIP)
        for c, p in zip(colors_, point):
            glColor3f(*c)
            glVertex2f(*p)
        glEnd()

    # Draw the busses
    for bus_position in bus_positions:
        if bus_position != (None, None):
            glPointSize(7.0)
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.0, 0.0)
            glVertex2f(bus_position[0], bus_position[1])
            glEnd()

    glutSwapBuffers()
    


class BusRenderer:
    def __init__(self, road_points, bus_points, interval_seconds):
        '''
        Road poitns is a list of points defining the network. This is fixed for all times
        Bus points is a list of points defining the position of the bus. This changes for each time step
        '''
        self.road_points = road_points
        self.bus_points = bus_points
        self.current_idx = 0
        self.interval_seconds = interval_seconds
        self.last_update_time = time.time()
        self.is_rendering = True
    def display(self):
        draw_bus_network(self.road_points, self.bus_points[self.current_idx])
    
    def timer(self, value):
        # Value could be used to set the new interval?
        # Stop updating index when end is reached
        if self.current_idx >= len(self.bus_points)-1:
            self.is_rendering = False
            return

        # Update the current element index
        self.current_idx += 1

        # Redraw the scene
        glutPostRedisplay()

        # Set the timer for the next update - change this to take into account
        # that time intervals of the simulation might change, i.e. by setting interval_seconds as a list
        # and using the current index iin the code below
        glutTimerFunc(int(self.interval_seconds * 1000), self.timer, 0)

        # Update the last update time
        self.last_update_time = time.time()


class MultipleBusRenderer:
    def __init__(self, road_points, bus_points, interval_seconds):
        '''
        Road poitns is a list of points defining the network. This is fixed for all times
        Bus points is a list of points defining the position of the bus. This changes for each time step
        '''
        self.road_points = road_points
        self.bus_points = bus_points
        self.current_idx = 0
        self.interval_seconds = interval_seconds
        self.last_update_time = time.time()
        self.is_rendering = True
    def display(self):
        draw_busses_network(self.road_points, [points[self.current_idx] for points in self.bus_points])
    
    def timer(self, value):
        # Value could be used to set the new interval?
        # Stop updating index when end is reached
        if self.current_idx >= len(self.bus_points[0])-1:
            self.is_rendering = False
            return

        # Update the current element index
        self.current_idx += 1

        # Redraw the scene
        glutPostRedisplay()

        # Set the timer for the next update - change this to take into account
        # that time intervals of the simulation might change, i.e. by setting interval_seconds as a list
        # and using the current index iin the code below
        glutTimerFunc(int(self.interval_seconds * 1000), self.timer, 0)

        # Update the last update time
        self.last_update_time = time.time()

def find_points(network, bus, bus_lengths):
    # For each timestep, find the coordinates of the bus, using the positions of the roads
    x_shift = [0 for _ in range(len(bus_lengths))]
    y_shift = [0 for _ in range(len(bus_lengths))]
    positions = [(None, None) for _ in range(len(bus_lengths))]
    for i, length in enumerate(bus_lengths):
        # Find the road the bus is on
        road_id, length_travelled = bus.get_road_id_at_length(length)
        if road_id == "":
            # Bus has reached the end of the route -> don't draw the bus
            # The bus will not reenter the route, so the for loop can be broken
            break
        
        # Find the road the bus is on
        road = network.get_road(road_id)
        left = road.left_pos
        right = road.right_pos

        if road_id[-2:] == 'fw':
            if left[0] < right[0]:
                # Road goes from left to right
                # Shift road down
                y_shift[i] = -0.1
            else:
                # Road goes from top to bottom
                # Shift road to the left
                x_shift[i] = -0.05
        elif road_id[-2:] == 'bw':
            if left[0] < right[0]:
                # Road goes from right to the left
                # Shift road up
                y_shift[i] = 0.1
            else:
                # Road goes from bottom to left
                # Shift road to the right
                x_shift[i] = 0.05
        

        # The bus has travelled length/road.L of the road
        relative_length = length_travelled / road.L # Going from 0 to b
        # 0 -> left, b -> right
        x = 1/road.b * ((road.b - relative_length) * left[0] +  relative_length * right[0])
        y = 1/road.b * ((road.b - relative_length) * left[1] +  relative_length * right[1])
        positions[i] = (x, y)

    return positions, x_shift, y_shift

def find_points_of_busses(network, busses, bus_lengths):
    # For each timestep, find the coordinates of the bus, using the positions of the roads
    x_shift = [[0 for _ in range(len(bus_lengths[i]))] for i in range(len(busses))]
    y_shift = [[0 for _ in range(len(bus_lengths[i]))] for i in range(len(busses))]
    positions = [[(None, None) for _ in range(len(bus_lengths[i]))] for i in range(len(busses))]
    for i, lengths in enumerate(bus_lengths):
        for j, length in enumerate(lengths):
            # Find the road the bus is on
            road_id, length_travelled = busses[i].get_road_id_at_length(length)
            if road_id == "":
                # Bus has reached the end of the route -> don't draw the bus
                # The bus will not reenter the route, so the for loop can be broken
                break
            
            # Find the road the bus is on
            road = network.get_road(road_id)
            left = road.left_pos
            right = road.right_pos

            if road_id[-2:] == 'fw':
                if left[0] < right[0]:
                    # Road goes from left to right
                    # Shift road down
                    y_shift[i][j] = -0.1
                else:
                    # Road goes from top to bottom
                    # Shift road to the left
                    x_shift[i][j] = -0.05
            elif road_id[-2:] == 'bw':
                if left[0] < right[0]:
                    # Road goes from right to the left
                    # Shift road up
                    y_shift[i][j] = 0.1
                else:
                    # Road goes from bottom to left
                    # Shift road to the right
                    x_shift[i][j] = 0.05
        

            # The bus has travelled length/road.L of the road
            relative_length = length_travelled / road.L # Going from 0 to b
            # 0 -> left, b -> right
            x = 1/road.b * ((road.b - relative_length) * left[0] +  relative_length * right[0])
            y = 1/road.b * ((road.b - relative_length) * left[1] +  relative_length * right[1])
            positions[i][j] = (x, y)

    return positions, x_shift, y_shift
        

def draw_single_bus_timed(network, bus, bus_lengths, interval_seconds = 0.05):
    times = list(bus_lengths.keys())
    lengths = [float(bus_lengths[t]) for t in times]
    positions, x_shift, y_shift = find_points(network, bus, lengths)

    road_points, bus_points = get_positions_bus(network, positions, x_shift, y_shift)
    # Shift the road points
    # Get the road ids
  
    ids = [road.id for road in network.roads]
    
    # Shift the points
    road_points = shift_points(ids, road_points)
    # Shift the bus points
    # bus_points = shift_bus_points(ids, bus_points)
    renderer = BusRenderer(road_points, bus_points, interval_seconds=interval_seconds)

    # Initialize display
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutCreateWindow(b"OpenGL Bus Line")

    print("Window created")
    time.sleep(2.0)

    # Set up callback functions
    glutDisplayFunc(renderer.display)
    glutTimerFunc(1000, renderer.timer, 0) # send in different value here?

    glClearColor(1.0, 1.0, 1.0, 1.0)

    glutMainLoop()


def draw_busses_timed(bus_network, busses, bus_lengths, interval_seconds = 0.05):
    try:
        times = list(bus_lengths[0].keys())
        lengths = [[float(bus_lengths[i][t]) for t in times] for i in range(len(busses))]

    except:
        times = list(bus_lengths['0'].keys())
        lengths = [[float(bus_lengths[str(i)][t]) for t in times] for i in range(len(busses))]
    positions, x_shift, y_shift = find_points_of_busses(bus_network, busses, lengths)

    road_points, bus_points = get_positions_of_busses(bus_network, positions, x_shift, y_shift)
    
    # Shift the road points
    # Get the road ids
    ids = [road.id for road in network.roads]
    
    # Shift the points
    road_points = shift_points(ids, road_points)

    # Shift the bus points
    # bus_points = shift_bus_points(ids, bus_points)
    renderer = MultipleBusRenderer(road_points, bus_points, interval_seconds=interval_seconds)

    # Initialize display
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutCreateWindow(b"OpenGL Bus Line")

    print("Window created")
    time.sleep(2.0)

    # Set up callback functions
    glutDisplayFunc(renderer.display)
    glutTimerFunc(1000, renderer.timer, 0) # send in different value here?

    glClearColor(1.0, 1.0, 1.0, 1.0)

    glutMainLoop()



if __name__ == "__main__":
    import generate_kvadraturen as gk
    
    option = 7
    match option:
        case 0:
            network = gk.generate_kvadraturen_small(10.0)
            visualize_network(network)
            # points = get_road_positions(network)
            # ids = [road.id for road in network.roads]

        case 1:
            import network as nw
            import bus
            import loading_json as load
            import json
            T = 300
            network = gk.generate_kvadraturen_small(T)
            # for road in network.roads:
            ids = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
                "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
                "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]
            
            stops = []
            times = []
            bus = bus.Bus(ids, stops, times, network)
            print("Bus created!")


            roads = network.roads
            junctions = network.junctions
            T = network.T

            bus_network = nw.RoadNetwork(roads, junctions, T, [bus])
            print("Network created!")

            densities, queues, bus_lengths = bus_network.solve_cons_law()

            print("Solved conservation law!")

            bus_lengths = load.convert_from_tensor(bus_lengths)
            print("Dumping to json")
            with open(f'results/bus_lengths_{T}.json', 'w') as f:
                json.dump(bus_lengths, f)


            # draw_single_bus_timed(bus_network, bus, bus_lengths[0])
        
        case 2:
            import bus
            import json
            import network as nw
            
            with open('results/bus_lengths_300.json', 'r') as f:
                bus_lengths = json.load(f)
            network = gk.generate_kvadraturen_small(300)
            ids = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
                "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
                "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]
            
            
            bus_fw = bus.Bus(ids, [], [], network)

            bus_network = nw.RoadNetwork(network.roads, network.junctions, network.T, [bus_fw])
            draw_single_bus_timed(bus_network, bus_fw, bus_lengths['0'], 0.1)

        case 3:
            import bus
            import network as nw
            import json
            T = 300
            network = gk.generate_kvadraturen_small(T)


            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_6bw", "tollbod_5bw", "tollbod_4bw", 
                   "tollbod_3bw", "tollbod_2bw", "tollbod_1bw", "v_strand_7bw", "v_strand_6bw",
                     "v_strand_5bw", "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"]            
            

            stops = []
            times = []
            bus_bw = bus.Bus(ids_bw, stops, times, network)

            ids_fw = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
                "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
                "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]
            
            bus_fw = bus.Bus(ids_fw, stops, times, network)


            roads = network.roads
            junctions = network.junctions
            T = network.T

            bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw])

            densities, queues, bus_lengths = bus_network.solve_cons_law()

            print("Solved conservation law!")


            bus_lengths = load.convert_from_tensor(bus_lengths)
            print("Dumping to json")
            with open(f'results/two_bus_lengths_{T}.json', 'w') as f:
                json.dump(bus_lengths, f)

        case 4:
            import bus
            import json
            import network as nw
            
            with open('results/two_bus_lengths_300.json', 'r') as f:
                bus_lengths = json.load(f)

            network = gk.generate_kvadraturen_small(300)
            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_6bw", "tollbod_5bw", "tollbod_4bw", 
                   "tollbod_3bw", "tollbod_2bw", "tollbod_1bw", "v_strand_7bw", "v_strand_6bw",
                     "v_strand_5bw", "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"]            
            

            stops = []
            times = []
            bus_bw = bus.Bus(ids_bw, stops, times, network)

            ids_fw = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
                "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
                "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]
            
            bus_fw = bus.Bus(ids_fw, stops, times, network)

            bus_network = nw.RoadNetwork(network.roads, network.junctions, network.T, [bus_fw, bus_bw])
            draw_busses_timed(bus_network, [bus_fw, bus_bw], bus_lengths, 0.1)

        case 5:
            import bus
            import network as nw

            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_6bw", "tollbod_5bw", "tollbod_4bw", 
                   "tollbod_3bw", "tollbod_2bw", "tollbod_1bw", "v_strand_7bw", "v_strand_6bw",
                     "v_strand_5bw", "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"] 
            stops = [("tollbod_6bw", 50), ("tollbod_3bw", 90), ("tollbod_1bw", 30), ("v_strand_3bw", 25)]
            times = []
            network = gk.generate_kvadraturen_small(10)
            bus_bw = bus.Bus(ids_bw, stops, times, network)

            print(bus_bw.stop_lengths)

        case 6:
            import bus
            import network as nw
            import json
            T = 400
            network = gk.generate_kvadraturen_small(T)


            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_6bw", "tollbod_5bw", "tollbod_4bw", 
                   "tollbod_3bw", "tollbod_2bw", "tollbod_1bw", "v_strand_7bw", "v_strand_6bw",
                     "v_strand_5bw", "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"]            
            

            stops_bw = [("tollbod_6bw", 50), ("tollbod_3bw", 90), ("tollbod_1bw", 30), ("v_strand_3bw", 25)]
            times = []
            bus_bw = bus.Bus(ids_bw, stops_bw, times, network)

            ids_fw = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
                "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
                "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]
            
            stops_fw = [("h_w_3", 30), ("festning_5fw", 40), ("tollbod_4fw", 25), 
                        ("tollbod_6fw", 60)]
            bus_fw = bus.Bus(ids_fw, stops_fw, times, network)


            roads = network.roads
            junctions = network.junctions
            T = network.T

            bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw])

            densities, queues, bus_lengths = bus_network.solve_cons_law()

            print("Solved conservation law!")


            bus_lengths = load.convert_from_tensor(bus_lengths)

            with open(f'results/two_bus_lengths_stops_{T}.json', 'w') as f:
                json.dump(bus_lengths, f)
            # draw_busses_timed(bus_network, [bus_fw, bus_bw], bus_lengths, 0.2)

        case 7:
            import bus
            import network as nw
            import json
            T = 300
            network = gk.generate_kvadraturen_small(T)


            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_6bw", "tollbod_5bw", "tollbod_4bw", 
                   "tollbod_3bw", "tollbod_2bw", "tollbod_1bw", "v_strand_7bw", "v_strand_6bw",
                     "v_strand_5bw", "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"]            
            

            stops_bw = [("tollbod_6bw", 50), ("tollbod_3bw", 90), ("tollbod_1bw", 30), ("v_strand_3bw", 25)]
            times = []
            bus_bw = bus.Bus(ids_bw, stops_bw, times, network)

            ids_fw = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
                "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
                "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]
            
            stops_fw = [("h_w_3", 30), ("festning_5fw", 40), ("tollbod_4fw", 25), 
                        ("tollbod_6fw", 60)]
            bus_fw = bus.Bus(ids_fw, stops_fw, times, network)


            roads = network.roads
            junctions = network.junctions
            T = network.T

            bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw])
            with open('results/two_bus_lengths_stops_400.json', 'r') as f:
                bus_lengths = json.load(f)

            draw_busses_timed(bus_network, [bus_fw, bus_bw], bus_lengths, 0.1)