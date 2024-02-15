from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np

color_gradient = 2

def draw_line_with_colors(colors, points, line_width):
    glLineWidth(line_width)
    glBegin(GL_LINE_STRIP)
    
    for color, point in zip(colors, points):
        glColor3f(*color)
        glVertex2f(*point)

    glEnd()


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
            color_map_points = [(1.7, 1.8), (1.7, 0), (1.7, -1.8)]

            for c, p in zip(color_map_colors, color_map_points):
                glColor3f(*c)
                glVertex2f(*p)
        case 1:
            color_map_colors = [(0,0,1), (0,1,1), (0,1,0),
                                (1,1,0), (1,0,0)]
            color_map_points = [(1.7, -1.8), (1.7, -0.9), (1.7, 0.0),
                                (1.7, 0.9), (1.7, 1.8)]

            for c, p in zip(color_map_colors, color_map_points):
                glColor3f(*c)
                glVertex2f(*p)
        case 2:
            color_map_colors = [(0,0,1), (0,0.5,0.5), (0,1,0),
                                (0.5,0.5,0), (1,0,0)]
            color_map_points = [(1.7, -1.8), (1.7, -0.9), (1.7, 0.0),
                                (1.7, 0.9), (1.7, 1.8)]
            for c, p in zip(color_map_colors, color_map_points):
                glColor3f(*c)
                glVertex2f(*p)
    glEnd()

    # Add text to colormap
    glColor3f(0.0, 0.0, 0.0)
    glRasterPos2f(1.8, 1.8)
    text = "1.0"
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    glColor3f(0.0, 0.0, 0.0)
    glRasterPos2f(1.8, 0.0)
    text = "0.5"
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    glColor3f(0.0, 0.0, 0.0)
    glRasterPos2f(1.8, -1.8)
    text = "0.0"
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))
    
    glutSwapBuffers()

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
            # print(f"Key: {key}")
            # print(f"Left: {left_point}")
            # print(f"Right: {right_point}")

            if key[:4] == 'Left':
                # print("Left edge found")
                left_point = road_positions[key]
                # print(f"Left: {left_point}")
                left_found = True

            elif key[:5] == 'Right':
                # print("Right edge found")
                right_point = road_positions[key]
                # print(f"Right: {right_point}")

                right_found = True

            if not left_found and right_found and key[:1] == 'J':
                # print("Right found, left at junction")
                # Right edge already found - left edge at junction
                left_point = road_positions[key]
                # print(f"Left: {left_point}")

                left_found = True

            elif left_found and not right_found and key[:1] == 'J':
                # print("Left found, right at junction")
                right_point = road_positions[key]
                # print(f"Right: {right_point}")

                right_found = True

    # print(left_found, right_found)
    # print(left_point, right_point)
    # At this point left and right points should be found
    # Add as many intermediate points as necessary
    x = np.linspace(left_point[0], right_point[0], n)
    y = np.linspace(left_point[1], right_point[1], n)

    points = [(x[i], y[i]) for i in range(len(x))]
    # print(points)
    return points




def draw_network(network, densities):
    
    # Get positions of nodes of network - could also change this to make 
    # user specify coordinates of the road
    road_positions = network.get_node_pos()
    print(road_positions)

    # densities are evaluated at a specific point in time
    colors = [None] * len(densities)
    points = [None] * len(densities)
    for i, d in enumerate(densities):
        # d is an array of densities -> convert to colors
        colors[i] = [map_value_to_color(rho) for rho in d]
        points[i] = convert_to_points(road_positions[i], len(colors[i])) # SHould be same length as colors[i]
    
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

def draw_network2(network, densities):
    
    # Instead of using networkx to get road positions - assume they are already specified
    # road_positions = network.get_node_pos()
    # print(road_positions)

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


def draw_colored_line2(colors, points):
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

    # # Add text to colormap
    # glColor3f(0.0, 0.0, 0.0)
    # glRasterPos2f(1.6, 0.5)
    # text = "1.0"
    # for char in text:
    #     glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    # glColor3f(0.0, 0.0, 0.0)
    # glRasterPos2f(1.6, 0.0)
    # text = "0.5"
    # for char in text:
    #     glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    # glColor3f(0.0, 0.0, 0.0)
    # glRasterPos2f(1.6, -0.5)
    # text = "0.0"
    # for char in text:
    #     glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))
    
    glutSwapBuffers()

def draw_network3(network, densities):
    
    # Instead of using networkx to get road positions - assume they are already specified
    # road_positions = network.get_node_pos()
    # print(road_positions)

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
    glutDisplayFunc(lambda: draw_colored_line2(colors, points))
    glutIdleFunc(lambda: draw_colored_line2(colors, points))

    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    glutMainLoop()


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutCreateWindow(b"OpenGL Colored Line")

    # Set the window size
    glutInitWindowSize(800, 800)


    densities = [[0, 0.2, 0.4, 0.5, 0.8, 1],
                 [0.4, 0.2, 0.5, 0.4]]
    colors = [[map_value_to_color(rho) for rho in densities[i]] for i in range(len(densities))]
    print(colors)

    points = [[(-1.5, 0.2), 
              (-0.75, 0.2), 
              (-0.2, 0.2),
              (0.2, 0.2), 
              (0.9, 0.2),
              (1.5, 0.2)],
              [(0, 1.5),
               (0, 1.0),
               (0, 0),
               (0, -1)]]

    # Lambda function needed because the function should not contain any
    # arguments - define new function that does not contain arguments
    glutDisplayFunc(lambda: draw_colored_line2(colors, points))
    glutIdleFunc(lambda: draw_colored_line2(colors, points))

    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    glutMainLoop()
    
if __name__ == "__main__":
    option = 3
    match option:
        case 0:
            main()
        
        case 1:
            import loading_json as load
            import plotting as plot
            import torch


            loaded_roads, loaded_junctions, network = load.initialize_road_network("networks/2-2_coupledlights.json")
            # for road in network.roads:
            #     print(road.left_pos)
            #     print(road.right_pos)
            densities, queues = network.solve_cons_law()

            end_time = list(densities[0].keys())[-1]
            end_densities = [densities[i][end_time] for i in range(len(densities))]

            print(end_densities) # Could multiply by max density, but don't do that for now
            # end_densities = [
            #     torch.tensor([0.8000, 0.8000, 0.8000, 0.8000, 0.8001, 0.8001, 0.8001, 0.8002, 0.8003,
            #     0.8004, 0.8006, 0.8009, 0.8013, 0.8019, 0.8026, 0.8035, 0.8046, 0.8061,
            #     0.8078, 0.8099, 0.8122, 0.8149, 0.8178, 0.8209, 0.8242, 0.8275, 0.8308,
            #     0.8340, 0.8371, 0.8399, 0.8426, 0.8448, 0.8468, 0.8484, 0.8498, 0.8508,
            #     0.8517, 0.8522, 0.8527, 0.8530, 0.8532, 0.8533, 0.8534, 0.8535, 0.8535,
            #     0.8535, 0.8535, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536]),
            #     torch.tensor([0.4729, 0.4729, 0.4526, 0.4401, 0.4280, 0.4163, 0.4048, 0.3935, 0.3824,
            #     0.3715, 0.3607, 0.3500, 0.3395, 0.3291, 0.3188, 0.3086, 0.2986, 0.2888,
            #     0.2790, 0.2695, 0.2601, 0.2508, 0.2417, 0.2329, 0.2242, 0.2157, 0.2075,
            #     0.1995, 0.1917, 0.1843, 0.1771, 0.1702, 0.1635, 0.1573, 0.1513, 0.1458,
            #     0.1404, 0.1356, 0.1310, 0.1269, 0.1230, 0.1197, 0.1165, 0.1139, 0.1114,
            #     0.1094, 0.1075, 0.1061, 0.1047, 0.1038, 0.1029, 0.1023, 0.1023, 0.1023])]
            
            # # road_positions  = network.get_node_pos()
            # # print(road_positions)
            # # print(end_densities)
            draw_network2(network, end_densities)

            
            # road_positions = network.get_node_pos()

            # densities are evaluated at a specific point in time
            # colors = [None] * len(end_densities)
            # points = [None] * len(end_densities)
            # for i, d in enumerate(end_densities):
            #     # d is an array of densities -> convert to colors
            #     colors[i] = [map_value_to_color(rho) for rho in d]

            #     convert_to_points(road_positions[i], len(colors[i]))
            
            # print(colors)

        case 2:
            import loading_json as load
            import plotting as plot
            import torch
            import matplotlib.pyplot as plt

            loaded_roads, loaded_junctions, network = load.initialize_road_network("networks/2-2_coupledlights.json")
            # for road in network.roads:
            #     print(road.left_pos)
            #     print(road.right_pos)
            densities, queues = network.solve_cons_law()

            # Plot the two activation functions
            times = np.array(list(densities[0].keys()))

            for j in network.junctions:
                for light in j.coupled_trafficlights:
                    plt.plot(times, [light.a_activation(float(t)).detach() for t in times], 
                             label = "a activation")
                    plt.plot(times, [light.b_activation(float(t)).detach() for t in times],
                             label = "b activation")
            plt.legend()
            plt.show()

        case 3:
            import loading_json as load
            import plotting as plot
            import torch


            # loaded_roads, loaded_junctions, network = load.initialize_road_network("networks/complex_bad_case.json")
            loaded_roads, loaded_junctions, network = load.initialize_road_network("networks/complex_bad_case.json")
            densities, queues = network.solve_cons_law()

            end_time = list(densities[0].keys())[-1]
            end_densities = [densities[i][end_time] for i in range(len(densities))]

            print(end_densities) # Could multiply by max density, but don't do that for now
            # end_densities = [
            #     torch.tensor([0.8000, 0.8000, 0.8000, 0.8000, 0.8001, 0.8001, 0.8001, 0.8002, 0.8003,
            #     0.8004, 0.8006, 0.8009, 0.8013, 0.8019, 0.8026, 0.8035, 0.8046, 0.8061,
            #     0.8078, 0.8099, 0.8122, 0.8149, 0.8178, 0.8209, 0.8242, 0.8275, 0.8308,
            #     0.8340, 0.8371, 0.8399, 0.8426, 0.8448, 0.8468, 0.8484, 0.8498, 0.8508,
            #     0.8517, 0.8522, 0.8527, 0.8530, 0.8532, 0.8533, 0.8534, 0.8535, 0.8535,
            #     0.8535, 0.8535, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536]),
            #     torch.tensor([0.4729, 0.4729, 0.4526, 0.4401, 0.4280, 0.4163, 0.4048, 0.3935, 0.3824,
            #     0.3715, 0.3607, 0.3500, 0.3395, 0.3291, 0.3188, 0.3086, 0.2986, 0.2888,
            #     0.2790, 0.2695, 0.2601, 0.2508, 0.2417, 0.2329, 0.2242, 0.2157, 0.2075,
            #     0.1995, 0.1917, 0.1843, 0.1771, 0.1702, 0.1635, 0.1573, 0.1513, 0.1458,
            #     0.1404, 0.1356, 0.1310, 0.1269, 0.1230, 0.1197, 0.1165, 0.1139, 0.1114,
            #     0.1094, 0.1075, 0.1061, 0.1047, 0.1038, 0.1029, 0.1023, 0.1023, 0.1023])]
            
            # # road_positions  = network.get_node_pos()
            # # print(road_positions)
            # # print(end_densities)
            draw_network3(network, end_densities)