import torch
import road as rd
import junction as jn
import traffic_lights as tl
import network as nw
import FV_schemes as fv
import roundabout as rb
import initial_and_bc as ibc
import bus

def single_lane_network(T, N , speed_limit = [torch.tensor(40.0)], 
                        control_points = [], track_grad = True):
    
    # Creating the road:
    # Configuration of the single lane
    L = 50
    b = 4
    if torch.is_tensor(speed_limit[0]):
        speed = [v / 3.6 for v in speed_limit]
    else:
        speed = [torch.tensor(v / 3.6) for v in speed_limit]
        
    for v in speed:
        v.requires_grad = track_grad

    initial_fnc = lambda x : torch.ones_like(x) * 0.4
    boundary_fnc = ibc.boundary_conditions(1, max_dens = 1, densities = torch.tensor([0.4]),
                                           time_jumps = [], in_speed = torch.tensor(40.0),
                                           L = L)

    road = rd.Road(b, L, N, speed, control_points, max_dens=1, initial=initial_fnc,
                   boundary_fnc=boundary_fnc, id="road_1")
    
    # Creating temporary network:
    temp_network = nw.RoadNetwork([road], [], T, [], [])

    # Creating the bus:
    ids = ["road_1"]
    stops = [("road_1", 150)]
    times = [0]
    bus_1 = bus.Bus(ids, stops, times, temp_network, id="bus_1")

    # Creating the network:
    network = nw.RoadNetwork([road], [], T, [], [bus_1])

    return network

    
def single_junction_network(T, N, speed_limits=[[torch.tensor(50.0)], [torch.tensor(50.0)]], 
                            control_points = [[], []],
                            cycle_times = [torch.tensor(60.0), torch.tensor(60.0)], 
                            track_grad = True):
    
    # Creating the road:
    # Configuration of the single lane
    L = 50
    b = 2
    if torch.is_tensor(speed_limits[0][0]):
        speed_1 = [v / 3.6 for v in speed_limits[0]]
        speed_2 = [v / 3.6 for v in speed_limits[1]]

    else:
        speed_1 = [torch.tensor(v / 3.6) for v in speed_limits[0]]
        speed_2 = [torch.tensor(v / 3.6) for v in speed_limits[1]]

    # print(speed_1, speed_2)
    # print(control_points)

        
    for v_1, v_2 in zip(speed_1, speed_2):
        v_1.requires_grad = track_grad
        v_2.requires_grad = track_grad

    # print("--------")
    # print(speed_1)
    # print(speed_2)


    initial_fnc = lambda x : torch.ones_like(x) * 0.4
    boundary_fnc = ibc.boundary_conditions(1, max_dens = 1, densities = torch.tensor([0.4]),
                                           time_jumps = [], in_speed = torch.tensor(40.0/3.6),
                                           L = L)

    road_1 = rd.Road(b, L, N, speed_1, control_points[0], max_dens=1, initial=initial_fnc,
                   boundary_fnc=boundary_fnc, id="first_road")
    road_2 = rd.Road(b, L, N, speed_2, control_points[1], max_dens=1, initial=initial_fnc,
                   boundary_fnc=None, id="second_road")
    

    new_cycle = []
    for c in cycle_times:
        if torch.is_tensor(c):
            c.requires_grad = track_grad
            new_cycle.append(c)
        else:
            c = torch.tensor(c, requires_grad=track_grad)
            new_cycle.append(c)
    
    # Create junction with traffic light
    light = tl.TrafficLightContinous(True, [0], [1], new_cycle)
    junction = jn.Junction([road_1, road_2], [0], [1], [[1.0]], [light], [])


    
    # Creating temporary network:
    temp_network = nw.RoadNetwork([road_1, road_2], [junction], T, [], [])

    # Creating the bus:
    ids = ["first_road", "second_road"]
    stops = [("second_road", 50)]
    times = [0]
    bus_1 = bus.Bus(ids, stops, times, temp_network, id="bus_1",
                    start_time=00)

    # Creating the network:
    network = nw.RoadNetwork([road_1, road_2], [junction], T, [], [bus_1])

    return network

def two_two_junction(T, N, speed_limits=[[torch.tensor(50.0)], [torch.tensor(50.0)],
                                         [torch.tensor(50.0)], [torch.tensor(50.0)]], 
                    control_points = [[], [], [], []],
                    cycle_times = [torch.tensor(60.0), torch.tensor(60.0)], 
                    track_grad = True):
    
    # Creating the road:
    # Configuration of the single lane
    L = 50
    b = 2
    if torch.is_tensor(speed_limits[0][0]):
        speed_1 = [v / 3.6 for v in speed_limits[0]]
        speed_2 = [v / 3.6 for v in speed_limits[1]]
        speed_3 = [v / 3.6 for v in speed_limits[2]]
        speed_4 = [v / 3.6 for v in speed_limits[3]]

    else:
        speed_1 = [torch.tensor(v / 3.6) for v in speed_limits[0]]
        speed_2 = [torch.tensor(v / 3.6) for v in speed_limits[1]]
        speed_3 = [torch.tensor(v / 3.6) for v in speed_limits[2]]
        speed_4 = [torch.tensor(v / 3.6) for v in speed_limits[3]]

    # print(speed_1, speed_2)
    # print(control_points)

        
    for v_1, v_2, v_3, v_4 in zip(speed_1, speed_2, speed_3, speed_4):
        v_1.requires_grad = track_grad
        v_2.requires_grad = track_grad
        v_3.requires_grad = track_grad
        v_4.requires_grad = track_grad

    # print("--------")
    # print(speed_1)
    # print(speed_2)


    initial_fnc = lambda x : torch.ones_like(x) * 0.4
    boundary_fnc = ibc.boundary_conditions(1, max_dens = 1, densities = torch.tensor([0.4]),
                                           time_jumps = [], in_speed = torch.tensor(40.0),
                                           L = L)

    road_1 = rd.Road(b, L, N, speed_1, control_points[0], max_dens=1, initial=initial_fnc,
                   boundary_fnc=boundary_fnc, id="first_road")
    road_2 = rd.Road(b, L, N, speed_2, control_points[1], max_dens=1, initial=initial_fnc,
                   boundary_fnc=None, id="second_road")
    road_3 = rd.Road(b, L, N, speed_3, control_points[2], max_dens=1, initial=initial_fnc,
                   boundary_fnc=boundary_fnc, id="third")
    road_4 = rd.Road(b, L, N, speed_4, control_points[3], max_dens=1, initial=initial_fnc,
                   boundary_fnc=None, id="fourth")
    
    roads = [road_1, road_2, road_3, road_4]
    

    new_cycle = []
    for c in cycle_times:
        if torch.is_tensor(c):
            c.requires_grad = track_grad
            new_cycle.append(c)
        else:
            c = torch.tensor(c, requires_grad=track_grad)
            new_cycle.append(c)
    
    # Create junction with traffic light
    light = tl.CoupledTrafficLightContinuous(False, [0], [1], [2], [3], new_cycle)
    junction = jn.Junction(roads, [0,2], [1,3], [[1.0, 0.0],[0.0, 1.0]], [], [light])


    
    # Creating temporary network:
    temp_network = nw.RoadNetwork(roads, [junction], T, [], [])

    # Creating the bus:
    ids = ["first_road", "second_road"]
    stops = [("second_road", 50)]
    times = [0]
    bus_1 = bus.Bus(ids, stops, times, temp_network, id="bus_1")

    ids = ["third", "fourth"]
    stops = [("fourth", 50)]
    times = [0]
    bus_2 = bus.Bus(ids, stops, times, temp_network, id="bus_2")

    # Creating the network:
    network = nw.RoadNetwork(roads, [junction], T, [], [bus_1, bus_2])

    return network

def compare_grid_size_network(T, N, speed1 = [torch.tensor(50.0/3.6)], speed2 = [torch.tensor(50.0/3.6)],
                              speed3 = [torch.tensor(50.0/3.6)], cycle1 = [torch.tensor(60.0), torch.tensor(80.0)],
                              cycle2 = [torch.tensor(50.0), torch.tensor(50.0)],
                              cycle3 = [torch.tensor(70.0), torch.tensor(70.0)], 
                              densities = [0.3, 0.3, 0.3], track_grad = False,
                              scheme = 3):
    '''
    Create a closed smaller network for comparing the resolution of different grid sizes
    The network consists of a three armed network where each arm leads to a junction. In each junction there is a 
    traffic light
    Inner roundabout has roads of lengths 25 metres
    Outgoing roads leading out from roundabout has roads of length 50 metres
    Outer roads have lengths of 125 metres
    Theese lengths are not physically correct if all junctions have zero size and roads are straight, but with some curve 
    in the roads or some size of the junction it is okay
    '''

    # Creating the roads:
    L = 25
    outer_fw = [None for _ in range(3)]
    outer_bw = [None for _ in range(3)]
    inner_fw = [None for _ in range(3)]
    inner_bw = [None for _ in range(3)]
    mainlines = [None for _ in range(3)]

    init_fncs = [lambda x : torch.ones_like(x) * rho for rho in densities]
    outer_positions = [(0,0), (3, 7.8), (6,0)]
    inner_positions = [(2.3, 2), (3, 3.76), (3.7, 2)]

    # Creating the roads:
    for i in range(3):
        outer_fw[i] = rd.Road(5, L, N, speed1, [], initial=init_fncs[0], left_pos=outer_positions[i], 
                              right_pos=outer_positions[(i+1)%3], id="outer_"+str(i+1)+"fw", scheme=scheme)
        outer_bw[i] = rd.Road(5, L, N, speed1, [], initial=init_fncs[0], left_pos=outer_positions[(i+1)%3], 
                              right_pos=outer_positions[i], id="outer_"+str(i+1)+"bw", scheme=scheme)
        
        inner_fw[i] = rd.Road(2, L, N, speed2, [], initial=init_fncs[1], left_pos=outer_positions[i], 
                              right_pos=inner_positions[i], id="inner_"+str(i+1)+"fw", scheme=scheme)
        inner_bw[i] = rd.Road(2, L, N, speed2, [], initial=init_fncs[1], left_pos=inner_positions[i], 
                              right_pos=outer_positions[i], id="inner_"+str(i+1)+"bw", scheme=scheme)
        
        mainlines[i] = rd.Road(1, L, N, speed3, [], initial=init_fncs[2], left_pos=inner_positions[i],
                               right_pos=inner_positions[(i+1)%3], id="mainline_"+str(i+1), scheme=scheme)

    # Creating the traffic lights:
    traffic_lights = [None for _ in range(3)]
    cycles = [cycle1, cycle2, cycle3]
    for i in range(3):
        traffic_lights[i] = tl.CoupledTrafficLightContinuous(True, [0,3], [1,2,4], [5], [1,2],
                                                          cycle=cycles[i])

    # Creating the outer junctions:
    distribution = [[0, 0.6, 0.4],
                    [0.6, 0, 0.4],
                    [0.5, 0.5, 0]]
    priorities = [[0, 1, 2],
                  [1, 0, 1],
                  [2, 2, 0]]
    crossings = [[[], [], [(1,0)]],
                 [[], [], []],
                 [[], [], []]]
    outer_jncs = [None for _ in range(3)]
    for i in range(3):
        outer_jncs[i] = jn.Junction([outer_fw[i], outer_bw[i], outer_fw[(i+1)%3],
                                     outer_bw[(i+1)%3], inner_fw[i], inner_bw[i]],
                                     entering=[0,3,5], leaving=[1,2,4], distribution=distribution,
                                     trafficlights=[], coupled_trafficlights=[traffic_lights[i]],
                                     duty_to_gw=True, priorities=priorities, crossing_connections=crossings)

        # outer_jncs[i] = jn.Junction([outer_fw[i], outer_bw[i], outer_fw[(i+1)%3],
        #                              outer_bw[(i+1)%3], inner_fw[i], inner_bw[i]],
        #                              entering=[0,3,5], leaving=[1,2,4], distribution=distribution,
        #                              trafficlights=[], coupled_trafficlights=[],
        #                              duty_to_gw=True, priorities=priorities, crossing_connections=crossings)
        
    # Creating the roundabout junctions:
    roundabout_jncs = [None for _ in range(3)]

    for i in range(3):
        roundabout_jncs[i] = rb.RoundaboutJunction(mainlines[i], mainlines[(i+1)%3], alpha = 0.5,
                                                   second_in=inner_fw[i], second_out=inner_bw[i],
                                                   queue_junction=False)
    # Creating the roundabout:
    roundabout = rb.Roundabout(mainlines, inner_fw, inner_bw, roundabout_jncs)

    # Creating the network:
    roads = outer_fw + outer_bw + inner_fw + inner_bw + mainlines
    network = nw.RoadNetwork(roads, outer_jncs, T = T, roundabouts=[roundabout], busses=[])

    return network

def medium_complex_network(T, N, speed_limits, control_points, cycle_times, track_grad = False):
    '''
    '''
    L = 50

    if torch.is_tensor(speed_limits[0][0]):
        speed_1 = [v / 3.6 for v in speed_limits[0]]
        for v in speed_1:
            v.requires_grad = track_grad
        speed_2 = [v / 3.6 for v in speed_limits[1]]
        for v in speed_2:
            v.requires_grad = track_grad
        speed_3 = [v / 3.6 for v in speed_limits[2]]
        for v in speed_3:
            v.requires_grad = track_grad
        speed_4 = [v / 3.6 for v in speed_limits[3]]
        for v in speed_4:
            v.requires_grad = track_grad
        speed_5 = [v / 3.6 for v in speed_limits[4]]
        for v in speed_5:
            v.requires_grad = track_grad
        speed_6 = [v / 3.6 for v in speed_limits[5]]
        for v in speed_6:
            v.requires_grad = track_grad
        speed_7 = [v / 3.6 for v in speed_limits[6]]
        for v in speed_7:
            v.requires_grad = track_grad
        speed_8 = [v / 3.6 for v in speed_limits[7]]
        for v in speed_8:
            v.requires_grad = track_grad
    else:
        speed_1 = [torch.tensor(v / 3.6, requires_grad=track_grad) for v in speed_limits[0]]
        speed_2 = [torch.tensor(v / 3.6, requires_grad=track_grad) for v in speed_limits[1]]
        speed_3 = [torch.tensor(v / 3.6, requires_grad=track_grad) for v in speed_limits[2]]
        speed_4 = [torch.tensor(v / 3.6, requires_grad=track_grad) for v in speed_limits[3]]
        speed_5 = [torch.tensor(v / 3.6, requires_grad=track_grad) for v in speed_limits[4]]
        speed_6 = [torch.tensor(v / 3.6, requires_grad=track_grad) for v in speed_limits[5]]
        speed_7 = [torch.tensor(v / 3.6, requires_grad=track_grad) for v in speed_limits[6]]
        speed_8 = [torch.tensor(v / 3.6, requires_grad=track_grad) for v in speed_limits[7]]

    new_cycle = []
    for c in cycle_times[0]:
        if torch.is_tensor(c):
            c.requires_grad = track_grad
            new_cycle.append(c)
        else:
            c = torch.tensor(c, requires_grad=track_grad)
            new_cycle.append(c)

    initial_fnc = lambda x : torch.ones_like(x) * 0.4
    boundary_fnc = ibc.boundary_conditions(1, max_dens = 1, densities = torch.tensor([0.2]),
                                           time_jumps = [], in_speed = torch.tensor(40.0/3.6),
                                           L = L)

    # Creating the regular roads of the network
    south_fwd = [None for _ in range(4)]
    south_fwd[0] = rd.Road(1, L, N, speed_1, control_points[0], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="south_one_fw")
    south_fwd[1] = rd.Road(1, L, N, speed_2, control_points[1], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="south_two_fw")
    south_fwd[2] = rd.Road(1, L, N, speed_3, control_points[2], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="south_three_fw")
    south_fwd[3] = rd.Road(1, L, N, speed_4, control_points[3], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="south_four_fw")
    
    south_bwd = [None for _ in range(4)]
    south_bwd[0] = rd.Road(1, L, N, speed_1, control_points[0], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="south_one_bw")
    south_bwd[1] = rd.Road(1, L, N, speed_2, control_points[1], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=boundary_fnc, id="south_two_bw")
    south_bwd[2] = rd.Road(1, L, N, speed_3, control_points[2], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="south_three_bw")
    south_bwd[3] = rd.Road(1, L, N, speed_4, control_points[3], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=boundary_fnc, id="south_four_bw")

    north_fwd = [None for _ in range(2)]
    north_fwd[0] = rd.Road(1, L, N, speed_5, control_points[4], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="north_one_fw")
    north_fwd[1] = rd.Road(1, L, N, speed_6, control_points[5], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="north_two_fw")
    
    north_bwd = [None for _ in range(2)]
    north_bwd[0] = rd.Road(1, L, N, speed_5, control_points[4], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="north_one_bw")
    north_bwd[1] = rd.Road(1, L, N, speed_6, control_points[5], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=boundary_fnc, id="north_two_bw")

    middle_road = [None]
    middle_road[0] = rd.Road(1, L, N, speed_7, control_points[6], max_dens = 1, initial=initial_fnc,
                           boundary_fnc=None, id="middle")

    # Creating the junctions of the network
    junctions = [None for _ in range(3)]
    # First junction with coupled traffic light - probably should have duty to give way also...
    distribution_1 = [[0.0, 0.4, 0.6],
                      [0.5, 0.0, 0.5],
                      [0.6, 0.4, 0.0]]
    priorities_1 = [[0, 1, 1],
                    [1, 0, 2],
                    [2, 2, 0]]
    crossings = [[[], [], []],
                 [[], [], []],
                 [[], [], []]]
    # Alt start: True
    # Orig: False
    light = tl.CoupledTrafficLightContinuous(False, [0, 5], [1, 2, 4], [3], [1,2,4],
                                             cycle=new_cycle)
    # light = tl.TrafficLightContinous(True, [0], [2,4], cycle=new_cycle)
    # Check this junction!!!!
    junctions[0] = jn.Junction([south_fwd[0], south_bwd[0], south_fwd[1], south_bwd[1],
                                south_fwd[2], south_bwd[2]], [0,3,5], [1,2,4],
                                distribution_1, [], [light], True, priorities_1, crossings)

    distribution_2 = [[0.0, 0.8, 0.2],
                      [0.7, 0.0, 0.3]]
    priorities_2 = [[0, 1, 2],
                    [1, 0, 1]]
    crossings_2 = [[[], [], [(1,0)]],
                 [[], [], []]]
    junctions[1] = jn.Junction([south_fwd[2], south_bwd[2], south_fwd[3], south_bwd[3], middle_road[0]],
                               [0, 3], [1, 2, 4], distribution_2, [], [], True, priorities_2, crossings_2)
    
    distribution_3 = [[0.0, 1.0],
                      [1.0, 0.0],
                      [0.5, 0.5]]
    priorities_3 = [[0, 1],
                    [1, 0],
                    [2, 2]]
    crossings_3 = [[[], []],
                   [[], []],
                   [[(0,1)], []]]
    junctions[2] = jn.Junction([north_fwd[0], north_bwd[0], north_fwd[1], north_bwd[1], middle_road[0]],
                               [0, 3, 4], [1, 2], distribution_3, [], [], True, priorities_3, crossings_3)

    # Creating the main roads of the roundabout
    mainline_roads = [None for _ in range(3)]
    mainline_roads[0] = rd.Road(1, L, N, speed_8, control_points[7], initial=initial_fnc,
                        id="mainline_1fw")
    mainline_roads[1] = rd.Road(1, L, N, speed_8, control_points[7], initial=initial_fnc,
                        id="mainline_2fw")
    mainline_roads[2] = rd.Road(1, L, N, speed_8, control_points[7], initial=initial_fnc,
                        id="mainline_3fw")

    # Creating the secondary roads of the roundabout
    secondary_in = rb.RoundaboutRoad(boundary_fnc, fv.flux(torch.tensor(0.5), mainline_roads[0].Vmax[0] / L))

    roundabouts = [None for _ in range(3)]
    roundabouts[0] = rb.RoundaboutJunction(mainline_roads[2], mainline_roads[0], 0.5, secondary_in,
                                           None, True)
    roundabouts[1] = rb.RoundaboutJunction(mainline_roads[0], mainline_roads[1], 0.5, south_bwd[0],
                                           south_fwd[0], False)
    roundabouts[2] = rb.RoundaboutJunction(mainline_roads[1], mainline_roads[2], 0.5, north_bwd[0],
                                           north_fwd[0], False)
    
    roundabout = rb.Roundabout(mainline_roads, [secondary_in, south_bwd[0], north_bwd[0]],
                               [None, south_fwd[0], north_fwd[0]], roundabouts)

    # Creating the temporary network
    roads = south_fwd + south_bwd + north_fwd + north_bwd + middle_road + mainline_roads
    temp_network = nw.RoadNetwork(roads, junctions, T, [roundabout], busses = [])

    # Adding busses
    ids_1 = ["south_two_bw", "south_three_fw", "middle", "north_two_fw"]
    # ids_1 = ["mainline_1fw", "south_one_fw", "south_three_fw", "middle", "north_two_fw"]
    ids_2 = ["south_four_bw", "south_three_bw", "south_one_bw", "mainline_2fw", "mainline_3fw"]
    ids_3 = ["mainline_1fw", "mainline_2fw", "north_one_fw", "north_two_fw"]
    
    stops_1 = [("north_two_fw", 25)]
    stops_2 = [("south_one_bw", 25)]
    stops_3 = [("north_one_fw", 25)]

    times_1 = [0]
    times_2 = [0]
    times_3 = [0]
    ##################################
    bus_1 = bus.Bus(ids_1, stops_1, times_1, temp_network, id = "bus_1", start_time=0.0)
    bus_2 = bus.Bus(ids_2, stops_2, times_2, temp_network, id = "bus_2", start_time=0.0)
    bus_3 = bus.Bus(ids_3, stops_3, times_3, temp_network, id = "bus_3", start_time=0.0)

    # Creating full network
    network = nw.RoadNetwork(roads, junctions, T, [roundabout], busses = [bus_1, bus_2, bus_3])
    # network = nw.RoadNetwork(roads, junctions, T, [roundabout], busses = [bus_1])

    return network

if __name__ == "__main__":
    option = 0
    match option:
        case 0:
            import time

            network = compare_grid_size_network(T = 100, N = 2)
            start = time.time()
            network.solve_cons_law()
            print(time.time() - start)

            network = compare_grid_size_network(T = 100, N = 4)
            start = time.time()
            network.solve_cons_law()
            print(time.time() - start)

            network = compare_grid_size_network(T = 100, N = 8)
            start = time.time()
            network.solve_cons_law()
            print(time.time() - start)

        case 1: 
            network = compare_grid_size_network(T = 100, N = 2)
            densities, queues, _, _ = network.solve_cons_law()
