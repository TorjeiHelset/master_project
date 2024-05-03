import torch
import road as rd
import junction as jn
import traffic_lights as tl
import network as nw
import FV_schemes as fv
import roundabout as rb
import initial_and_bc as ibc
import bus

def single_lane_network(T, N , speed_limit = [torch.tensor(30.0)], 
                        control_points = [], track_grad = True):
    
    # Creating the road:
    # Configuration of the single lane
    L = 25
    N = 5
    b = 8
    if torch.is_tensor(speed_limit[0]):
        speed = [v / 3.6 for v in speed_limit]
    else:
        speed = [torch.tensor(v / 3.6) for v in speed_limit]
    for v in speed:
        v.requires_grad = track_grad

    road = rd.Road(b, L, N, speed, control_points, max_dens=1, initial=...,
                   boundary_fnc=...)
    
    # Creating the bus:
    ...

    # Creating the network:
    
def single_junction_network(T, N, speed1 = [torch.tensor(30.0)], control_points1 = [],
                            speed2 = [torch.tensor(30.0)], control_points2 = [],
                            cycle_times = [torch.tensor(60.0), torch.tensor(60.0)], 
                            track_grad = True):
    ...


def compare_grid_size_network(T, N, speed1 = [torch.tensor(50.0)], speed2 = [torch.tensor(50.0)],
                              speed3 = [torch.tensor(50.0)], cycle1 = [torch.tensor(60.0), torch.tensor(80.0)],
                              cycle2 = [torch.tensor(50.0), torch.tensor(50.0)],
                              cycle3 = [torch.tensor(70.0), torch.tensor(70.0)], 
                              densities = [0.3, 0.3, 0.3], track_grad = False):
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
                              right_pos=outer_positions[(i+1)%3])
        outer_bw[i] = rd.Road(5, L, N, speed1, [], initial=init_fncs[0], left_pos=outer_positions[(i+1)%3], 
                              right_pos=outer_positions[i])
        
        inner_fw[i] = rd.Road(2, L, N, speed2, [], initial=init_fncs[1], left_pos=outer_positions[i], 
                              right_pos=inner_positions[i])
        inner_bw[i] = rd.Road(2, L, N, speed2, [], initial=init_fncs[1], left_pos=inner_positions[i], 
                              right_pos=outer_positions[i])
        
        mainlines[i] = rd.Road(1, L, N, speed3, [], initial=init_fncs[2], left_pos=inner_positions[i],
                               right_pos=outer_positions[(i+1)%3])

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

        

if __name__ == "__main__":
    option = 1
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

            # Wrie to json file


    

