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