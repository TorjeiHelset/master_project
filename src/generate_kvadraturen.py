import torch
import road as rd
import junction as jn
import traffic_lights as tl
import network as nw
import FV_schemes as fv
import roundabout as rb
import initial_and_bc as ibc
import bus

'''
Ideas for visualizing the network:

- All roads have a position from (-1, 0) to (8, 9)
- If ending of id is fw or bw then the road is 2-way, and the visualization need to shift them somewhat (right/left or up/down
    depending on the direction)
- If ending is not fw, then no shifting is needed
- Map from (-1,0) - (8,9) coordinates to opengl coordinates
'''

#####################################################
# Helper functions for creating components of network
#####################################################

def create_roads_minimal_junctions_for_roundabout(N = 2, v_strand_speeds = [torch.tensor(50.0)], v_strand_controls = [],
                                                  h_w_speeds = [torch.tensor(30.0)], h_w_controls = [],
                                                  tollbod_speeds = [torch.tensor(30.0)], tollbod_controls = [],
                                                  elvegate_speeds = [torch.tensor(30.0)], elvegate_controls = [],
                                                  dronning_speeds = [torch.tensor(50.0)], dronning_controls = [],
                                                  festning_speeds = [torch.tensor(50.0)], festning_controls = [],
                                                  lundsbro_speeds = [torch.tensor(50.0)], lundsbro_controls = [],
                                                  init_densities = None, boundary_configs = None,
                                                  track_grad = True):

    '''
    Combine roads that do not need a junction between them to form a single road

    Idea to reduce memory usage - let the speed limit on each of the roads be the same tensor
    At the very least reduces the number of variables, but potentially also the memory cost(?)

    Change function to take in parameters for the speed limits as well as the control points.

    Also test out defining boundary conditions for the inflow
    For now only do piecewise constant function
    '''
    # Parameters shared for all roads
    L = 25 # Length of road
    offset = 0.1
    tilt = 0.025

    # Creating functions for the initial distribution of densities
    if init_densities is None:
        initial_fncs = [lambda x : torch.ones_like(x) * 0.2 for _ in range(7)]
    else:
        initial_fncs = [ibc.init_density(init_densities[i], 1) for i in range(7)]

    # Creating boundary functions for roads entering the network
    if boundary_configs is None:
        boundary_fncs = []
        hw_bndry = ibc.boundary_conditions(1, max_dens = 1, densities = torch.tensor([0.3, 0.1, 0.3]),
                                           time_jumps = [300, 600], in_speed = torch.tensor(30.0),
                                           L = L)
        festning_bndry = ibc.boundary_conditions(1, max_dens = 2, densities = torch.tensor([0.05, 0.6, 0.1]),
                                           time_jumps = [100, 400], in_speed = torch.tensor(50.0),
                                           L = L)
        lundsbro_bndry = ibc.boundary_conditions(1, max_dens = 2, densities = torch.tensor([0.15, 0.2, 0.15]),
                                           time_jumps = [200, 800], in_speed = torch.tensor(50.0),
                                           L = L)
        boundary_fncs = [hw_bndry, festning_bndry, lundsbro_bndry]
    else:
        boundary_fncs = []
        hw_config = boundary_configs[0]
        hw_bndry = ibc.boundary_conditions(hw_config["type"], max_dens=hw_config["max_dens"], 
                                           densities = torch.tensor(hw_config["densities"]),
                                           time_jumps = hw_config["jumps"], in_speed = torch.tensor(hw_config["in_speed"]),
                                           L = L)
        festning_config = boundary_configs[1]
        festning_bndry = ibc.boundary_conditions(festning_config["type"], max_dens=festning_config["max_dens"], 
                                           densities = torch.tensor(festning_config["densities"]),
                                           time_jumps = festning_config["jumps"], in_speed = torch.tensor(festning_config["in_speed"]),
                                           L = L)
        lundsbro_config = boundary_configs[2]
        lundsbro_bndry = ibc.boundary_conditions(lundsbro_config["type"], max_dens=lundsbro_config["max_dens"], 
                                           densities = torch.tensor(lundsbro_config["densities"]),
                                           time_jumps = lundsbro_config["jumps"], in_speed = torch.tensor(lundsbro_config["in_speed"]),
                                           L = L)
        boundary_fncs = [hw_bndry, festning_bndry, lundsbro_bndry]

    ########################################################
    # Vestre Strandgate:
    ########################################################

    # junctions 1 and 2 not needed, all other junctions needed
    # No inflow conditions needed
    v_strand_fw = [None] * 6
    v_strand_bw = [None] * 6
    if torch.is_tensor(v_strand_speeds[0]):
        v_strand_speed_limit = [v / 3.6 for v in v_strand_speeds]
    else:
        v_strand_speed_limit = [torch.tensor(v / 3.6) for v in v_strand_speeds]
    for v in v_strand_speed_limit:
        v.requires_grad = track_grad

    v_strand_control_points = v_strand_controls
    for i in range(6):
        if i == 0:
            # Combine first three to form one road
            # b = 3 since three roads are combined
            v_strand_fw[0] = rd.Road(6, L, N, v_strand_speed_limit, v_strand_control_points, left_pos=(0, 0), right_pos=(0, 3-offset),
                                     id="v_strand_" + str(1) + "fw", initial=initial_fncs[0], boundary_fnc=None, max_dens=2)
            v_strand_bw[0] = rd.Road(6, L, N, v_strand_speed_limit, v_strand_control_points, left_pos=(0, 3-offset), right_pos=(0, 0),
                                     id="v_strand_" + str(1) + "bw",initial=initial_fncs[0], boundary_fnc=None, max_dens=2)
        else:
            v_strand_fw[i] = rd.Road(2, L, N, v_strand_speed_limit, v_strand_control_points, left_pos=(0, i+2+offset), right_pos=(0, i+3-offset),
                                     id="v_strand_" + str(i+1) + "fw",initial=initial_fncs[0], boundary_fnc=None, max_dens=2)
            v_strand_bw[i] = rd.Road(2, L, N, v_strand_speed_limit, v_strand_control_points, left_pos=(0, i+3-offset), right_pos=(0, i+2+offset),
                                     id="v_strand_" + str(i+1) + "bw",initial=initial_fncs[0], boundary_fnc=None, max_dens=2)

    ########################################################
    # Henrik Wergeland:
    ########################################################

    # junction 1 needed, junctions 2 and 3 not needed, junction 4 needed
    # Boundary inflow condition needed for first road
    h_w = [None] * 2
    if torch.is_tensor(h_w_speeds[0]):
        h_w_speed = [v / 3.6 for v in h_w_speeds]
    else:
        h_w_speed = [torch.tensor(v / 3.6) for v in h_w_speeds]

    for v in h_w_speed:
        v.requires_grad = track_grad
    h_w_control_points = h_w_controls
    
    h_w[0] = rd.Road(2, L, N, h_w_speed, h_w_control_points, left_pos=(-0.5, 3), right_pos=(0-offset, 3),
                     id="h_w_" + str(1),initial=initial_fncs[1], boundary_fnc=boundary_fncs[0])
    h_w[1] = rd.Road(2*6, L, N, h_w_speed, h_w_control_points, left_pos=(0+offset, 3), right_pos=(3-offset+6*tilt, 3),
                     id="h_w_" + str(2),initial=initial_fncs[1], boundary_fnc=None)

    ########################################################
    # Tollbodgata:
    ########################################################

    # Incoming junction needed, junctions 2 and 3 not needed, last junction needed
    # No boundary function needed
    tollbod_fw = [None] * 2
    tollbod_bw = [None] * 2
    if torch.is_tensor(tollbod_speeds[0]):
        tollbod_speed = [v / 3.6 for v in tollbod_speeds]
    else:
        tollbod_speed = [torch.tensor(v / 3.6) for v in tollbod_speeds]
    for v in tollbod_speed:
        v.requires_grad = track_grad
    tollbod_control_points = tollbod_controls
    tollbod_bw[0] = rd.Road(2*6, L, N, tollbod_speed, tollbod_control_points, left_pos=(3-offset+2*tilt, 7), right_pos=(0+offset, 7),
                            id="tollbod_" + str(1) + "bw", initial=initial_fncs[2], boundary_fnc=None)
    tollbod_bw[1] = rd.Road(2*6, L, N, tollbod_speed, tollbod_control_points, left_pos=(6-offset, 7), right_pos=(3+offset+2*tilt, 7),
                            id="tollbod_" + str(2) + "bw", initial=initial_fncs[2], boundary_fnc=None)
    tollbod_fw[1] = rd.Road(2*6, L, N, tollbod_speed, tollbod_control_points, left_pos=(3+offset+2*tilt, 7), right_pos=(6-offset, 7),
                            id="tollbod_" + str(2) + "fw", initial=initial_fncs[2], boundary_fnc=None)

    ########################################################
    # Elvegata:
    ########################################################

    # Both junctions needed (strictly speaking elvegate into tollbodgata is not needed, but keep¨
    # since the road changes direction/name)
    # No boundary function needed
    if torch.is_tensor(elvegate_speeds[0]):
        elvegate_speed = [v / 3.6 for v in elvegate_speeds]
    else:
        elvegate_speed = [torch.tensor(v / 3.6) for v in elvegate_speeds]
    for v in elvegate_speed:
        v.requires_grad = track_grad
    elvegate_control_points = elvegate_controls
    elvegata_fw = [rd.Road(2, L, N, elvegate_speed, elvegate_control_points, left_pos=(6, 7+offset), right_pos=(6, 8-offset),
                           id="elvegata_fw", initial=initial_fncs[3], boundary_fnc=None)]
    elvegata_bw = [rd.Road(2, L, N, elvegate_speed, elvegate_control_points, left_pos=(6, 8-offset), right_pos=(6, 7+offset),
                           id="elvegata_bw", initial=initial_fncs[3], boundary_fnc=None)]

    ########################################################
    # Dronningens gate:
    ########################################################

    # Incoming already accounted for, junction 2 not needed, junctions 3 and 4 needed,
    # junction 5 and 6 not needed, last junction needed
    # No boundary function needed
    dronning_fw = [None] * 3
    dronning_bw = [None] * 3
    if torch.is_tensor(dronning_speeds[0]):
        dronning_speed = [v / 3.6 for v in dronning_speeds]
    else:
        dronning_speed = [torch.tensor(v / 3.6) for v in dronning_speeds]
    for v in dronning_speed:
        v.requires_grad = track_grad
    dronning_control_points = dronning_controls
    dronning_fw[0] = rd.Road(2*4, L, N, dronning_speed, dronning_control_points, left_pos=(0+offset, 8), right_pos=(2-offset, 8),
                             id="dronning_" + str(1) + "fw", initial=initial_fncs[4], boundary_fnc=None, max_dens=1)
    dronning_bw[0] = rd.Road(2*4, L, N, dronning_speed, dronning_control_points, left_pos=(2-offset, 8), right_pos=(0+offset, 8),
                             id="dronning_" + str(1) + "bw", initial=initial_fncs[4], boundary_fnc=None, max_dens=1)
    dronning_fw[1] = rd.Road(4, L, N, dronning_speed, dronning_control_points, left_pos=(2+offset, 8), right_pos=(3-offset, 8),
                             id="dronning_" + str(2) + "fw", initial=initial_fncs[4], boundary_fnc=None, max_dens=1)
    dronning_bw[1] = rd.Road(4, L, N, dronning_speed, dronning_control_points, left_pos=(3-offset, 8), right_pos=(2+offset, 8),
                             id="dronning_" + str(2) + "bw", initial=initial_fncs[4], boundary_fnc=None, max_dens=1)
    dronning_fw[2] = rd.Road(2*6, L, N, dronning_speed, dronning_control_points, left_pos=(3+offset, 8), right_pos=(6-offset, 8),
                             id="dronning_" + str(3) + "fw", initial=initial_fncs[4], boundary_fnc=None, max_dens=1)
    dronning_bw[2] = rd.Road(2*6, L, N, dronning_speed, dronning_control_points, left_pos=(6-offset, 8), right_pos=(3+offset, 8),
                             id="dronning_" + str(3) + "bw", initial=initial_fncs[4], boundary_fnc=None, max_dens=1)

    ########################################################
    # Festningsgata:
    ########################################################

    # junction 1 needed, junction 2 not needed, junctions 3 and 4 needed, 5 and 6 not needed, 7 and 8 needed
    # In picture, festningsgate is not completly vertical. The top is leaning to the right slightly
    # Try 0.05 shift for every piece of road
    # Boundary function needed on last backwards road

    festning_fw = [None] * 6
    festning_bw = [None] * 6
    if torch.is_tensor(festning_speeds[0]):
        festning_speed = [v / 3.6 for v in festning_speeds]
    else:
        festning_speed = [torch.tensor(v / 3.6) for v in festning_speeds]

    for v in festning_speed:
        v.requires_grad = track_grad
    festning_control_points = festning_controls

    # Add (9-i)*tilt to the roads
    for i in range(6):
        if i == 0:
            festning_fw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(9-i)*tilt, i+offset),
                                     right_pos=(3+(8-i)*tilt, i+1-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fncs[5], boundary_fnc=None,max_dens=2)
            festning_bw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(8-i)*tilt, i+1-offset),
                                     right_pos=(3+(9-i)*tilt, i+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fncs[5], boundary_fnc=None,max_dens=2)
        elif i == 1:
            festning_fw[i] = rd.Road(4, L, N, festning_speed, festning_control_points, left_pos=(3+(9-i)*tilt, i+offset),
                                     right_pos=(3+(7-i)*tilt, i+2-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fncs[5], boundary_fnc=None,max_dens=2)
            festning_bw[i] = rd.Road(4, L, N, festning_speed, festning_control_points, left_pos=(3+(7-i)*tilt, i+2-offset),
                                     right_pos=(3+(9-i)*tilt, i+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fncs[5], boundary_fnc=None,max_dens=2)
        elif i == 2:
            festning_fw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(8-i)*tilt, i+1+offset),
                                     right_pos=(3+(7-i)*tilt, i+2-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fncs[5], boundary_fnc=None,max_dens=2)
            festning_bw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(7-i)*tilt, i+2-offset),
                                     right_pos=(3+(8-i)*tilt, i+1+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fncs[5], boundary_fnc=None,max_dens=2)
        elif i == 3:
            festning_fw[i] = rd.Road(6, L, N, festning_speed, festning_control_points, left_pos=(3+(8-i)*tilt, i+1+offset),
                                     right_pos=(3+(5-i)*tilt, i+4-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fncs[5], boundary_fnc=None,max_dens=2)
            festning_bw[i] = rd.Road(6, L, N, festning_speed, festning_control_points, left_pos=(3+(5-i)*tilt, i+4-offset),
                                     right_pos=(3+(8-i)*tilt, i+1+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fncs[5], boundary_fnc=None,max_dens=2)
        elif i == 4:
            # i = 4
            # i = 5
            festning_fw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(6-i)*tilt, i+3+offset),
                                     right_pos=(3+(5-i)*tilt, i+4-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fncs[5],boundary_fnc=None, max_dens=2)
            festning_bw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(5-i)*tilt, i+4-offset),
                                     right_pos=(3+(6-i)*tilt, i+3+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fncs[5],boundary_fnc=None, max_dens=2)
        else:
            festning_fw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(6-i)*tilt, i+3+offset),
                                     right_pos=(3+(5-i)*tilt, i+4-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fncs[5],boundary_fnc=None, max_dens=2)
            festning_bw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(5-i)*tilt, i+4-offset),
                                     right_pos=(3+(6-i)*tilt, i+3+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fncs[5], boundary_fnc=boundary_fncs[1], max_dens=2)
    
    ########################################################
    # Lundsbroa:
    ########################################################

    # Only one road
    # Boundary condition needed on backward road
    if torch.is_tensor(lundsbro_speeds[0]):
        lundsbro_speed = [v / 3.6 for v in lundsbro_speeds]
    else:
        lundsbro_speed = [torch.tensor(v / 3.6) for v in lundsbro_speeds]
        
    for v in lundsbro_speed:
        v.requires_grad = track_grad
    lundsbro_control_points = lundsbro_controls

    lundsbro_fw = [rd.Road(4, L, N, lundsbro_speed, lundsbro_control_points, left_pos=(6+offset, 8), right_pos=(7, 7.8),
                           id="lundsbro_fw", initial=initial_fncs[6], boundary_fnc=None)]
    lundsbro_bw = [rd.Road(4, L, N, lundsbro_speed, lundsbro_control_points, left_pos=(7, 7.8), right_pos=(6+offset, 8),
                           id="lundsbro_bw", initial=initial_fncs[6], boundary_fnc=boundary_fncs[2], max_dens=2)]

    return v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, lundsbro_fw, lundsbro_bw

def create_roundabouts(v_strand_fw, v_strand_bw, festning_fw, festning_bw,
                       speed1 = [torch.tensor(50.0)], speed2 = [torch.tensor(50.0)], 
                       controls1 = [], controls2 = [], N = 2,
                       initial_densities = None, inflow_configs = None, track_grad = True):

    # Shared parameters for both roundabouts:
    offset = 0.1
    tilt = 0.025
    L = 25 # Length of road

    # Initial distribution functions:
    if initial_densities is None:
        initial_fncs = [lambda x : torch.ones_like(x) * 0.2 for _ in range(7)]
    else:
        initial_fncs = [ibc.init_density(initial_densities[-2], 1), 
                        ibc.init_density(initial_densities[-1], 1)]
        
    # Inflow functions:
    if inflow_configs is None:
        # Vestre strandgate:
        secondary_gamma = torch.tensor(50 / (25 * 3.6))
        max_inflow = fv.flux(torch.tensor(0.5), secondary_gamma)
        secondary_big_gamma = torch.tensor(60 / (25 * 3.6))
        max_inflow_big = fv.flux(torch.tensor(0.5), secondary_big_gamma)
        rho_1, rho_2, rho_3 = torch.tensor(0.25),torch.tensor(0.05), torch.tensor(0.05)
        inflow_1 = lambda t : fv.flux(rho_1, secondary_gamma)
        inflow_2 = lambda t : fv.flux(rho_2, secondary_gamma)
        inflow_3 = lambda t : fv.flux(rho_3, secondary_big_gamma)

        # Festningsgate:
        rho_4 = torch.tensor(0.15)
        inflow_4 = lambda t : fv.flux(rho_4, secondary_gamma)

        inflows = [inflow_1, inflow_2, inflow_3, inflow_4, inflow_4]
        max_inflows = [max_inflow, max_inflow, max_inflow_big, max_inflow, max_inflow]

    else:
        # Use config to create the inflow conditions:
        inflows = []
        max_inflows = []
        speeds = inflow_configs["speeds"]
        configs = inflow_configs["inflows"]

        for s, c in zip(speeds, configs):
            max_dens = c["max_dens"]
            gamma = torch.tensor(s / (L*3.6))
            max_inflow = fv.flux(torch.tensor(0.5), gamma) * max_dens
            fluxes = [fv.flux(torch.tensor(rho), gamma) for rho in c["densities"]]
            inflow = ibc.boundary_conditions(3, max_dens, c["densities"], fluxes, c["jumps"],
                                             in_speed = s, L = L)
            inflows.append(inflow)
            max_inflows.append(max_inflow)
            

    ##############################################
    # Vestre Strandgate roundabout:
    ##############################################

    # Four arm roundabout with 4 junctions, where 2 roads are a part of simulation
    # Secondary roads can be described by a small object with a queue and an inflow condition
    # Creating the mainline
    if torch.is_tensor(speed1[0]):
        main_speed_limit = [v / 3.6 for v in speed1]
    else:
        main_speed_limit = [torch.tensor(v / 3.6) for v in speed1]
    for v in main_speed_limit:
        v.requires_grad = track_grad

    vs_main_1 = rd.Road(1, L, N, main_speed_limit, controls1, initial=initial_fncs[0], left_pos=(0.05,0), right_pos=(0.25, -0.25),
                        id="vs_mainline_1")
    vs_main_2 = rd.Road(1, L, N, main_speed_limit, controls1, initial=initial_fncs[0], left_pos=(0.25, -0.25), right_pos=(0.08, -0.55),
                        id="vs_mainline_2")
    vs_main_3 = rd.Road(1, L, N, main_speed_limit, controls1, initial=initial_fncs[0], left_pos=(0.05, -0.55), right_pos=(-0.15, -0.3),
                        id="vs_mainline_3")
    vs_main_4 = rd.Road(1, L, N, main_speed_limit, controls1, initial=initial_fncs[0], left_pos=(-0.13, -0.28), right_pos=(-0.02,0),
                        id="vs_mainline_4")

    # Creating the secondary incoming roads
    # These are not proper road objects, but rather a smal classs containing only
    # the queue length and the inflow function
    # Inflow function should be specified by some config object...

    # Need secondary speeds
    vs_secondary_1 = rb.RoundaboutRoad(inflows[0], max_inflows[0])
    vs_secondary_2 = rb.RoundaboutRoad(inflows[1], max_inflows[1])
    vs_secondary_3 = rb.RoundaboutRoad(inflows[2], max_inflows[2])
    # Creating the roundabout junctions
    vs_jnc_1 = rb.RoundaboutJunction(vs_main_4, vs_main_1, 0.6, v_strand_bw, v_strand_fw, queue_junction = False)
    vs_jnc_2 = rb.RoundaboutJunction(vs_main_1, vs_main_2, 0.6, vs_secondary_1, None, queue_junction = True)
    vs_jnc_3 = rb.RoundaboutJunction(vs_main_2, vs_main_3, 0.6, vs_secondary_2, None, queue_junction = True)
    vs_jnc_4 = rb.RoundaboutJunction(vs_main_3, vs_main_4, 0.6, vs_secondary_3, None, queue_junction = True)
    vs_junctions = [vs_jnc_1, vs_jnc_2, vs_jnc_3, vs_jnc_4]
    vs_roundabout = rb.Roundabout([vs_main_1, vs_main_2, vs_main_3, vs_main_4],
                           [v_strand_fw, vs_secondary_1, vs_secondary_2, vs_secondary_3],
                           [v_strand_bw, None, None, None],vs_junctions)

    ##############################################
    # Festningsgate roundabout:
    ##############################################

    if torch.is_tensor(speed2[0]):
        secondary_speed_limit = [v / 3.6 for v in speed2]
    else:
        secondary_speed_limit = [torch.tensor(v / 3.6) for v in speed2]
        
    for v in secondary_speed_limit:
        v.requires_grad = track_grad

    # Similar to the vestre strandgate roundabout, but with only three incoming roads
    fn_main_1 = rd.Road(1, L, N, secondary_speed_limit, controls2, initial=initial_fncs[1], left_pos=(3+9*tilt+0.04, offset), right_pos=(3+9*tilt+0.15, offset-0.4),
                        id="fn_mainline_1")
    fn_main_2 = rd.Road(1, L, N, secondary_speed_limit, controls2, initial=initial_fncs[1], left_pos=(3+9*tilt+0.15, offset-0.5), right_pos=(3+9*tilt-0.15, offset-0.5),
                        id="fn_mainline_2")
    fn_main_3 = rd.Road(1, L, N, secondary_speed_limit, controls2, initial=initial_fncs[1], left_pos=(3+9*tilt-0.15, offset-0.4), right_pos=(3+9*tilt-0.04, offset),
                        id="fn_mainline_3")
    
    # Creating the secondary incoming roads
    
    fn_secondary_1 = rb.RoundaboutRoad(inflows[3], max_inflows[3])
    fn_secondary_2 = rb.RoundaboutRoad(inflows[4], max_inflows[4])
    # Creating the roundabout junctions
    fn_jnc_1 = rb.RoundaboutJunction(fn_main_3, fn_main_1, 0.6, festning_bw, festning_fw, queue_junction = False)
    fn_jnc_2 = rb.RoundaboutJunction(fn_main_1, fn_main_2, 0.6, fn_secondary_1, None, queue_junction = True)
    fn_jnc_3 = rb.RoundaboutJunction(fn_main_2, fn_main_3, 0.6, fn_secondary_2, None, queue_junction = True)
    fn_junctions = [fn_jnc_1, fn_jnc_2, fn_jnc_3]
    fn_roundabout = rb.Roundabout([fn_main_1, fn_main_2, fn_main_3],
                                  [festning_fw, fn_secondary_1, fn_secondary_2],
                                  [festning_bw, None, None], fn_junctions)

    return [vs_main_1, vs_main_2, vs_main_3, vs_main_4,fn_main_1, fn_main_2, fn_main_3], [vs_roundabout, fn_roundabout]

def convert_junction_input(vs_cycles, hw_cycle, tollbod_cycle, dronning_cycles, 
                           festning_cycles, track_grad):
    if track_grad:
        # Set requires grad = True for all cycle times:
        new_vs = []
        for cycle in vs_cycles:
            new_cycle = []
            for c in cycle:
                if torch.is_tensor(c):
                    c.requires_grad = True
                    new_cycle.append(c)
                else:
                    c = torch.tensor(c, requires_grad=True)
                    new_cycle.append(c)
            new_vs.append(new_cycle)
        vs_cycles = new_vs

        new_hw = []
        for c in hw_cycle:
            if torch.is_tensor(c):
                c.requires_grad = True
                new_hw.append(c)
            else:
                c = torch.tensor(c, requires_grad=True)
                new_hw.append(c)
        hw_cycle = new_hw

        new_tollbod = []
        for c in tollbod_cycle:
            if torch.is_tensor(c):
                c.requires_grad = True
                new_tollbod.append(c)
            else:
                c = torch.tensor(c, requires_grad=True)
                new_tollbod.append(c)
        tollbod_cycle = new_tollbod

        new_dronning = []
        for cycle in dronning_cycles:
            new_cycle = []
            for c in cycle:
                if torch.is_tensor(c):
                    c.requires_grad = True
                    new_cycle.append(c)
                else:
                    c = torch.tensor(c, requires_grad=True)
                    new_cycle.append(c)
            new_dronning.append(new_cycle)
        dronning_cycles = new_dronning

        new_festning = []
        for cycle in festning_cycles:
            new_cycle = []
            for c in cycle:
                if torch.is_tensor(c):
                    c.requires_grad = True
                    new_cycle.append(c)
                else:
                    c = torch.tensor(c, requires_grad=True)
                    new_cycle.append(c)
            new_festning.append(new_cycle)
        festning_cycles = new_festning
    else:
        # Set requires grad = False for all cycle times:
        new_vs = []
        for cycle in vs_cycles:
            new_cycle = []
            for c in cycle:
                if torch.is_tensor(c):
                    c.requires_grad = False
                    new_cycle.append(c)
                else:
                    c = torch.tensor(c, requires_grad=False)
                    new_cycle.append(c)
            new_vs.append(new_cycle)
        vs_cycles = new_vs

        new_hw = []
        for c in hw_cycle:
            if torch.is_tensor(c):
                c.requires_grad = False
                new_hw.append(c)
            else:
                c = torch.tensor(c, requires_grad=False)
                new_hw.append(c)
        hw_cycle = new_hw

        new_tollbod = []
        for c in tollbod_cycle:
            if torch.is_tensor(c):
                c.requires_grad = False
                new_tollbod.append(c)
            else:
                c = torch.tensor(c, requires_grad=False)
                new_tollbod.append(c)
        tollbod_cycle = new_tollbod

        new_dronning = []
        for cycle in dronning_cycles:
            new_cycle = []
            for c in cycle:
                if torch.is_tensor(c):
                    c.requires_grad = False
                    new_cycle.append(c)
                else:
                    c = torch.tensor(c, requires_grad=False)
                    new_cycle.append(c)
            new_dronning.append(new_cycle)
        dronning_cycles = new_dronning

        new_festning = []
        for cycle in festning_cycles:
            new_cycle = []
            for c in cycle:
                if torch.is_tensor(c):
                    c.requires_grad = False
                    new_cycle.append(c)
                else:
                    c = torch.tensor(c, requires_grad=False)
                    new_cycle.append(c)
            new_festning.append(new_cycle)
        festning_cycles = new_festning

    return vs_cycles, hw_cycle, tollbod_cycle, dronning_cycles, festning_cycles

def create_minimal_junctions_w_params(v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw,
                                      elvegata_fw, elvegata_bw, dronning_fw, dronning_bw,
                                      festning_fw, festning_bw, lundsbro_fw, lundsbro_bw,
                                      vs_cycles = [[torch.tensor(50.0), torch.tensor(100.0)], [torch.tensor(50.0), torch.tensor(100.0)],
                                                   [torch.tensor(50.0), torch.tensor(100.0)], [torch.tensor(50.0), torch.tensor(100.0)],
                                                   [torch.tensor(50.0), torch.tensor(100.0)], [torch.tensor(60.0), torch.tensor(60.0)]],
                                      hw_cycle = [torch.tensor(100.0), torch.tensor(50.0)], tollbod_cycle = [torch.tensor(50.0), torch.tensor(100.0)],
                                      dronning_cycles = [[torch.tensor(60.0), torch.tensor(60.0)], [torch.tensor(100.0), torch.tensor(50.0)],
                                                         [torch.tensor(50.0), torch.tensor(100.0)]],
                                      festning_cycles = [[torch.tensor(60.0), torch.tensor(60.0)], [torch.tensor(60.0), torch.tensor(60.0)]],
                                      track_grad = True):
    
    vs_cycles, hw_cycle, tollbod_cycle, dronning_cycles, festning_cycles = convert_junction_input(vs_cycles, hw_cycle, tollbod_cycle, 
                                                                                                  dronning_cycles, festning_cycles, track_grad)

    
    # Creating junctions for vestre strandgate
    v_strand_jncs = [None] * 6
    for i in range(6):
        if i == 0:
            # strandgate to henrik Wergeland 4 way + traffic light
            distribution = [[0.7, 0.0, 0.3],
                            [0.0, 0.7, 0.3],
                            [0.3, 0.3, 0.4]]
            priorities = [[1, 0, 2],
                          [0, 1, 1],
                          [2, 2, 3]]
            crossings = [[[], [], [(1,1)]],
                         [[], [], []],
                         [[], [(0,0)], [(0,0), (1,1)]]]
            trafficlight = tl.CoupledTrafficLightContinuous(True, [0,3], [1,2,5], [4], [1,2,5],
                                                            vs_cycles[i])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            h_w[0], h_w[1]], [0,3,4], [1,2,5],
                                            distribution, [], [trafficlight], duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)

        elif i < 4:
            # strandgate to strandgate - trafficlight - no r.o.w.
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    vs_cycles[i])

            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])
        elif i==4:
            # strandgate to tollbod 1 way + traffic light
            distribution = [[1.0, 0.0],
                            [0.0, 1.0],
                            [0.3, 0.7]]
            priorities = [[1, 0],
                          [0, 1],
                          [2, 2]]
            crossings = [[[], []],
                         [[], []],
                         [[(1,1)], []]]
            
            trafficlight = tl.CoupledTrafficLightContinuous(True, [0,3], [1,2], [4], [1,2],
                                                            vs_cycles[i])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            tollbod_bw[0]], [0,3,4], [1,2],
                                            distribution, [], [trafficlight],
                                            duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)
        else:
            # strandgate to dronningens gate + traffic light
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    vs_cycles[i])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], dronning_fw[0],
                                            v_strand_bw[i], dronning_bw[0]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])

    # For Henrik Wergelands:
    # Junctions where henrik Wergeland is connected (and nt vestre strandgate)
    h_w_jncs = [None]
    #henrik Wergeland to festningsgata 2/3 + traffic light
    # r.o.w. needed
    distribution = [[0.5, 0.5],
                    [1.0, 0.0],
                    [0.0, 1.0]]
    priorities = [[2, 2],
                    [1, 0],
                    [0, 1]]
    crossings = [[[], [(1,0)]],
                    [[], []],
                    [[], []]]
    
    trafficlight = tl.CoupledTrafficLightContinuous(True, [0], [2,3], [1,4], [2,3],
                                                    hw_cycle)


    h_w_jncs[0] = jn.Junction([h_w[1], festning_fw[1], festning_fw[2],
                                festning_bw[1], festning_bw[2]],
                                [0,1,4], [2,3], distribution, [], [trafficlight],
                                duty_to_gw=True, priorities=priorities, crossing_connections=crossings)

    # For Tollbodgata:
    # Junctions where tollbodgata is connected (and not vestre strandgate or henrik Wergeland)
    tollbod_jncs = [None] * 2
    for i in range(2):
        if i == 0:
            # tollbodgata to festningsgata 4/5 + traffic light - r.o.w. needed
            distribution = [[0.1, 0.8, 0.0, 0.1],
                            [0.1, 0.0, 0.8, 0.1],
                            [0.5, 0.25, 0.25, 0.0]]
            priorities = [[1, 1, 0, 2],
                          [2, 0, 1, 1],
                          [3, 2, 2, 0]]
            crossings = [[[], [], [], []],
                         [[(0,1)], [], [], []],
                         [[(0,1), (1,2)], [(1,2)], [], []]]
            trafficlight = tl.CoupledTrafficLightContinuous(True, [1,4], [0,2,3,5], [6], [0,2,3,5],
                                                            tollbod_cycle)
            tollbod_jncs[i] = jn.Junction([tollbod_bw[i], festning_fw[3], festning_fw[4],
                                           festning_bw[3], festning_bw[4], tollbod_fw[i+1],
                                           tollbod_bw[i+1]], [1,4,6], [0,2,3,5],
                                           distribution, [], [trafficlight], duty_to_gw=True,
                                           priorities=priorities, crossing_connections=crossings)
        else:
            # tollbodgata to elvegata + no traffic light - r.o.w. not relevant
            # Junction kept just since the road changes direction and name
            tollbod_jncs[i] = jn.Junction([tollbod_fw[i], elvegata_fw[0],
                                            tollbod_bw[i], elvegata_bw[0]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

    # For Dronningens gate:
    # Junctions where dronningens gate is connected (and not vestre strandgate, henrik Wergeland or tollbodgata)
    dronning_jncs = [None] * 3
    for i in range(3):
        if i == 0:
            # dronningens gate to dronningens gate - trafficlight - r.o.w. not relevant
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    dronning_cycles[i])
            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])

        elif i == 1:
            # dronningens gate to festningsgata 5/6 + traffic light - r.o.w. needed
            # Double check this junction...
            distribution = [[0.7, 0.0, 0.15, 0.15],
                            [0.0, 0.7, 0.15, 0.15],
                            [0.15, 0.15, 0.7, 0.0],
                            [0.15, 0.15, 0.0, 0.7]]
            priorities = [[1, 0, 1, 3],
                          [0, 1, 3, 1],
                          [2, 3, 2, 0],
                          [3, 2, 0, 2]]
            crossings = [[[], [], [], [(1,1), (2,2), (2,0)]],
                         [[], [], [(0,0), (3,1), (3,3)], []],
                         [[], [(1,1), (3,3)], [(0,0), (1,1)], []],
                         [[(0,0)], [], [], [(0,0), (1,1)]]]
            trafficlight = tl.CoupledTrafficLightContinuous(True, [0,3], [1,2,5,6], [4,7], [1,2,5,6],
                                                            dronning_cycles[i])

            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1],
                                            festning_fw[4], festning_fw[5],
                                            festning_bw[4], festning_bw[5]],
                                            [0,3,4,7], [1,2,5,6], distribution, [], [trafficlight],
                                            duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)
        else:
            # dronningens gate to elvegata and lundsbro + traffic light - also r.o.w.
            distribution = [[0.7, 0.0, 0.3],
                            [0.0, 0.7, 0.3],
                            [0.5, 0.5, 0.0]]
            priorities = [[1, 0, 2],
                          [0, 1, 1],
                          [2, 2, 0]]
            crossings = [[[], [], [(1,1)]],
                         [[], [], []],
                         [[(1,1)], [], []]]
            trafficlight = tl.CoupledTrafficLightContinuous(True, [0,3], [1,2,5], [4], [1,2,5],
                                                            dronning_cycles[i])

            dronning_jncs[i] = jn.Junction([dronning_fw[i], lundsbro_fw[0],
                                            dronning_bw[i], lundsbro_bw[0],
                                            elvegata_fw[0], elvegata_bw[0]],
                                            [0,3,4], [1,2,5], distribution, [], [trafficlight],
                                            duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)
    festning_jncs = [None] * 2
    # festningsgata to festningsgata - trafficlight - no r.o.w
    trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                            festning_cycles[0])
    festning_jncs[0] = jn.Junction([festning_fw[0], festning_fw[1],
                                    festning_bw[0], festning_bw[1]],
                                    [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])
    # festningsgata 4 to festningsgata 5 - trafficlight - shifted one index- no r.o.w
    # This appear to be the problem junction, but no clear issues...
    trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    festning_cycles[1])
    festning_jncs[1] = jn.Junction([festning_fw[2], festning_fw[3],
                                    festning_bw[2], festning_bw[3]],
                                    [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])

    return v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs

def create_busses(fwd_schedules, bwd_schedules, network):
    # The busses share the same stops
    ids_fw = ["vs_mainline_3", "vs_mainline_4", "v_strand_1fw", "h_w_2", "festning_3fw", "festning_4fw", "tollbod_2fw",
        "elvegata_fw", "lundsbro_fw"]
    stops_fw = [("h_w_2", 130), ("festning_4fw", 35), ("tollbod_2fw", 30), ("tollbod_2fw", 260)]

    ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_2bw", "tollbod_1bw", "v_strand_5bw",
        "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw","vs_mainline_1", "vs_mainline_2"]
    stops_bw = [("tollbod_2bw", 45), ("tollbod_1bw", 80), ("tollbod_1bw", 235), ("v_strand_1bw", 25)]

    busses = []
    id = 1
    for schedule in fwd_schedules:
        # Add forward going bus with schedule specified in config
        start = schedule["start_time"]
        times = schedule["times"]
        new_bus = bus.Bus(ids_fw, stops_fw, times, network, id = str(id), start_time=start)
        busses.append(new_bus)
        id += 1

    for schedule in bwd_schedules:
        # Add backward going bus with schedule
        start = schedule["start_time"]
        times = schedule["times"]
        new_bus = bus.Bus(ids_bw, stops_bw, times, network, id = str(id), start_time=start)
        busses.append(new_bus)
        id += 1

    return busses

###########################################
# Functions for creating full road networks
###########################################

def generate_kvadraturen_w_roundabout(T):
    # Create the roads
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_minimal_junctions_for_roundabout()

    # Create the junctions
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_minimal_junctions_w_params(v_strand_fw, v_strand_bw, h_w,
                                                                                                 tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                 elvegata_bw, dronning_fw, dronning_bw,
                                                                                                 festning_fw, festning_bw, lundsbro_fw,
                                                                                                 lundsbro_bw)

    # Create the roundabouts
    mainline_roads, roundabouts = create_roundabouts(v_strand_fw[0], v_strand_bw[0],
                                                     festning_fw[0], festning_bw[0])

    # Create the network
    roads = v_strand_fw + v_strand_bw + h_w + tollbod_fw[1:] + tollbod_bw + elvegata_fw + \
            elvegata_bw + dronning_fw + dronning_bw + festning_fw + festning_bw + lundsbro_fw + \
            lundsbro_bw + mainline_roads
    junctions = v_strand_jncs + h_w_jncs + tollbod_jncs + dronning_jncs + festning_jncs

    network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts)

    return network

def generate_kvadraturen_w_bus(T, track_grad=True):

    # Create the roads - should be possible to pass speed limits as arguments
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_minimal_junctions_for_roundabout(track_grad=track_grad)

    # Create the junctions - should be possible to pass cycle times of traffic lights as arguments
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_minimal_junctions_w_params(v_strand_fw, v_strand_bw, h_w,
                                                                                                 tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                 elvegata_bw, dronning_fw, dronning_bw,
                                                                                                 festning_fw, festning_bw, lundsbro_fw,
                                                                                                 lundsbro_bw, track_grad=track_grad)

    # Create the roundabouts - pass speed limits of mainline roads as argument?
    mainline_roads, roundabouts = create_roundabouts(v_strand_fw[0], v_strand_bw[0],
                                                     festning_fw[0], festning_bw[0], track_grad=track_grad)

    # Create the network
    roads = v_strand_fw + v_strand_bw + h_w + tollbod_fw[1:] + tollbod_bw + elvegata_fw + \
            elvegata_bw + dronning_fw + dronning_bw + festning_fw + festning_bw + lundsbro_fw + \
            lundsbro_bw + mainline_roads
    junctions = v_strand_jncs + h_w_jncs + tollbod_jncs + dronning_jncs + festning_jncs

    temp_network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts)

    # Adding the busses
    ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_2bw", "tollbod_1bw", "v_strand_5bw",
          "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw","vs_mainline_1", "vs_mainline_2"]
    stops_bw = [("tollbod_2bw", 50), ("tollbod_1bw", 90), ("tollbod_1bw", 230), ("v_strand_1bw", 25)]
    stops_bw = [("tollbod_2bw", 45), ("tollbod_1bw", 80), ("tollbod_1bw", 235), ("v_strand_1bw", 25)]
    # times_bw = [40, 130, 190, 250]
    times_bw = [4, 130, 190, 250]
    bus_bw = bus.Bus(ids_bw, stops_bw, times_bw, temp_network, id = "2", start_time = 0.0)

    ids_fw = ["vs_mainline_3", "vs_mainline_4", "v_strand_1fw", "h_w_2", "festning_3fw", "festning_4fw", "tollbod_2fw",
            "elvegata_fw", "lundsbro_fw"]
    stops_fw = [("h_w_2", 130), ("festning_4fw", 40), ("tollbod_2fw", 25),
                ("tollbod_2fw", 260)]
    stops_fw = [("h_w_2", 130), ("festning_4fw", 35), ("tollbod_2fw", 30),
                ("tollbod_2fw", 260)]
    # times_fw = [30, 110, 130, 230]
    times_fw = [3, 110, 130, 230]
    bus_fw = bus.Bus(ids_fw, stops_fw, times_fw, temp_network, id = "1")

    times_bw_2 = [240, 330, 390, 450]
    times_fw_2 = [530, 610, 630, 830]

    bus_bw_2 = bus.Bus(ids_bw, stops_bw, times_bw_2, temp_network, id = "3", start_time = 200.0)
    bus_fw_2 = bus.Bus(ids_fw, stops_fw, times_fw_2, temp_network, id = "4", start_time = 500.0)

    bus_network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts,
                                 busses = [bus_fw, bus_bw, bus_fw_2, bus_bw_2], store_densities=True)
    return bus_network

def generate_kvadraturen_roundabout_w_params(T, N, speed_limits, control_points, cycle_times, track_grad = True):
    # Create the roads
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_minimal_junctions_for_roundabout(N, speed_limits[0], control_points[0],
                                                                             speed_limits[1], control_points[1],
                                                                             speed_limits[2], control_points[2],
                                                                             speed_limits[3], control_points[3],
                                                                             speed_limits[4], control_points[4],
                                                                             speed_limits[5], control_points[5],
                                                                             speed_limits[6], control_points[6],
                                                                             track_grad = track_grad)

    # Create the junctions
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_minimal_junctions_w_params(v_strand_fw, v_strand_bw, h_w,
                                                                                                            tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                            elvegata_bw, dronning_fw, dronning_bw,
                                                                                                            festning_fw, festning_bw, lundsbro_fw,
                                                                                                            lundsbro_bw, 
                                                                                                            [cycle_times[0], cycle_times[1], cycle_times[2], cycle_times[3], 
                                                                                                            cycle_times[4], cycle_times[5]], cycle_times[6], cycle_times[7],
                                                                                                            [cycle_times[8], cycle_times[9], cycle_times[10]],
                                                                                                            [cycle_times[11], cycle_times[12]],
                                                                                                            track_grad=track_grad)

    # Create the roundabouts
    mainline_roads, roundabouts = create_roundabouts(v_strand_fw[0], v_strand_bw[0],
                                                     festning_fw[0], festning_bw[0],
                                                     speed_limits[7], speed_limits[8], 
                                                     control_points[6], control_points[7], N,
                                                     track_grad=track_grad)

    # Create the network
    roads = v_strand_fw + v_strand_bw + h_w + tollbod_fw[1:] + tollbod_bw + elvegata_fw + \
            elvegata_bw + dronning_fw + dronning_bw + festning_fw + festning_bw + lundsbro_fw + \
            lundsbro_bw + mainline_roads
    junctions = v_strand_jncs + h_w_jncs + tollbod_jncs + dronning_jncs + festning_jncs

    temp_network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts)


    # Adding the busses
    ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_2bw", "tollbod_1bw", "v_strand_5bw",
          "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw","vs_mainline_1", "vs_mainline_2"]
    stops_bw = [("tollbod_2bw", 50), ("tollbod_1bw", 90), ("tollbod_1bw", 230), ("v_strand_1bw", 25)]
    stops_bw = [("tollbod_2bw", 45), ("tollbod_1bw", 80), ("tollbod_1bw", 235), ("v_strand_1bw", 25)]



    # times_bw = [40, 130, 190, 250]
    times_bw = [4, 130, 190, 250]
    bus_bw = bus.Bus(ids_bw, stops_bw, times_bw, temp_network, id = "2", start_time = 0.0)

    ids_fw = ["vs_mainline_3", "vs_mainline_4", "v_strand_1fw", "h_w_2", "festning_3fw", "festning_4fw", "tollbod_2fw",
            "elvegata_fw", "lundsbro_fw"]
    stops_fw = [("h_w_2", 130), ("festning_4fw", 40), ("tollbod_2fw", 25),
                ("tollbod_2fw", 260)]
    stops_fw = [("h_w_2", 130), ("festning_4fw", 35), ("tollbod_2fw", 30),
                ("tollbod_2fw", 260)]
    # times_fw = [30, 110, 130, 230]
    times_fw = [3, 110, 130, 230]
    bus_fw = bus.Bus(ids_fw, stops_fw, times_fw, temp_network, id = "1")

    times_bw_2 = [240, 330, 390]#, 450]
    stops_bw_2 = [("tollbod_2bw", 50), ("tollbod_1bw", 90), ("tollbod_1bw", 230)]#, ("v_strand_1bw", 25)]
    stops_bw_2 = [("tollbod_2bw", 45), ("tollbod_1bw", 80), ("tollbod_1bw", 235)]#, ("v_strand_1bw", 25)]


    # times_fw_2 = [530, 610, 630, 830]

    bus_bw_2 = bus.Bus(ids_bw, stops_bw_2, times_bw_2, temp_network, id = "3", start_time = 200.0)
    # bus_fw_2 = bus.Bus(ids_fw, stops_fw, times_fw_2, temp_network, id = "4", start_time = 500.0)

    # bus_network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts,
    #                              busses = [bus_fw, bus_bw, bus_fw_2, bus_bw_2], store_densities=True)
    bus_network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts,
                                 busses = [bus_fw, bus_bw, bus_bw_2], store_densities=True)


    return bus_network

def generate_kvadraturen_from_config(T, N, speed_limits, control_points, cycle_times, config = None, track_grad = True):
    if config is None:
        return generate_kvadraturen_roundabout_w_params(T, N, speed_limits, control_points, cycle_times, track_grad=track_grad)
    else:
        # Check that config is on the correct format, then create a bus network using the config
        # Check that all needed keys exist
        assert "init_densities" in config
        assert "boundary_fncs" in config
        assert "fwd_schedules" in config
        assert "bwd_schedules" in config
        assert "roundabout_inflows" in config

        # Check that the number of initial densities, boundary_fncs and roundabout inflow fncs are correct
        assert len(config["init_densities"]) == 9
        assert len(config["boundary_fncs"]) == 3
        assert len(config["roundabout_inflows"]["speeds"]) == 5
        assert len(config["roundabout_inflows"]["inflows"]) == 5


        # Initialize roads from configuration:
        v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
        elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
        lundsbro_fw, lundsbro_bw = create_roads_minimal_junctions_for_roundabout(N, speed_limits[0], control_points[0],
                                                                             speed_limits[1], control_points[1],
                                                                             speed_limits[2], control_points[2],
                                                                             speed_limits[3], control_points[3],
                                                                             speed_limits[4], control_points[4],
                                                                             speed_limits[5], control_points[5],
                                                                             speed_limits[6], control_points[6],
                                                                             init_densities=config["init_densities"],
                                                                             boundary_configs=config["boundary_fncs"],
                                                                             track_grad = track_grad)
        # Initialize junctions
        v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_minimal_junctions_w_params(v_strand_fw, v_strand_bw, h_w,
                                                                                                            tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                            elvegata_bw, dronning_fw, dronning_bw,
                                                                                                            festning_fw, festning_bw, lundsbro_fw,
                                                                                                            lundsbro_bw, 
                                                                                                            [cycle_times[0], cycle_times[1], cycle_times[2], cycle_times[3], 
                                                                                                            cycle_times[4], cycle_times[5]], cycle_times[6], cycle_times[7],
                                                                                                            [cycle_times[8], cycle_times[9], cycle_times[10]],
                                                                                                            [cycle_times[11], cycle_times[12]],
                                                                                                            track_grad=track_grad)

        # Initialize roundabouts from configuration
        mainline_roads, roundabouts = create_roundabouts(v_strand_fw[0], v_strand_bw[0],
                                                     festning_fw[0], festning_bw[0],
                                                     speed_limits[7], speed_limits[8], 
                                                     control_points[6], control_points[7], N,
                                                     initial_densities = config["init_densities"],
                                                     inflow_configs = config["roundabout_inflows"],
                                                     track_grad=track_grad)
        
        # Create the network
        roads = v_strand_fw + v_strand_bw + h_w + tollbod_fw[1:] + tollbod_bw + elvegata_fw + \
                elvegata_bw + dronning_fw + dronning_bw + festning_fw + festning_bw + lundsbro_fw + \
                lundsbro_bw + mainline_roads
        junctions = v_strand_jncs + h_w_jncs + tollbod_jncs + dronning_jncs + festning_jncs

        temp_network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts)

        # Initialize busses from configuration
        busses = create_busses(config["fwd_schedules"], config["bwd_schedules"], temp_network)

        bus_network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts, 
                                     busses = busses, store_densities=True)
        return bus_network
        

if __name__ == "__main__":
    import json
    f = open("kvadraturen_networks/network_1.json")
    data = json.load(f)
    f.close()
    T = data["T"]
    # T = 30
    N = data["N"]
    speed_limits = data["speed_limits"] # Nested list
    control_points = data["control_points"] # Nested list
    cycle_times = data["cycle_times"] # Nested list

    f = open("kvadraturen_networks/config_1_1.json")
    config = json.load(f)
    f.close()

    print(T)
    bus_network = generate_kvadraturen_from_config(T, N, speed_limits, control_points, cycle_times, config)
    bus_network.solve_cons_law()