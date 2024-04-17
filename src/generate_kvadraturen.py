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
def create_roads_minimal_junctions():
    '''
    Combine roads that do not need a junction between them to form a single road

    Idea to reduce memory usage - let the speed limit on each of the roads be the same tensor
    At the very least reduces the number of variables, but potentially also the memory cost(?)
    '''
    L = 50 # Length of road
    N = 5 # Number of nodes to be used for every 50 meters
    offset = 0.1
    tilt = 0.025
    # Vestre Strandgate:
    # junctions 1 and 2 not needed, all other junctions needed
    v_strand_fw = [None] * 6
    v_strand_bw = [None] * 6
    for i in range(6):
        if i == 0:
            # Combine first three to form one road
            # b = 3 since three roads are combined
            v_strand_fw[0] = rd.Road(3, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, 0), right_pos=(0, 3-offset),
                                inflow = 0.3, id="v_strand_" + str(1) + "fw")
            v_strand_bw[0] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, 3-offset), right_pos=(0, 0),
                                inflow = 0.1, id="v_strand_" + str(1) + "bw")
        else:
            v_strand_fw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, i+2+offset), right_pos=(0, i+3-offset),
                                    inflow = 0.05, id="v_strand_" + str(i+1) + "fw")
            v_strand_bw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, i+3-offset), right_pos=(0, i+2+offset),
                                    inflow = 0.1, id="v_strand_" + str(i+1) + "bw")

    # Henrik Wergeland:
    # junction 1 needed, junctions 2 and 3 not needed, junction 4 needed
    h_w = [None] * 2
    h_w[0] = rd.Road(1, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(-0.5, 3), right_pos=(0-offset, 3),
                    inflow = 0.1, id="h_w_" + str(1))
    h_w[1] = rd.Road(2*3, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(0+offset, 3), right_pos=(3-offset+6*tilt, 3),
                    inflow = 0.1, id="h_w_" + str(2))


    # Tollbodgata:
    # Incoming junction needed, junctions 2 and 3 not needed, last junction needed
    tollbod_fw = [None] * 2
    tollbod_bw = [None] * 2
    tollbod_bw[0] = rd.Road(2*3, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(3-offset+2*tilt, 7), right_pos=(0+offset, 7),
                            inflow = 0.1, id="tollbod_" + str(1) + "bw")
    tollbod_bw[1] = rd.Road(2*3, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(6-offset, 7), right_pos=(3+offset+2*tilt, 7),
                            inflow = 0.1, id="tollbod_" + str(2) + "bw")
    tollbod_fw[1] = rd.Road(2*3, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(3+offset+2*tilt, 7), right_pos=(6-offset, 7),
                            inflow = 0.1, id="tollbod_" + str(2) + "fw")

    # Elvegata:
    # Both junctions needed (strictly speaking elvegate into tollbodgata is not needed, but keep¨
    # since the road changes direction/name)
    elvegata_fw = [rd.Road(1, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(6, 7+offset), right_pos=(6, 8-offset),
                        inflow = 0.1, id="elvegata_fw")]
    elvegata_bw = [rd.Road(1, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(6, 8-offset), right_pos=(6, 7+offset),
                        inflow = 0.1, id="elvegata_bw")]

    # Dronningens gate:
    # Incoming already accounted for, junction 2 not needed, junctions 3 and 4 needed,
    # junction 5 and 6 not needed, last junction needed
    dronning_fw = [None] * 3
    dronning_bw = [None] * 3
    dronning_fw[0] = rd.Road(2*2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(0+offset, 8), right_pos=(2-offset, 8),
                            inflow = 0.1, id="dronning_" + str(1) + "fw")
    dronning_bw[0] = rd.Road(2*2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(2-offset, 8), right_pos=(0+offset, 8),
                            inflow = 0.1, id="dronning_" + str(1) + "bw")
    dronning_fw[1] = rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(2+offset, 8), right_pos=(3-offset, 8),
                            inflow = 0.1, id="dronning_" + str(2) + "fw")
    dronning_bw[1] = rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3-offset, 8), right_pos=(2+offset, 8),
                            inflow = 0.1, id="dronning_" + str(2) + "bw")
    dronning_fw[2] = rd.Road(2*3, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+offset, 8), right_pos=(6-offset, 8),
                            inflow = 0.1, id="dronning_" + str(3) + "fw")
    dronning_bw[2] = rd.Road(2*3, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(6-offset, 8), right_pos=(3+offset, 8),
                            inflow = 0.1, id="dronning_" + str(3) + "bw")


    # Festningsgata:
    # junction 1 needed, junction 2 not needed, junctions 3 and 4 needed, 5 and 6 not needed, 7 and 8 needed
    # In picture, festningsgate is not completly vertical. The top is leaning to the right slightly
    # Try 0.05 shift for every piece of road

    festning_fw = [None] * 6
    festning_bw = [None] * 6
    # Add (9-i)*tilt to the roads
    for i in range(6):
        if i == 0:
            festning_fw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(9-i)*tilt, i+offset),
                                     right_pos=(3+(8-i)*tilt, i+1-offset), inflow = 0.2, id="festning_" + str(i+1) + "fw")
            festning_bw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(8-i)*tilt, i+1-offset),
                                     right_pos=(3+(9-i)*tilt, i+offset),inflow = 0.1, id="festning_" + str(i+1) + "bw")
        elif i == 1:
            festning_fw[i] = rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(9-i)*tilt, i+offset),
                                     right_pos=(3+(7-i)*tilt, i+2-offset),inflow = 0.05, id="festning_" + str(i+1) + "fw")
            festning_bw[i] = rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(7-i)*tilt, i+2-offset),
                                     right_pos=(3+(9-i)*tilt, i+offset), inflow = 0.1, id="festning_" + str(i+1) + "bw")
        elif i == 2:
            festning_fw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(8-i)*tilt, i+1+offset),
                                     right_pos=(3+(7-i)*tilt, i+2-offset), inflow = 0.05, id="festning_" + str(i+1) + "fw")
            festning_bw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(7-i)*tilt, i+2-offset),
                                     right_pos=(3+(8-i)*tilt, i+1+offset), inflow = 0.1, id="festning_" + str(i+1) + "bw")
        elif i == 3:
            festning_fw[i] = rd.Road(3, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(8-i)*tilt, i+1+offset),
                                     right_pos=(3+(5-i)*tilt, i+4-offset), inflow = 0.05, id="festning_" + str(i+1) + "fw")
            festning_bw[i] = rd.Road(3, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(5-i)*tilt, i+4-offset),
                                     right_pos=(3+(8-i)*tilt, i+1+offset), inflow = 0.1, id="festning_" + str(i+1) + "bw")
        else:
            # i = 4
            # i = 5
            festning_fw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(6-i)*tilt, i+3+offset),
                                     right_pos=(3+(5-i)*tilt, i+4-offset), inflow = 0.05, id="festning_" + str(i+1) + "fw")
            festning_bw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3+(5-i)*tilt, i+4-offset),
                                     right_pos=(3+(6-i)*tilt, i+3+offset), inflow = 0.1, id="festning_" + str(i+1) + "bw")
    # Lundsbroa:
    # Only one road
    lundsbro_fw = [rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(6+offset, 8), right_pos=(7, 7.8),
                inflow = 0.1, id="lundsbro_fw")]
    lundsbro_bw = [rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(7, 7.8), right_pos=(6+offset, 8),
                inflow = 0.05, id="lundsbro_bw")]

    return v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, lundsbro_fw, lundsbro_bw

def create_roads_minimal_junctions_for_roundabout(N = 2, v_strand_speeds = [50.0], v_strand_controls = [],
                                                  h_w_speeds = [30.0], h_w_controls = [],
                                                  tollbod_speeds = [30.0], tollbod_controls = [],
                                                  elvegate_speeds = [30.0], elvegate_controls = [],
                                                  dronning_speeds = [50.0], dronning_controls = [],
                                                  festning_speeds = [50.0], festning_controls = [],
                                                  lundsbro_speeds = [50.0], lundsbro_controls = []):

    '''
    Combine roads that do not need a junction between them to form a single road

    Idea to reduce memory usage - let the speed limit on each of the roads be the same tensor
    At the very least reduces the number of variables, but potentially also the memory cost(?)

    Change function to take in parameters for the speed limits as well as the control points.

    Also test out defining boundary conditions for the inflow
    For now only do piecewise constant function
    '''
    initial_fnc = lambda x : torch.ones_like(x) * 0.2
    L = 25 # Length of road
    # N: Number of cells to be used for every 25 meters
    offset = 0.1
    tilt = 0.025
    # Vestre Strandgate:
    # junctions 1 and 2 not needed, all other junctions needed
    # No inflow conditions needed
    v_strand_fw = [None] * 6
    v_strand_bw = [None] * 6
    # v_strand_speed_limit = torch.tensor(v_strand_speeds, requires_grad=True)
    v_strand_speed_limit = [torch.tensor(v / 3.6, requires_grad=True) for v in v_strand_speeds]
    v_strand_control_points = v_strand_controls
    for i in range(6):
        if i == 0:
            # Combine first three to form one road
            # b = 3 since three roads are combined
            v_strand_fw[0] = rd.Road(6, L, N, v_strand_speed_limit, v_strand_control_points, left_pos=(0, 0), right_pos=(0, 3-offset),
                                     id="v_strand_" + str(1) + "fw", initial=initial_fnc, boundary_fnc=None, max_dens=2)
            v_strand_bw[0] = rd.Road(2, L, N, v_strand_speed_limit, v_strand_control_points, left_pos=(0, 3-offset), right_pos=(0, 0),
                                     id="v_strand_" + str(1) + "bw",initial=initial_fnc, boundary_fnc=None, max_dens=2)
        else:
            v_strand_fw[i] = rd.Road(2, L, N, v_strand_speed_limit, v_strand_control_points, left_pos=(0, i+2+offset), right_pos=(0, i+3-offset),
                                     id="v_strand_" + str(i+1) + "fw",initial=initial_fnc, boundary_fnc=None, max_dens=2)
            v_strand_bw[i] = rd.Road(2, L, N, v_strand_speed_limit, v_strand_control_points, left_pos=(0, i+3-offset), right_pos=(0, i+2+offset),
                                     id="v_strand_" + str(i+1) + "bw",initial=initial_fnc, boundary_fnc=None, max_dens=2)

    # Henrik Wergeland:
    # junction 1 needed, junctions 2 and 3 not needed, junction 4 needed
    # Boundary inflow condition needed for first road
    h_w = [None] * 2
    # h_w_speed = torch.tensor(h_w_speeds, requires_grad=True)
    h_w_speed = [torch.tensor(v / 3.6, requires_grad=True) for v in h_w_speeds]
    h_w_control_points = h_w_controls
    # Piecewise constant boundary condition:
    boundary_fnc = ibc.boundary_conditions(1, max_dens = 1, densities = torch.tensor([0.3, 0.1, 0.3]),
                                           time_jumps = [300, 600], in_speed = torch.tensor(30.0),
                                           L = L)
    h_w[0] = rd.Road(2, L, N, h_w_speed, h_w_control_points, left_pos=(-0.5, 3), right_pos=(0-offset, 3),
                     id="h_w_" + str(1),initial=initial_fnc, boundary_fnc=boundary_fnc)
    h_w[1] = rd.Road(2*6, L, N, h_w_speed, h_w_control_points, left_pos=(0+offset, 3), right_pos=(3-offset+6*tilt, 3),
                     id="h_w_" + str(2),initial=initial_fnc, boundary_fnc=None)


    # Tollbodgata:
    # Incoming junction needed, junctions 2 and 3 not needed, last junction needed
    # No boundary function needed
    tollbod_fw = [None] * 2
    tollbod_bw = [None] * 2
    # tollbod_speed = torch.tensor(tollbod_speeds, requires_grad=True)
    tollbod_speed = [torch.tensor(v / 3.6, requires_grad=True) for v in tollbod_speeds]
    tollbod_control_points = tollbod_controls
    tollbod_bw[0] = rd.Road(2*6, L, N, tollbod_speed, tollbod_control_points, left_pos=(3-offset+2*tilt, 7), right_pos=(0+offset, 7),
                            id="tollbod_" + str(1) + "bw", initial=initial_fnc, boundary_fnc=None)
    tollbod_bw[1] = rd.Road(2*6, L, N, tollbod_speed, tollbod_control_points, left_pos=(6-offset, 7), right_pos=(3+offset+2*tilt, 7),
                            id="tollbod_" + str(2) + "bw", initial=initial_fnc, boundary_fnc=None)
    tollbod_fw[1] = rd.Road(2*6, L, N, tollbod_speed, tollbod_control_points, left_pos=(3+offset+2*tilt, 7), right_pos=(6-offset, 7),
                            id="tollbod_" + str(2) + "fw", initial=initial_fnc, boundary_fnc=None)

    # Elvegata:
    # Both junctions needed (strictly speaking elvegate into tollbodgata is not needed, but keep¨
    # since the road changes direction/name)
    # No boundary function needed
    # elvegate_speed = torch.tensor(elvegate_speeds, requires_grad=True)
    elvegate_speed = [torch.tensor(v / 3.6, requires_grad=True) for v in elvegate_speeds]
    elvegate_control_points = elvegate_controls
    elvegata_fw = [rd.Road(2, L, N, elvegate_speed, elvegate_control_points, left_pos=(6, 7+offset), right_pos=(6, 8-offset),
                           id="elvegata_fw", initial=initial_fnc, boundary_fnc=None)]
    elvegata_bw = [rd.Road(2, L, N, elvegate_speed, elvegate_control_points, left_pos=(6, 8-offset), right_pos=(6, 7+offset),
                           id="elvegata_bw", initial=initial_fnc, boundary_fnc=None)]

    # Dronningens gate:
    # Incoming already accounted for, junction 2 not needed, junctions 3 and 4 needed,
    # junction 5 and 6 not needed, last junction needed
    # No boundary function needed
    dronning_fw = [None] * 3
    dronning_bw = [None] * 3
    # dronning_speed = torch.tensor(dronning_speeds, requires_grad=True)
    dronning_speed = [torch.tensor(v / 3.6, requires_grad=True) for v in dronning_speeds]
    dronning_control_points = dronning_controls
    dronning_fw[0] = rd.Road(2*4, L, N, dronning_speed, dronning_control_points, left_pos=(0+offset, 8), right_pos=(2-offset, 8),
                             id="dronning_" + str(1) + "fw", initial=initial_fnc, boundary_fnc=None, max_dens=1)
    dronning_bw[0] = rd.Road(2*4, L, N, dronning_speed, dronning_control_points, left_pos=(2-offset, 8), right_pos=(0+offset, 8),
                             id="dronning_" + str(1) + "bw", initial=initial_fnc, boundary_fnc=None, max_dens=1)
    dronning_fw[1] = rd.Road(4, L, N, dronning_speed, dronning_control_points, left_pos=(2+offset, 8), right_pos=(3-offset, 8),
                             id="dronning_" + str(2) + "fw", initial=initial_fnc, boundary_fnc=None, max_dens=1)
    dronning_bw[1] = rd.Road(4, L, N, dronning_speed, dronning_control_points, left_pos=(3-offset, 8), right_pos=(2+offset, 8),
                             id="dronning_" + str(2) + "bw", initial=initial_fnc, boundary_fnc=None, max_dens=1)
    dronning_fw[2] = rd.Road(2*6, L, N, dronning_speed, dronning_control_points, left_pos=(3+offset, 8), right_pos=(6-offset, 8),
                             id="dronning_" + str(3) + "fw", initial=initial_fnc, boundary_fnc=None, max_dens=1)
    dronning_bw[2] = rd.Road(2*6, L, N, dronning_speed, dronning_control_points, left_pos=(6-offset, 8), right_pos=(3+offset, 8),
                             id="dronning_" + str(3) + "bw", initial=initial_fnc, boundary_fnc=None, max_dens=1)


    # Festningsgata:
    # junction 1 needed, junction 2 not needed, junctions 3 and 4 needed, 5 and 6 not needed, 7 and 8 needed
    # In picture, festningsgate is not completly vertical. The top is leaning to the right slightly
    # Try 0.05 shift for every piece of road
    # Boundary function needed on last backwards road

    festning_fw = [None] * 6
    festning_bw = [None] * 6
    # festning_speed = torch.tensor(festning_speeds, requires_grad=True)
    festning_speed = [torch.tensor(v / 3.6, requires_grad=True) for v in festning_speeds]
    festning_control_points = festning_controls
    boundary_fnc = ibc.boundary_conditions(1, max_dens = 2, densities = torch.tensor([0.05, 0.6, 0.1]),
                                           time_jumps = [100, 400], in_speed = torch.tensor(50.0),
                                           L = L)
    # Add (9-i)*tilt to the roads
    for i in range(6):
        if i == 0:
            festning_fw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(9-i)*tilt, i+offset),
                                     right_pos=(3+(8-i)*tilt, i+1-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fnc, boundary_fnc=None,max_dens=2)
            festning_bw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(8-i)*tilt, i+1-offset),
                                     right_pos=(3+(9-i)*tilt, i+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fnc, boundary_fnc=None,max_dens=2)
        elif i == 1:
            festning_fw[i] = rd.Road(4, L, N, festning_speed, festning_control_points, left_pos=(3+(9-i)*tilt, i+offset),
                                     right_pos=(3+(7-i)*tilt, i+2-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fnc, boundary_fnc=None,max_dens=2)
            festning_bw[i] = rd.Road(4, L, N, festning_speed, festning_control_points, left_pos=(3+(7-i)*tilt, i+2-offset),
                                     right_pos=(3+(9-i)*tilt, i+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fnc, boundary_fnc=None,max_dens=2)
        elif i == 2:
            festning_fw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(8-i)*tilt, i+1+offset),
                                     right_pos=(3+(7-i)*tilt, i+2-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fnc, boundary_fnc=None,max_dens=2)
            festning_bw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(7-i)*tilt, i+2-offset),
                                     right_pos=(3+(8-i)*tilt, i+1+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fnc, boundary_fnc=None,max_dens=2)
        elif i == 3:
            festning_fw[i] = rd.Road(6, L, N, festning_speed, festning_control_points, left_pos=(3+(8-i)*tilt, i+1+offset),
                                     right_pos=(3+(5-i)*tilt, i+4-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fnc, boundary_fnc=None,max_dens=2)
            festning_bw[i] = rd.Road(6, L, N, festning_speed, festning_control_points, left_pos=(3+(5-i)*tilt, i+4-offset),
                                     right_pos=(3+(8-i)*tilt, i+1+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fnc, boundary_fnc=None,max_dens=2)
        elif i == 4:
            # i = 4
            # i = 5
            festning_fw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(6-i)*tilt, i+3+offset),
                                     right_pos=(3+(5-i)*tilt, i+4-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fnc,boundary_fnc=None, max_dens=2)
            festning_bw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(5-i)*tilt, i+4-offset),
                                     right_pos=(3+(6-i)*tilt, i+3+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fnc,boundary_fnc=None, max_dens=2)
        else:
            festning_fw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(6-i)*tilt, i+3+offset),
                                     right_pos=(3+(5-i)*tilt, i+4-offset), id="festning_" + str(i+1) + "fw",
                                     initial=initial_fnc,boundary_fnc=None, max_dens=2)
            festning_bw[i] = rd.Road(2, L, N, festning_speed, festning_control_points, left_pos=(3+(5-i)*tilt, i+4-offset),
                                     right_pos=(3+(6-i)*tilt, i+3+offset), id="festning_" + str(i+1) + "bw",
                                     initial=initial_fnc, boundary_fnc=boundary_fnc, max_dens=2)

    # Lundsbroa:
    # Only one road
    # Boundary condition needed on backward road
    # lundsbro_speed = torch.tensor(lundsbro_speeds, requires_grad=True)
    lundsbro_speed = [torch.tensor(v / 3.6, requires_grad=True) for v in lundsbro_speeds]
    lundsbro_control_points = lundsbro_controls
    boundary_fnc = ibc.boundary_conditions(1, max_dens = 2, densities = torch.tensor([0.15, 0.2, 0.15]),
                                           time_jumps = [200, 800], in_speed = torch.tensor(50.0),
                                           L = L)
    lundsbro_fw = [rd.Road(4, L, N, lundsbro_speed, lundsbro_control_points, left_pos=(6+offset, 8), right_pos=(7, 7.8),
                           id="lundsbro_fw", initial=initial_fnc, boundary_fnc=None)]
    lundsbro_bw = [rd.Road(4, L, N, lundsbro_speed, lundsbro_control_points, left_pos=(7, 7.8), right_pos=(6+offset, 8),
                           id="lundsbro_bw", initial=initial_fnc, boundary_fnc=boundary_fnc, max_dens=2)]

    return v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, lundsbro_fw, lundsbro_bw

def create_roundabouts(v_strand_fw, v_strand_bw, festning_fw, festning_bw,
                       speed1 = [50.0], speed2 = [50.0], N = 2):
    # TODO: Modify to include inflow conditions
    # For now, choose these manually

    offset = 0.1
    tilt = 0.025

    L = 25 # Length of road
    #N: Number of nodes to be used for every 50 meters

    # Vestre Strandgate roundabout:
    # Four arm roundabout with 4 junctions, where 2 roads are a part of simulation
    # Secondary roads can be described by a small object with a queue and an inflow condition
    # Creating the mainline
    # main_speed_limit = torch.tensor(speed1, requires_grad=True)
    main_speed_limit = [torch.tensor(v / 3.6, requires_grad=True) for v in speed1]

    vs_main_1 = rd.Road(1, L, N, main_speed_limit, [], left_pos=(0.05,0), right_pos=(0.25, -0.25),
                        id="vs_mainline_1")
    vs_main_2 = rd.Road(1, L, N, main_speed_limit, [], left_pos=(0.25, -0.25), right_pos=(0.08, -0.55),
                        id="vs_mainline_2")
    vs_main_3 = rd.Road(1, L, N, main_speed_limit, [], left_pos=(0.05, -0.55), right_pos=(-0.15, -0.3),
                        id="vs_mainline_3")
    vs_main_4 = rd.Road(1, L, N, main_speed_limit, [], left_pos=(-0.13, -0.28), right_pos=(-0.02,0),
                        id="vs_mainline_4")

    # Creating the secondary incoming roads
    # These are not proper road objects, but rather a small classs containing only
    # the queue length and the inflow function
    secondary_gamma = torch.tensor(50 / (25 * 3.6))
    max_inflow = fv.flux(torch.tensor(0.5), secondary_gamma)
    secondary_big_gamma = torch.tensor(60 / (25 * 3.6))
    max_inflow_big = fv.flux(torch.tensor(0.5), secondary_big_gamma)
    rho_1, rho_2, rho_3 = torch.tensor(0.25),torch.tensor(0.05), torch.tensor(0.05)
    inflow_1 = lambda t : fv.flux(rho_1, secondary_gamma)
    inflow_2 = lambda t : fv.flux(rho_2, secondary_gamma)
    inflow_3 = lambda t : fv.flux(rho_3, secondary_big_gamma)
    vs_secondary_1 = rb.RoundaboutRoad(inflow_1, max_inflow)
    vs_secondary_2 = rb.RoundaboutRoad(inflow_2, max_inflow)
    vs_secondary_3 = rb.RoundaboutRoad(inflow_3, max_inflow_big)
    # Creating the roundabout junctions
    vs_jnc_1 = rb.RoundaboutJunction(vs_main_4, vs_main_1, 0.6, v_strand_bw, v_strand_fw, queue_junction = False)
    vs_jnc_2 = rb.RoundaboutJunction(vs_main_1, vs_main_2, 0.6, vs_secondary_1, None, queue_junction = True)
    vs_jnc_3 = rb.RoundaboutJunction(vs_main_2, vs_main_3, 0.6, vs_secondary_2, None, queue_junction = True)
    vs_jnc_4 = rb.RoundaboutJunction(vs_main_3, vs_main_4, 0.6, vs_secondary_3, None, queue_junction = True)
    vs_junctions = [vs_jnc_1, vs_jnc_2, vs_jnc_3, vs_jnc_4]
    vs_roundabout = rb.Roundabout([vs_main_1, vs_main_2, vs_main_3, vs_main_4],
                           [v_strand_fw, vs_secondary_1, vs_secondary_2, vs_secondary_3],
                           [v_strand_bw, None, None, None],vs_junctions)

    # Festningsgate roundabout:
    # secondary_speed_limit = torch.tensor(speed_2, requires_grad=True)
    secondary_speed_limit = [torch.tensor(v / 3.6, requires_grad=True) for v in speed2]
    # Similar to the vestre strandgate roundabout, but with only three incoming roads
    fn_main_1 = rd.Road(1, L, N, secondary_speed_limit, [], left_pos=(3+9*tilt+0.04, offset), right_pos=(3+9*tilt+0.15, offset-0.4),
                        id="fn_mainline_1")
    fn_main_2 = rd.Road(1, L, N, secondary_speed_limit, [], left_pos=(3+9*tilt+0.15, offset-0.5), right_pos=(3+9*tilt-0.15, offset-0.5),
                        id="fn_mainline_2")
    fn_main_3 = rd.Road(1, L, N, secondary_speed_limit, [], left_pos=(3+9*tilt-0.15, offset-0.4), right_pos=(3+9*tilt-0.04, offset),
                        id="fn_mainline_3")
    # Creating the secondary incoming roads
    rho_4 = torch.tensor(0.15)
    inflow_4 = lambda t : fv.flux(rho_4, secondary_gamma)
    fn_secondary_1 = rb.RoundaboutRoad(inflow_4, max_inflow)
    fn_secondary_2 = rb.RoundaboutRoad(inflow_4, max_inflow)
    # Creating the roundabout junctions
    fn_jnc_1 = rb.RoundaboutJunction(fn_main_3, fn_main_1, 0.6, festning_bw, festning_fw, queue_junction = False)
    fn_jnc_2 = rb.RoundaboutJunction(fn_main_1, fn_main_2, 0.6, fn_secondary_1, None, queue_junction = True)
    fn_jnc_3 = rb.RoundaboutJunction(fn_main_2, fn_main_3, 0.6, fn_secondary_2, None, queue_junction = True)
    fn_junctions = [fn_jnc_1, fn_jnc_2, fn_jnc_3]
    fn_roundabout = rb.Roundabout([fn_main_1, fn_main_2, fn_main_3],
                                  [festning_fw, fn_secondary_1, fn_secondary_2],
                                  [festning_bw, None, None], fn_junctions)

    return [vs_main_1, vs_main_2, vs_main_3, vs_main_4,fn_main_1, fn_main_2, fn_main_3], [vs_roundabout, fn_roundabout]

def create_minimal_junctions(v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw,
                           elvegata_fw, elvegata_bw, dronning_fw, dronning_bw,
                           festning_fw, festning_bw, lundsbro_fw, lundsbro_bw):

    # Junctions for vestre strandgate
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
                                                            [50.0, 100.0])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            h_w[0], h_w[1]], [0,3,4], [1,2,5],
                                            distribution, [], [trafficlight], duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)

        elif i < 4:
            # strandgate to strandgate - trafficlight - no r.o.w.
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])

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
                                                            [100.0, 50.0])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            tollbod_bw[0]], [0,3,4], [1,2],
                                            distribution, [], [trafficlight],
                                            duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)
        else:
            # strandgate to dronningens gate + traffic light
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
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
                                                    [100.0, 50.0])


    h_w_jncs[0] = jn.Junction([h_w[1], festning_fw[1], festning_fw[2],
                                festning_bw[1], festning_bw[2]],
                                [0,1,4], [2,3], distribution, [], [trafficlight],
                                duty_to_gw=True, priorities=priorities, crossing_connections=crossings)

    # For Tollbodgata:
    # Junctions where tollbodgata is connected (and not vestre strandgate or henrik Wergeland)
    tollbod_jncs = [None] * 2
    for i in range(2):
        if i == 0:
            # tollbodgata to festningsgata 4/5 + no traffic light - r.o.w. needed
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
                                                            [50.0, 100.0])
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
                                                    [60.0, 60.0])
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
                                                            [100.0, 50.0])

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
                                                            [50.0, 100.0])

            dronning_jncs[i] = jn.Junction([dronning_fw[i], lundsbro_fw[0],
                                            dronning_bw[i], lundsbro_bw[0],
                                            elvegata_fw[0], elvegata_bw[0]],
                                            [0,3,4], [1,2,5], distribution, [], [trafficlight],
                                            duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)
    festning_jncs = [None] * 2
    # festningsgata to festningsgata - trafficlight - no r.o.w
    trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                            [60.0, 60.0])
    festning_jncs[0] = jn.Junction([festning_fw[0], festning_fw[1],
                                    festning_bw[0], festning_bw[1]],
                                    [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])
    # festningsgata 4 to festningsgata 5 - trafficlight - shifted one index- no r.o.w
    # This appear to be the problem junction, but no clear issues...
    trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
    festning_jncs[1] = jn.Junction([festning_fw[2], festning_fw[3],
                                    festning_bw[2], festning_bw[3]],
                                    [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])

    return v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs

def create_minimal_junctions_w_params(v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw,
                                      elvegata_fw, elvegata_bw, dronning_fw, dronning_bw,
                                      festning_fw, festning_bw, lundsbro_fw, lundsbro_bw,
                                      vs_cycle_1 = [50.0, 100.0], vs_cycle_2 = [50.0, 100.0],
                                      vs_cycle_3 = [50.0, 100.0], vs_cycle_4 = [50.0, 100.0],
                                      vs_cycle_5 = [50.0, 100.0], vs_cycle_6 = [60.0, 60.0],
                                      hw_cycle = [100.0, 50.0], tollbod_cycle = [50.0, 100.0],
                                      dronning_cycle_1 = [60.0, 60.0], dronning_cycle_2 = [100.0, 50.0],
                                      dronning_cycle_3 = [50.0, 100.0], festning_cycle_1 = [60.0, 60.0],
                                      festning_cycle_2 = [60.0, 60.0]):

    vs_cycles = [vs_cycle_1, vs_cycle_2, vs_cycle_3, vs_cycle_4, vs_cycle_5, vs_cycle_6]
    dronning_cycles = [dronning_cycle_1, dronning_cycle_2, dronning_cycle_3]
    festning_cycles = [festning_cycle_1, festning_cycle_2]
    # Junctions for vestre strandgate
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

def create_roads_small():
    # Parameters for all roads
    L = 50
    N = 5

    # Vestre strandgate:
    #   all roads are 2-way and equivalent, (although the junctions are not)
    #   lengths are all 50 meters
    v_strand_fw = [None] * 8
    v_strand_bw = [None] * 8
    for i in range(8):
        v_strand_fw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, i), right_pos=(0, i+1),
                                inflow = 0.05, id="v_strand_" + str(i+1) + "fw")
        v_strand_bw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, i+1), right_pos=(0, i),
                                inflow = 0.1, id="v_strand_" + str(i+1) + "bw")

    # Henrik Wergelands:
    #   all roads are 1-way and equivalent, (although the junctions are not)
    #   lengths are all 100 meters
    h_w = [None] * 4
    for i in range(4):
        h_w[i] = rd.Road(2, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(i-1, 3), right_pos=(i, 3),
                                inflow = 0.1, id="h_w_" + str(i+1))

    # Tollbodgata:
    #   roads 1-3 are 1-way and equivalent, roads 4-6 are 2-way and equivalent
    #   lengths are all 100 meters
    tollbod_fw = [None] * 6
    tollbod_bw = [None] * 6
    for i in range(6):
        tollbod_bw[i] = rd.Road(2, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(i+1, 7), right_pos=(i, 7),
                                inflow = 0.1, id="tollbod_" + str(i+1) + "bw")
        if i >= 3:
            tollbod_fw[i] = rd.Road(2, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(i, 7), right_pos=(i+1, 7),
                                inflow = 0.1, id="tollbod_" + str(i+1) + "fw")


    # Elvegata:
    #   only one 2-way road of length 50
    elvegata_fw = [rd.Road(1, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(6, 7), right_pos=(6, 8),
                        inflow = 0.1, id="elvegata_fw")]
    elvegata_bw = [rd.Road(1, L, N, torch.tensor([30.0], requires_grad=True), [], left_pos=(6, 8), right_pos=(6, 7),
                        inflow = 0.1, id="elvegata_bw")]

    # Dronningens gate:
    #   all roads are 2-way and equivalent, (although the junctions are not)
    #   lengths are all 100 meters
    dronning_fw = [None] * 6
    dronning_bw = [None] * 6
    for i in range(6):
        dronning_fw[i] = rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(i, 8), right_pos=(i+1, 8),
                                inflow = 0.1, id="dronning_" + str(i+1) + "fw")
        dronning_bw[i] = rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(i+1, 8), right_pos=(i, 8),
                                inflow = 0.1, id="dronning_" + str(i+1) + "bw")

    # Festningsgata:
    #   all roads are 2-way and equivalent, (although the junctions are not)
    #   lengths are all 50 meters
    festning_fw = [None] * 9
    festning_bw = [None] * 9
    for i in range(9):
        festning_fw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3, i), right_pos=(3, i+1),
                                inflow = 0.05, id="festning_" + str(i+1) + "fw")
        festning_bw[i] = rd.Road(1, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(3, i+1), right_pos=(3, i),
                                inflow = 0.1, id="festning_" + str(i+1) + "bw")

    # Lundsbroa:
    #   only one 2-way road of length 100
    lundsbro_fw = [rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(6, 8), right_pos=(7, 8),
                        inflow = 0.1, id="lundsbro_fw")]
    lundsbro_bw = [rd.Road(2, L, N, torch.tensor([50.0], requires_grad=True), [], left_pos=(7, 8), right_pos=(6, 8),
                        inflow = 0.05, id="lundsbro_bw")]



    return v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, lundsbro_fw, lundsbro_bw

def create_roads_small_w_params(v_strand_speed = 50, h_w_speed = 30, tollbod_speed = 30,
                                elvegata_speed = 30, dronning_speed = 50, festning_speed = 50,
                                lundsbro_speed = 50):
    # Assume speed does not change along roads
    # Parameters for all roads
    L = 50
    N = 5

    # Vestre strandgate:
    #   all roads are 2-way and equivalent, (although the junctions are not)
    #   lengths are all 50 meters
    v_strand_fw = [None] * 8
    v_strand_bw = [None] * 8
    for i in range(8):
        v_strand_fw[i] = rd.Road(1, L, N, torch.tensor([v_strand_speed], requires_grad=True), [], left_pos=(0, i), right_pos=(0, i+1),
                                inflow = 0.05, id="v_strand_" + str(i+1) + "fw")
        v_strand_bw[i] = rd.Road(1, L, N, torch.tensor([v_strand_speed], requires_grad=True), [], left_pos=(0, i+1), right_pos=(0, i),
                                inflow = 0.1, id="v_strand_" + str(i+1) + "bw")

    # Henrik Wergelands:
    #   all roads are 1-way and equivalent, (although the junctions are not)
    #   lengths are all 100 meters
    h_w = [None] * 4
    for i in range(4):
        h_w[i] = rd.Road(2, L, N, torch.tensor([h_w_speed], requires_grad=True), [], left_pos=(i-1, 3), right_pos=(i, 3),
                                inflow = 0.1, id="h_w_" + str(i+1))

    # Tollbodgata:
    #   roads 1-3 are 1-way and equivalent, roads 4-6 are 2-way and equivalent
    #   lengths are all 100 meters
    tollbod_fw = [None] * 6
    tollbod_bw = [None] * 6
    for i in range(6):
        tollbod_bw[i] = rd.Road(2, L, N, torch.tensor([tollbod_speed], requires_grad=True), [], left_pos=(i+1, 7), right_pos=(i, 7),
                                inflow = 0.1, id="tollbod_" + str(i+1) + "bw")
        if i >= 3:
            tollbod_fw[i] = rd.Road(2, L, N, torch.tensor([tollbod_speed], requires_grad=True), [], left_pos=(i, 7), right_pos=(i+1, 7),
                                inflow = 0.1, id="tollbod_" + str(i+1) + "fw")


    # Elvegata:
    #   only one 2-way road of length 50
    elvegata_fw = [rd.Road(1, L, N, torch.tensor([elvegata_speed], requires_grad=True), [], left_pos=(6, 7), right_pos=(6, 8),
                        inflow = 0.1, id="elvegata_fw")]
    elvegata_bw = [rd.Road(1, L, N, torch.tensor([elvegata_speed], requires_grad=True), [], left_pos=(6, 8), right_pos=(6, 7),
                        inflow = 0.1, id="elvegata_bw")]

    # Dronningens gate:
    #   all roads are 2-way and equivalent, (although the junctions are not)
    #   lengths are all 100 meters
    dronning_fw = [None] * 6
    dronning_bw = [None] * 6
    for i in range(6):
        dronning_fw[i] = rd.Road(2, L, N, torch.tensor([dronning_speed], requires_grad=True), [], left_pos=(i, 8), right_pos=(i+1, 8),
                                inflow = 0.1, id="dronning_" + str(i+1) + "fw")
        dronning_bw[i] = rd.Road(2, L, N, torch.tensor([dronning_speed], requires_grad=True), [], left_pos=(i+1, 8), right_pos=(i, 8),
                                inflow = 0.1, id="dronning_" + str(i+1) + "bw")

    # Festningsgata:
    #   all roads are 2-way and equivalent, (although the junctions are not)
    #   lengths are all 50 meters
    festning_fw = [None] * 9
    festning_bw = [None] * 9
    for i in range(9):
        festning_fw[i] = rd.Road(1, L, N, torch.tensor([festning_speed], requires_grad=True), [], left_pos=(3, i), right_pos=(3, i+1),
                                inflow = 0.05, id="festning_" + str(i+1) + "fw")
        festning_bw[i] = rd.Road(1, L, N, torch.tensor([festning_speed], requires_grad=True), [], left_pos=(3, i+1), right_pos=(3, i),
                                inflow = 0.1, id="festning_" + str(i+1) + "bw")

    # Lundsbroa:
    #   only one 2-way road of length 100
    lundsbro_fw = [rd.Road(2, L, N, torch.tensor([lundsbro_speed], requires_grad=True), [], left_pos=(6, 8), right_pos=(7, 8),
                        inflow = 0.1, id="lundsbro_fw")]
    lundsbro_bw = [rd.Road(2, L, N, torch.tensor([lundsbro_speed], requires_grad=True), [], left_pos=(7, 8), right_pos=(6, 8),
                        inflow = 0.05, id="lundsbro_bw")]



    return v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, lundsbro_fw, lundsbro_bw

def create_junctions_small(v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw,
                           elvegata_fw, elvegata_bw, dronning_fw, dronning_bw,
                           festning_fw, festning_bw, lundsbro_fw, lundsbro_bw):
    # Create the junctions
    # Should probably also have an id...

    # For Vestre strandgate:
    # vestre strandgate 1 is in/out road - only connected at one end
    # Junctions where vestre strandgate is connected
    v_strand_jncs = [None] * 8
    for i in range(8):
        if i < 2:
            # strandgate to strandgate - no trafficlights
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

        elif i == 2:
            # strandgate to henrik Wergeland 4 way + traffic light
            distribution = [[0.7, 0.0, 0.3],
                            [0.0, 0.7, 0.3],
                            [0.3, 0.3, 0.4]]

            trafficlight = tl.CoupledTrafficLightContinuous(True, [0,3], [1,2,5], [4], [1,2,5],
                                                            [50.0, 100.0])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            h_w[0], h_w[1]], [0,3,4], [1,2,5],
                                            distribution, [], [trafficlight])

        elif i < 6:
            # strandgate to strandgate - trafficlight
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])

            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])
        elif i==6:
            # strandgate to tollbod 1 way + traffic light
            distribution = [[1.0, 0.0],
                            [0.0, 1.0],
                            [0.3, 0.7]]
            trafficlight = tl.CoupledTrafficLightContinuous(True, [0,3], [1,2], [4], [1,2],
                                                            [100.0, 50.0])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            tollbod_bw[0]], [0,3,4], [1,2],
                                            distribution, [], [trafficlight])
        else:
            # strandgate to dronningens gate + traffic light
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], dronning_fw[0],
                                            v_strand_bw[i], dronning_bw[0]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])

    # For Henrik Wergelands:
    # Junctions where henrik Wergeland is connected (and nt vestre strandgate)
    h_w_jncs = [None] * 3
    for i in range(3):
        if i < 2:
            # henrik Wergeland to henrik Wergeland - no trafficlights
            h_w_jncs[i] = jn.Junction([h_w[i], h_w[i+1]],
                                        [0], [1], [[1.0]], [], [])
        if i == 2:
            #henrik Wergeland to festningsgata 3/4 + traffic light
            distribution = [[0.5, 0.5],
                            [1.0, 0.0],
                            [0.0, 1.0]]
            trafficlight = tl.CoupledTrafficLightContinuous(True, [0], [2,3], [1,4], [2,3],
                                                            [100.0, 50.0])


            h_w_jncs[i] = jn.Junction([h_w[i], festning_fw[2], festning_fw[3],
                                       festning_bw[2], festning_bw[3]],
                                       [0,1,4], [2,3], distribution, [], [trafficlight])

    # For Tollbodgata:
    # Junctions where tollbodgata is connected (and not vestre strandgate or henrik Wergeland)
    tollbod_jncs = [None] * 6
    for i in range(6):
        if i < 2:
            # tollbodgata to tollbodgata - no trafficlights - 1way
            tollbod_jncs[i] = jn.Junction([tollbod_bw[i], tollbod_bw[i+1]],
                                            [1], [0], [[1.0]], [], [])
        elif i == 2:
            # tollbodgata to festningsgata 7/8 + no traffic light
            distribution = [[0.1, 0.8, 0.0, 0.1],
                            [0.1, 0.0, 0.8, 0.1],
                            [0.5, 0.25, 0.25, 0.0]]
            trafficlight = tl.CoupledTrafficLightContinuous(True, [1,4], [0,2,3,5], [6], [0,2,3,5],
                                                            [50.0, 100.0])
            tollbod_jncs[i] = jn.Junction([tollbod_bw[i], festning_fw[6], festning_fw[7],
                                           festning_bw[6], festning_bw[7], tollbod_fw[i+1],
                                           tollbod_bw[i+1]], [1,4,6], [0,2,3,5],
                                           distribution, [], [trafficlight])
        elif i < 5:
            # tollbodgata to tollbodgata - no traffic light - 2way
            tollbod_jncs[i] = jn.Junction([tollbod_fw[i], tollbod_fw[i+1],
                                            tollbod_bw[i], tollbod_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
        else:
            # tollbodgata to elvegata + no traffic light
            tollbod_jncs[i] = jn.Junction([tollbod_fw[i], elvegata_fw[0],
                                            tollbod_bw[i], elvegata_bw[0]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

    # For Dronningens gate:
    # Junctions where dronningens gate is connected (and not vestre strandgate, henrik Wergeland or tollbodgata)
    dronning_jncs = [None] * 6
    for i in range(6):
        if i in [0,3,4]:
            # dronningens gate to dronningens gate - no trafficlights
            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

        elif i == 1:
            # dronningens gate to dronningens gate - trafficlight

            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])

        elif i == 2:
            # dronningens gate to festningsgata 8/9 + traffic light
            distribution = [[0.7, 0.0, 0.15, 0.15],
                            [0.0, 0.7, 0.15, 0.15],
                            [0.15, 0.15, 0.7, 0.0],
                            [0.15, 0.15, 0.0, 0.7]]
            trafficlight = tl.CoupledTrafficLightContinuous(True, [0,3], [1,2,5,6], [4,7], [1,2,5,6],
                                                            [100.0, 50.0])

            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1],
                                            festning_fw[7], festning_fw[8],
                                            festning_bw[7], festning_bw[8]],
                                            [0,3,4,7], [1,2,5,6], distribution, [], [trafficlight])
        elif i == 5:
            # dronningens gate to elvegata and lundsbro + traffic light
            distribution = [[0.7, 0.0, 0.3],
                            [0.0, 0.7, 0.3],
                            [0.5, 0.5, 0.0]]
            trafficlight = tl.CoupledTrafficLightContinuous(True, [0,3], [1,2,5], [4], [1,2,5],
                                                            [50.0, 100.0])

            dronning_jncs[i] = jn.Junction([dronning_fw[i], lundsbro_fw[0],
                                            dronning_bw[i], lundsbro_bw[0],
                                            elvegata_fw[0], elvegata_bw[0]],
                                            [0,3,4], [1,2,5], distribution, [], [trafficlight])


    # For Festningsgata:
    # Junctions where festningsgata is connected (and not vestre strandgate, henrik Wergeland, tollbodgata or dronningens gate)
    festning_jncs = [None] * 5
    for i in range(5):
        if i == 0:
            # festningsgata to festningsgata - trafficlight
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
            festning_jncs[i] = jn.Junction([festning_fw[i], festning_fw[i+1],
                                            festning_bw[i], festning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])
        elif i == 1:
            # festningsgata to festningsgata - no trafficlight
            festning_jncs[i] = jn.Junction([festning_fw[i], festning_fw[i+1],
                                            festning_bw[i], festning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
        elif i == 3:
            # festningsgata to festningsgata - trafficlight - shifted one index
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
            festning_jncs[i] = jn.Junction([festning_fw[i+1], festning_fw[i+2],
                                            festning_bw[i+1], festning_bw[i+2]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])
        elif i in [4, 5]:
            #  festningsgata to festningsgata - no trafficlights - shifted one index
            festning_jncs[i] = jn.Junction([festning_fw[i+1], festning_fw[i+2],
                                            festning_bw[i+1], festning_bw[i+2]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
    festning_jncs.pop(2)
    return v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs

def create_junctions_small_row(v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw,
                           elvegata_fw, elvegata_bw, dronning_fw, dronning_bw,
                           festning_fw, festning_bw, lundsbro_fw, lundsbro_bw):
    '''
    Generates the junctions of the interesting parts of kvadraturen also including right of way
    For every junction, a priority matrix and a matrix containing the crossing connections is then needed.
    '''

    # For Vestre strandgate:
    # vestre strandgate 1 is in/out road - only connected at one end
    # Junctions where vestre strandgate is connected
    v_strand_jncs = [None] * 8
    for i in range(8):
        if i < 2:
            # strandgate to strandgate - no trafficlights
            # Right of way not relevant for this junction
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

        elif i == 2:
            # strandgate to henrik Wergeland 4 way + traffic light
            # In this junction right of way is interesting
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
                                                            [50.0, 100.0])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            h_w[0], h_w[1]], [0,3,4], [1,2,5],
                                            distribution, [], [trafficlight], duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)

        elif i < 6:
            # strandgate to strandgate - trafficlight
            # Right of way not relevant
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])

            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])
        elif i==6:
            # strandgate to tollbod 1 way + traffic light
            # Right of way needed for incoming traffic from tollbod
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
                                                            [100.0, 50.0])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            tollbod_bw[0]], [0,3,4], [1,2],
                                            distribution, [], [trafficlight],
                                            duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)
        else:
            # strandgate to dronningens gate + traffic light
            # Traffic can only go one direction - no r.o.w. needed
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], dronning_fw[0],
                                            v_strand_bw[i], dronning_bw[0]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])

    # For Henrik Wergelands:
    # Junctions where henrik Wergeland is connected (and nt vestre strandgate)
    h_w_jncs = [None] * 3
    for i in range(3):
        if i < 2:
            # henrik Wergeland to henrik Wergeland - no trafficlights and no r.o.w.
            h_w_jncs[i] = jn.Junction([h_w[i], h_w[i+1]],
                                        [0], [1], [[1.0]], [], [])
        if i == 2:
            #henrik Wergeland to festningsgata 3/4 + traffic light
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
                                                            [100.0, 50.0])


            h_w_jncs[i] = jn.Junction([h_w[i], festning_fw[2], festning_fw[3],
                                       festning_bw[2], festning_bw[3]],
                                       [0,1,4], [2,3], distribution, [], [trafficlight],
                                       duty_to_gw=True, priorities=priorities, crossing_connections=crossings)

    # For Tollbodgata:
    # Junctions where tollbodgata is connected (and not vestre strandgate or henrik Wergeland)
    tollbod_jncs = [None] * 6
    for i in range(6):
        if i < 2:
            # tollbodgata to tollbodgata - no trafficlights - 1way - no r.o.w
            tollbod_jncs[i] = jn.Junction([tollbod_bw[i], tollbod_bw[i+1]],
                                            [1], [0], [[1.0]], [], [])
        elif i == 2:
            # tollbodgata to festningsgata 7/8 + no traffic light - r.o.w. needed
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
                                                            [50.0, 100.0])
            tollbod_jncs[i] = jn.Junction([tollbod_bw[i], festning_fw[6], festning_fw[7],
                                           festning_bw[6], festning_bw[7], tollbod_fw[i+1],
                                           tollbod_bw[i+1]], [1,4,6], [0,2,3,5],
                                           distribution, [], [trafficlight], duty_to_gw=True,
                                           priorities=priorities, crossing_connections=crossings)
        elif i < 5:
            # tollbodgata to tollbodgata - no traffic light - 2way - r.o.w. not relevant
            tollbod_jncs[i] = jn.Junction([tollbod_fw[i], tollbod_fw[i+1],
                                            tollbod_bw[i], tollbod_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
        else:
            # tollbodgata to elvegata + no traffic light - r.o.w. not relevant
            tollbod_jncs[i] = jn.Junction([tollbod_fw[i], elvegata_fw[0],
                                            tollbod_bw[i], elvegata_bw[0]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

    # For Dronningens gate:
    # Junctions where dronningens gate is connected (and not vestre strandgate, henrik Wergeland or tollbodgata)
    dronning_jncs = [None] * 6
    for i in range(6):
        if i in [0,3,4]:
            # dronningens gate to dronningens gate - no trafficlights - r.o.w. not relevant
            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

        elif i == 1:
            # dronningens gate to dronningens gate - trafficlight - r.o.w. not relevant

            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])

        elif i == 2:
            # dronningens gate to festningsgata 8/9 + traffic light - r.o.w. needed
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
                                                            [100.0, 50.0])

            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1],
                                            festning_fw[7], festning_fw[8],
                                            festning_bw[7], festning_bw[8]],
                                            [0,3,4,7], [1,2,5,6], distribution, [], [trafficlight],
                                            duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)
        elif i == 5:
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
                                                            [50.0, 100.0])

            dronning_jncs[i] = jn.Junction([dronning_fw[i], lundsbro_fw[0],
                                            dronning_bw[i], lundsbro_bw[0],
                                            elvegata_fw[0], elvegata_bw[0]],
                                            [0,3,4], [1,2,5], distribution, [], [trafficlight],
                                            duty_to_gw=True, priorities=priorities,
                                            crossing_connections=crossings)


    # For Festningsgata:
    # Junctions where festningsgata is connected (and not vestre strandgate, henrik Wergeland, tollbodgata or dronningens gate)
    festning_jncs = [None] * 5
    for i in range(5):
        if i == 0:
            # festningsgata to festningsgata - trafficlight - no r.o.w
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
            festning_jncs[i] = jn.Junction([festning_fw[i], festning_fw[i+1],
                                            festning_bw[i], festning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])
        elif i == 1:
            # festningsgata to festningsgata - no trafficlight - no r.o.w.
            festning_jncs[i] = jn.Junction([festning_fw[i], festning_fw[i+1],
                                            festning_bw[i], festning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
        elif i == 3:
            # festningsgata to festningsgata - trafficlight - shifted one index- no r.o.w
            # Should maybe not be shifted one index(?)
            # This appear to be the problem junction, but no clear issues...
            trafficlight = tl.TrafficLightContinous(True, [0,3], [1,2],
                                                    [60.0, 60.0])
            festning_jncs[i] = jn.Junction([festning_fw[i+1], festning_fw[i+2],
                                            festning_bw[i+1], festning_bw[i+2]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [trafficlight], [])
        elif i in [4, 5]:
            #  festningsgata to festningsgata - no trafficlights - shifted one index - no r.o.w.
            festning_jncs[i] = jn.Junction([festning_fw[i+1], festning_fw[i+2],
                                            festning_bw[i+1], festning_bw[i+2]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
    festning_jncs.pop(2)
    return v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs

def generate_kvadraturen_small(T):
    '''
    In the first iteration, do things manually
    This is rather unfeasible for the large case, but is okay in this smaller network
    '''

    # Create the roads
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_small()

    # Create the junctions
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_junctions_small(v_strand_fw, v_strand_bw, h_w,
                                                                                                 tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                 elvegata_bw, dronning_fw, dronning_bw,
                                                                                                 festning_fw, festning_bw, lundsbro_fw,
                                                                                                 lundsbro_bw)

    # Create the network
    roads = v_strand_fw + v_strand_bw + h_w + tollbod_fw[3:] + tollbod_bw + elvegata_fw + \
            elvegata_bw + dronning_fw + dronning_bw + festning_fw + festning_bw + lundsbro_fw + lundsbro_bw
    junctions = v_strand_jncs + h_w_jncs + tollbod_jncs + dronning_jncs + festning_jncs

    network = nw.RoadNetwork(roads, junctions, T)#, optimizing = False)


    return network

def generate_kvadraturen_minimal_junctions(T):
    # Create the roads
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_minimal_junctions()

    # Create the junctions
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_minimal_junctions(v_strand_fw, v_strand_bw, h_w,
                                                                                                 tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                 elvegata_bw, dronning_fw, dronning_bw,
                                                                                                 festning_fw, festning_bw, lundsbro_fw,
                                                                                                 lundsbro_bw)
    # Create the network
    roads = v_strand_fw + v_strand_bw + h_w + tollbod_fw[1:] + tollbod_bw + elvegata_fw + \
            elvegata_bw + dronning_fw + dronning_bw + festning_fw + festning_bw + lundsbro_fw + lundsbro_bw
    junctions = v_strand_jncs + h_w_jncs + tollbod_jncs + dronning_jncs + festning_jncs

    network = nw.RoadNetwork(roads, junctions, T)#, optimizing = False)

    return network

def generate_kvadraturen_w_roundabout(T):
    # Create the roads
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_minimal_junctions_for_roundabout()

    # Create the junctions
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_minimal_junctions(v_strand_fw, v_strand_bw, h_w,
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

def generate_kvadraturen_roundabout_w_params(T, N, speed_limits, control_points, cycle_times):

    # Create the roads
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_minimal_junctions_for_roundabout(N, speed_limits[0], control_points[0],
                                                                             speed_limits[1], control_points[1],
                                                                             speed_limits[2], control_points[2],
                                                                             speed_limits[3], control_points[3],
                                                                             speed_limits[4], control_points[4],
                                                                             speed_limits[5], control_points[5],
                                                                             speed_limits[6], control_points[6])

    # Create the junctions
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_minimal_junctions_w_params(v_strand_fw, v_strand_bw, h_w,
                                                                                                            tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                            elvegata_bw, dronning_fw, dronning_bw,
                                                                                                            festning_fw, festning_bw, lundsbro_fw,
                                                                                                            lundsbro_bw, cycle_times[0], cycle_times[1],
                                                                                                            cycle_times[2], cycle_times[3], cycle_times[4],
                                                                                                            cycle_times[5], cycle_times[6], cycle_times[7],
                                                                                                            cycle_times[8], cycle_times[9], cycle_times[10],
                                                                                                            cycle_times[11])

    # Create the roundabouts
    mainline_roads, roundabouts = create_roundabouts(v_strand_fw[0], v_strand_bw[0],
                                                     festning_fw[0], festning_bw[0],
                                                     speed_limits[7], speed_limits[8], N)

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
    # times_bw = [40, 130, 190, 250]
    times_bw = [4, 130, 190, 250]
    bus_bw = bus.Bus(ids_bw, stops_bw, times_bw, temp_network, id = "2", start_time = 0.0)

    ids_fw = ["vs_mainline_3", "vs_mainline_4", "v_strand_1fw", "h_w_2", "festning_3fw", "festning_4fw", "tollbod_2fw",
            "elvegata_fw", "lundsbro_fw"]
    stops_fw = [("h_w_2", 130), ("festning_4fw", 40), ("tollbod_2fw", 25),
                ("tollbod_2fw", 260)]
    # times_fw = [30, 110, 130, 230]
    times_fw = [3, 110, 130, 230]
    bus_fw = bus.Bus(ids_fw, stops_fw, times_fw, temp_network, id = "1")

    times_bw_2 = [240, 330, 390]#, 450]
    stops_bw_2 = [("tollbod_2bw", 50), ("tollbod_1bw", 90), ("tollbod_1bw", 230)]#, ("v_strand_1bw", 25)]

    # times_fw_2 = [530, 610, 630, 830]

    bus_bw_2 = bus.Bus(ids_bw, stops_bw_2, times_bw_2, temp_network, id = "3", start_time = 200.0)
    # bus_fw_2 = bus.Bus(ids_fw, stops_fw, times_fw_2, temp_network, id = "4", start_time = 500.0)

    # bus_network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts,
    #                              busses = [bus_fw, bus_bw, bus_fw_2, bus_bw_2], store_densities=True)
    bus_network = nw.RoadNetwork(roads, junctions, T, roundabouts=roundabouts,
                                 busses = [bus_fw, bus_bw, bus_bw_2], store_densities=True)


    return bus_network

def generate_kvadraturen_w_bus(T):
    # TODO: add more arguments to the function to allow the user more flexibility

    # Create the roads - should be possible to pass speed limits as arguments
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_minimal_junctions_for_roundabout()

    # Create the junctions - should be possible to pass cycle times of traffic lights as arguments
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_minimal_junctions(v_strand_fw, v_strand_bw, h_w,
                                                                                                 tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                 elvegata_bw, dronning_fw, dronning_bw,
                                                                                                 festning_fw, festning_bw, lundsbro_fw,
                                                                                                 lundsbro_bw)

    # Create the roundabouts - pass speed limits of mainline roads as argument?
    mainline_roads, roundabouts = create_roundabouts(v_strand_fw[0], v_strand_bw[0],
                                                     festning_fw[0], festning_bw[0])

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
    # times_bw = [40, 130, 190, 250]
    times_bw = [4, 130, 190, 250]
    bus_bw = bus.Bus(ids_bw, stops_bw, times_bw, temp_network, id = "2", start_time = 0.0)

    ids_fw = ["vs_mainline_3", "vs_mainline_4", "v_strand_1fw", "h_w_2", "festning_3fw", "festning_4fw", "tollbod_2fw",
            "elvegata_fw", "lundsbro_fw"]
    stops_fw = [("h_w_2", 130), ("festning_4fw", 40), ("tollbod_2fw", 25),
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

def minimal_kvadraturen(T = 100):
    road1 = rd.Road(1, 50, 5, torch.tensor([50.0], requires_grad=True), [], left_pos=(-1, 0), right_pos=(0, 0),
                    inflow = 0.1, id="road1fw")
    road2 = rd.Road(1, 50, 5, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, 0), right_pos=(1, 0),
                    inflow = 0.1, id="road1bw")
    road3 = rd.Road(1, 50, 5, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, -1), right_pos=(0, 0),
                    inflow = 0.1, id="road2fw")
    road4 = rd.Road(1, 50, 5, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, 0), right_pos=(0, 1),
                    inflow = 0.1, id="road2bw")
    traffic_light = tl.CoupledTrafficLightContinuous(True, [0], [1], [3], [2], [50.0, 50.0])
    junction = jn.Junction([road1, road2, road3, road4], [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [traffic_light])
    network = nw.RoadNetwork([road1, road2, road3, road4], [junction], T)#, optimizing = False)
    return network

def create_junctions_wo_tl(v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw,
                           elvegata_fw, elvegata_bw, dronning_fw, dronning_bw,
                           festning_fw, festning_bw, lundsbro_fw, lundsbro_bw):
    # Create the junctions
    # Should probably also have an id...

    # For Vestre strandgate:
    # vestre strandgate 1 is in/out road - only connected at one end
    # Junctions where vestre strandgate is connected
    v_strand_jncs = [None] * 8
    for i in range(8):
        if i < 2:
            # strandgate to strandgate - no trafficlights
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

        elif i == 2:
            # strandgate to henrik Wergeland 4 way + traffic light
            distribution = [[0.7, 0.0, 0.3],
                            [0.0, 0.7, 0.3],
                            [0.3, 0.3, 0.4]]

            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            h_w[0], h_w[1]], [0,3,4], [1,2,5],
                                            distribution, [], [])

        elif i < 6:
            # strandgate to strandgate - trafficlight

            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
        elif i==6:
            # strandgate to tollbod 1 way + traffic light
            distribution = [[1.0, 0.0],
                            [0.0, 1.0],
                            [0.3, 0.7]]
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], v_strand_fw[i+1],
                                            v_strand_bw[i], v_strand_bw[i+1],
                                            tollbod_bw[0]], [0,3,4], [1,2],
                                            distribution, [], [])
        else:
            # strandgate to dronningens gate + traffic light
            v_strand_jncs[i] = jn.Junction([v_strand_fw[i], dronning_fw[0],
                                            v_strand_bw[i], dronning_bw[0]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

    # For Henrik Wergelands:
    # Junctions where henrik Wergeland is connected (and nt vestre strandgate)
    h_w_jncs = [None] * 3
    for i in range(3):
        if i < 2:
            # henrik Wergeland to henrik Wergeland - no trafficlights
            h_w_jncs[i] = jn.Junction([h_w[i], h_w[i+1]],
                                        [0], [1], [[1.0]], [], [])
        if i == 2:
            #henrik Wergeland to festningsgata 3/4 + traffic light
            distribution = [[0.5, 0.5],
                            [1.0, 0.0],
                            [0.0, 1.0]]


            h_w_jncs[i] = jn.Junction([h_w[i], festning_fw[2], festning_fw[3],
                                       festning_bw[2], festning_bw[3]],
                                       [0,1,4], [2,3], distribution, [], [])

    # For Tollbodgata:
    # Junctions where tollbodgata is connected (and not vestre strandgate or henrik Wergeland)
    tollbod_jncs = [None] * 6
    for i in range(6):
        if i < 2:
            # tollbodgata to tollbodgata - no trafficlights - 1way
            tollbod_jncs[i] = jn.Junction([tollbod_bw[i], tollbod_bw[i+1]],
                                            [1], [0], [[1.0]], [], [])
        elif i == 2:
            # tollbodgata to festningsgata 7/8 + no traffic light
            distribution = [[0.1, 0.8, 0.0, 0.1],
                            [0.1, 0.0, 0.8, 0.1],
                            [0.5, 0.25, 0.25, 0.0]]

            tollbod_jncs[i] = jn.Junction([tollbod_bw[i], festning_fw[6], festning_fw[7],
                                           festning_bw[6], festning_bw[7], tollbod_fw[i+1],
                                           tollbod_bw[i+1]], [1,4,6], [0,2,3,5],
                                           distribution, [], [])
        elif i < 5:
            # tollbodgata to tollbodgata - no traffic light - 2way
            tollbod_jncs[i] = jn.Junction([tollbod_fw[i], tollbod_fw[i+1],
                                            tollbod_bw[i], tollbod_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
        else:
            # tollbodgata to elvegata + no traffic light
            tollbod_jncs[i] = jn.Junction([tollbod_fw[i], elvegata_fw[0],
                                            tollbod_bw[i], elvegata_bw[0]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

    # For Dronningens gate:
    # Junctions where dronningens gate is connected (and not vestre strandgate, henrik Wergeland or tollbodgata)
    dronning_jncs = [None] * 6
    for i in range(6):
        if i in [0,3,4]:
            # dronningens gate to dronningens gate - no trafficlights
            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

        elif i == 1:
            # dronningens gate to dronningens gate - trafficlight

            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])

        elif i == 2:
            # dronningens gate to festningsgata 8/9 + traffic light
            distribution = [[0.7, 0.0, 0.15, 0.15],
                            [0.0, 0.7, 0.15, 0.15],
                            [0.15, 0.15, 0.7, 0.0],
                            [0.15, 0.15, 0.0, 0.7]]

            dronning_jncs[i] = jn.Junction([dronning_fw[i], dronning_fw[i+1],
                                            dronning_bw[i], dronning_bw[i+1],
                                            festning_fw[7], festning_fw[8],
                                            festning_bw[7], festning_bw[8]],
                                            [0,3,4,7], [1,2,5,6], distribution, [], [])
        elif i == 5:
            # dronningens gate to elvegata and lundsbro + traffic light
            distribution = [[0.7, 0.0, 0.3],
                            [0.0, 0.7, 0.3],
                            [0.5, 0.5, 0.0]]

            dronning_jncs[i] = jn.Junction([dronning_fw[i], lundsbro_fw[0],
                                            dronning_bw[i], lundsbro_bw[0],
                                            elvegata_fw[0], elvegata_bw[0]],
                                            [0,3,4], [1,2,5], distribution, [], [])


    # For Festningsgata:
    # Junctions where festningsgata is connected (and not vestre strandgate, henrik Wergeland, tollbodgata or dronningens gate)
    festning_jncs = [None] * 5
    for i in range(5):
        if i == 0:
            # festningsgata to festningsgata - trafficlight
            festning_jncs[i] = jn.Junction([festning_fw[i], festning_fw[i+1],
                                            festning_bw[i], festning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
        elif i == 1:
            # festningsgata to festningsgata - no trafficlight
            festning_jncs[i] = jn.Junction([festning_fw[i], festning_fw[i+1],
                                            festning_bw[i], festning_bw[i+1]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
        elif i == 3:
            # festningsgata to festningsgata - trafficlight - shifted one index

            festning_jncs[i] = jn.Junction([festning_fw[i+1], festning_fw[i+2],
                                            festning_bw[i+1], festning_bw[i+2]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
        elif i in [4, 5]:
            #  festningsgata to festningsgata - no trafficlights - shifted one index
            festning_jncs[i] = jn.Junction([festning_fw[i+1], festning_fw[i+2],
                                            festning_bw[i+1], festning_bw[i+2]],
                                            [0,3], [1,2], [[1.0, 0.0],[0.0, 1.0]], [], [])
    festning_jncs.pop(2)
    return v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs

def generate_kvadraturen_wo_tl(T):
    '''
    In the first iteration, do things manually
    This is rather unfeasible for the large case, but is okay in this smaller network
    '''

    # Create the roads
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_small()

    # Create the junctions
    # Create the junctions
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_junctions_wo_tl(v_strand_fw, v_strand_bw, h_w,
                                                                                                 tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                 elvegata_bw, dronning_fw, dronning_bw,
                                                                                                 festning_fw, festning_bw, lundsbro_fw,
                                                                                                 lundsbro_bw)

    # Create the network
    roads = v_strand_fw + v_strand_bw + h_w + tollbod_fw[3:] + tollbod_bw + elvegata_fw + \
            elvegata_bw + dronning_fw + dronning_bw + festning_fw + festning_bw + lundsbro_fw + lundsbro_bw
    junctions = v_strand_jncs + h_w_jncs + tollbod_jncs + dronning_jncs + festning_jncs

    network = nw.RoadNetwork(roads, junctions, T)#, optimizing = False)

    return network

def generate_minimal(T):
    # Only four roads, one junction
    road1 = rd.Road(1, 50, 5, torch.tensor([50.0], requires_grad=True), [], left_pos=(-1, 0), right_pos=(0, 0),
                    inflow = 0.1, id="road1")
    road2 = rd.Road(1, 50, 5, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, 0), right_pos=(1, 0),
                    inflow = 0.1, id="road2")
    road3 = rd.Road(1, 50, 5, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, -1), right_pos=(0, 0),
                    inflow = 0.1, id="road3")
    road4 = rd.Road(1, 50, 5, torch.tensor([50.0], requires_grad=True), [], left_pos=(0, 0), right_pos=(0, 1),
                    inflow = 0.1, id="road4")

    trafficlight = tl.CoupledTrafficLightContinuous(False, [0], [2], [1], [3], [50.0, 50.0])
    junction = jn.Junction([road1, road2, road3, road4], [0,1], [2,3], [[1.0, 0.0],[0.0, 1.0]], [], [trafficlight])

    network = nw.RoadNetwork([road1, road2, road3, road4], [junction], T)#, optimizing = False)
    return network

# Can also have a network where most of the road speed limits do not require a gradient...

def generate_kvadraturen_small_w_row(T):
    '''
    In the first iteration, do things manually
    This is rather unfeasible for the large case, but is okay in this smaller network
    '''

    # Create the roads
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_small()

    # Create the junctions
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_junctions_small_row(v_strand_fw, v_strand_bw, h_w,
                                                                                                 tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                 elvegata_bw, dronning_fw, dronning_bw,
                                                                                                 festning_fw, festning_bw, lundsbro_fw,
                                                                                                 lundsbro_bw)

    # Create the network
    roads = v_strand_fw + v_strand_bw + h_w + tollbod_fw[3:] + tollbod_bw + elvegata_fw + \
            elvegata_bw + dronning_fw + dronning_bw + festning_fw + festning_bw + lundsbro_fw + lundsbro_bw
    junctions = v_strand_jncs + h_w_jncs + tollbod_jncs + dronning_jncs + festning_jncs

    network = nw.RoadNetwork(roads, junctions, T)#, optimizing = False)

    return network

def generate_kvadraturen_small_w_params(T, v_strand_speed = 50, h_w_speed = 30, tollbod_speed = 30,
                                        elvegata_speed = 30, dronning_speed = 50, festning_speed = 50,
                                        lundsbro_speed = 50):
    '''
    In the first iteration, do things manually
    This is rather unfeasible for the large case, but is okay in this smaller network
    '''

    # Create the roads
    v_strand_fw, v_strand_bw, h_w, tollbod_fw, tollbod_bw, elvegata_fw, \
    elvegata_bw, dronning_fw, dronning_bw, festning_fw, festning_bw, \
    lundsbro_fw, lundsbro_bw = create_roads_small_w_params(v_strand_speed, h_w_speed, tollbod_speed,
                                                        elvegata_speed, dronning_speed, festning_speed,
                                                        lundsbro_speed)

    # Create the junctions
    v_strand_jncs, h_w_jncs, tollbod_jncs, dronning_jncs, festning_jncs = create_junctions_small(v_strand_fw, v_strand_bw, h_w,
                                                                                                 tollbod_fw, tollbod_bw, elvegata_fw,
                                                                                                 elvegata_bw, dronning_fw, dronning_bw,
                                                                                                 festning_fw, festning_bw, lundsbro_fw,
                                                                                                 lundsbro_bw)

    # Create the network
    roads = v_strand_fw + v_strand_bw + h_w + tollbod_fw[3:] + tollbod_bw + elvegata_fw + \
            elvegata_bw + dronning_fw + dronning_bw + festning_fw + festning_bw + lundsbro_fw + lundsbro_bw
    junctions = v_strand_jncs + h_w_jncs + tollbod_jncs + dronning_jncs + festning_jncs

    network = nw.RoadNetwork(roads, junctions, T)#, optimizing = False)

    return network