import torch
import road as rd
import junction as jn
import traffic_lights as tl
import network as nw

'''
Ideas for visualizing the network:

- All roads have a position from (-1, 0) to (8, 9)
- If ending of id is fw or bw then the road is 2-way, and the visualization need to shift them somewhat (right/left or up/down
    depending on the direction)
- If ending is not fw, then no shifting is needed
- Map from (-1,0) - (8,9) coordinates to opengl coordinates
'''

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

    network = nw.RoadNetwork(roads, junctions, T, optimizing = False)


    return network

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
    network = nw.RoadNetwork([road1, road2, road3, road4], [junction], T, optimizing = False)
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

    network = nw.RoadNetwork(roads, junctions, T, optimizing = False)

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
    
    network = nw.RoadNetwork([road1, road2, road3, road4], [junction], T, optimizing = False)
    return network

# Can also have a network where most of the road speed limits do not require a gradient...



    