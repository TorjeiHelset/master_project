import json
import initial_and_bc as ib
import road as rd
import junction as jn
import traffic_lights as tl
import network as rn
import FV_schemes as fv


def read_json(filename):
    f = open(filename)
    data = json.load(f)
    f.close()

    T = data["T"]
    roads = []
    junctions = []

    for i, road in enumerate(data["roads"]):
        L = road["L"]
        b = road["b"]
        Vmax = [v for v in road["Vmax"]]
        if isinstance(road["Vlow"], list):
            low = [v for v in road["Vlow"]]
        else:
            low = [road["Vlow"]] * len(Vmax)
        
        if isinstance(road["Vhigh"], list):
            high = [v for v in road["Vhigh"]]
        else:
            high = [road["Vhigh"]] * len(Vmax)

        N = road["N"]
        Scheme = road["Scheme"]
        Init_dens = road["Init_dens"]
        Init_distr = ib.init_density(Init_dens, road["Init_distr"])
        Inflow =  road["Inflow"]
        ControlPoints = road["ControlPoints"]
        if road["f_in"] >= 0:
            f_in = fv.D_non_torchscript(road["f_in"], (80/3.6) / L)
        else:
            f_in = -1

        if "max_dens" in road:
            road_dict = {"L" : L, "b" : b, "Vmax" : Vmax, "N" : N,
                        "Scheme" : Scheme, "Init_dens" : Init_dens, 
                        "Init_distr": Init_distr, "Inflow" : Inflow,
                        "low" : low, "high" : high, "ControlPoints" : ControlPoints,
                        "max_dens" : road["max_dens"], 'left_pos' : road['left_pos'],
                        'right_pos' : road['right_pos'],
                        "f_in" : f_in}
        else:
            road_dict = {"L" : L, "b" : b, "Vmax" : Vmax, "N" : N,
                        "Scheme" : Scheme, "Init_dens" : Init_dens, 
                        "Init_distr": Init_distr, "Inflow" : Inflow,
                        "low" : low, "high" : high, "ControlPoints" : ControlPoints,
                        'left_pos' : road['left_pos'],'right_pos' : road['right_pos'],
                        "f_in" : f_in}
        
        print(f"{i+1} roads initialized")
        roads.append(road_dict)

    print("All roads initialized")

    for i, junction in enumerate(data["Junctions"]):
        j_roads = junction["Roads"]
        entering = junction["Entering"]
        leaving = junction["Leaving"]
        distribution = junction["Distribution"]
        trafficlights = junction["Trafficlights"]
        coupled = junction["CoupledTrafficlights"]


        junction_dict = {"j_roads" : j_roads, "entering" : entering, "leaving" : leaving,
                         "distribution" : distribution, "trafficlights" : trafficlights,
                         "coupled" : coupled}
        
        junctions.append(junction_dict)
        
        print(f"{i+1} junctions initialized")
    
    print("All junctions initialized")

    return T, roads, junctions


def initialize_road_network(filename):
    T, roads, junctions = read_json(filename)

    loaded_roads = []
    loaded_junctions = []
    for l, r in enumerate(roads):
        if "max_dens" in r:
            loaded_roads.append(rd.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=r['Vmax'],
                        control_points=r["ControlPoints"],
                        scheme=r['Scheme'], limiter="minmod",
                        initial = r["Init_distr"], inflow = r["Inflow"],
                        max_dens=r["max_dens"], left_pos = r['left_pos'],
                        right_pos = r['right_pos'],
                        flux_in = r["f_in"]))
        else:
            loaded_roads.append(rd.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=r['Vmax'],
                        control_points=r["ControlPoints"],
                        scheme=r['Scheme'], limiter="minmod",
                        initial = r["Init_distr"], inflow = r["Inflow"], 
                        left_pos = r['left_pos'], right_pos = r['right_pos'],
                        flux_in = r["f_in"]))

    for j in junctions:
        loaded_traffic_lights = []
        for light in j["trafficlights"]:
            # Go through all traffic lights in junction
            if light["StartingState"] == "Green":
                start = False
            else:
                start = True

            loaded_traffic_lights.append(tl.TrafficLightContinous(start, light["Entering"],
                                        light["Leaving"], light["Cycle"]))
        
        loaded_coupled = []
        for light in j["coupled"]:
            # Go through all coupled traffic lights in junction
            if light["StartingState"] == "Green":
                start = True
            else:
                start = False
            loaded_coupled.append(tl.CoupledTrafficLightContinuous(start, light["aEntering"],
                                    light["aLeaving"], light["bEntering"],
                                    light["bLeaving"], light["Cycle"]))
            
        loaded_junctions.append(jn.Junction(roads = [loaded_roads[i] for i in j["j_roads"]],
                                        entering = j["entering"], leaving = j["leaving"],
                                        distribution=j["distribution"], 
                                        trafficlights=loaded_traffic_lights,
                                        coupled_trafficlights=loaded_coupled))
        
    network = rn.RoadNetwork(loaded_roads, loaded_junctions, T)

    return loaded_roads, loaded_junctions, network

def convert_from_tensor(rho):
    out_dict = {t: None for t in rho.keys()}
    for t in rho.keys():
        out_dict[t] = {float(x) : None for x in rho[t].keys()}
        for x in rho[t].keys():
            try:
                out_dict[t][float(x)] = rho[t][x].tolist()
            except:
                out_dict[t][float(x)] = rho[t][x]

    return out_dict

