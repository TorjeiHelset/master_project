import json
import initial_and_bc as ib

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
        Vmax = road["Vmax"] #/ 3.6 # Translate from km/h to m/s
        low = road["Vlow"] #/ 3.6
        high = road["Vhigh"] #/ 3.6
        N = road["N"]
        Scheme = road["Scheme"]
        Init_dens = road["Init_dens"]
        Init_distr = ib.init_density(Init_dens, road["Init_distr"])
        Inflow =  road["Inflow"]

        road_dict = {"L" : L, "b" : b, "Vmax" : Vmax, "N" : N,
                     "Scheme" : Scheme, "Init_dens" : Init_dens, 
                      "Init_distr": Init_distr, "Inflow" : Inflow,
                      "low" : low, "high" : high}
        
        print(f"{i+1} roads initialized")
        roads.append(road_dict)

    print("All roads initialized")

    for i, junction in enumerate(data["Junctions"]):
        j_roads = junction["Roads"]
        entering = junction["Entering"]
        leaving = junction["Leaving"]
        distribution = junction["Distribution"]
        redlights = junction["Redlights"]


        junction_dict = {"j_roads" : j_roads, "entering" : entering, "leaving" : leaving,
                         "distribution" : distribution, "redlights" : redlights}
        
        junctions.append(junction_dict)
        
        print(f"{i+1} junctions initialized")
    
    print("All junctions initialized")

    return T, roads, junctions



    
if __name__ == "__main__":
    file = 1

    match file:
        case 0:
            read_json('configs/single_lane.json')

        case 1:
            read_json('configs/simple_1-1.json')
