# def initialize_road_network(T, roads, junctions, vmaxes):
#     '''
#     Initializing a road network given a time, roads and junctions as well as the speed limits
#     vmaxes.
#     Could probably instead just update densities and speed limits of existing system...

#     Need to be updated to take in vmaxes that may vary for each road
#     '''

#     n_speeds = len(control_points) + 1 # Number of different speed limits for each road

#     loaded_roads = []
#     loaded_junctions = []
#     for l, r in enumerate(roads):
#         # print(Vmax[l])
#         idx = l * n_speeds
#         loaded_roads.append(rn.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=[float(vmaxes[i]) for i in range(idx, idx+n_speeds)],
#                     scheme=r['Scheme'], limiter="minmod",
#                     initial = r["Init_distr"], inflow = r["Inflow"]))

#     for j in junctions:
#         loaded_junctions.append(rn.Junction(roads = [loaded_roads[i] for i in j["j_roads"]],
#                                         entering = j["entering"], leaving = j["leaving"],
#                                         distribution=j["distribution"], redlights=[]))
#     network = rn.RoadNetwork(loaded_roads, loaded_junctions, T, control_points)

#     return loaded_roads, loaded_junctions, network