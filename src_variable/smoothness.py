import road as rd
import network as nw 
import torch
import traffic_lights as tl
import junction as jn

def objective_function(densities, network):
    '''
    Calculates the total travel time on each road, and sums the results together
    '''
    total_travel = 0

    for j in range(len(densities)):
        # Go through each road
        road = network.roads[j]
        rho = densities[j]
        max_dens = road.max_dens
        # print(rho[0])

        times = list(rho.keys())    
        road_travel = 0

        prev_time = 0
        for l in range(road.pad+1, len(rho[0]) - road.pad):
            # Calculate inner integral for first time using inner points
            prev_time += road.dx * max_dens * (rho[0][l] + rho[0][l-1]) / 2

        for k in range(1, len(times)):
            t = times[k]
            dt = times[k] - times[k-1]
            # Go through all times
            # Calculate integral of density
            new_time = 0
            for l in range(road.pad+1, len(rho[t])-road.pad):
                # Go through all interior points
                new_time += road.dx * max_dens * (rho[t][l] + rho[t][l-1]) / 2
            
            road_travel += dt*(prev_time + new_time) / 2
            prev_time = new_time

        total_travel += road_travel

    return total_travel


def objective_function_queues(densities, queues, network):
    total_travel = 0

    for j in range(len(densities)):
        # Go through each road
        road = network.roads[j]
        rho = densities[j]
        max_dens = road.max_dens
        # print(rho[0])
        queue = not road.left

        times = list(rho.keys())    
        road_travel = 0

        prev_time = 0
        prev_queue = 0
        for l in range(road.pad+1, len(rho[0]) - road.pad):
            # Calculate inner integral for first time using inner points
            prev_time += road.dx *max_dens* (rho[0][l] + rho[0][l-1]) / 2

        if queue:
            prev_queue = queues[j][0] * max_dens

        for k in range(1, len(times)):
            t = times[k]
            dt = times[k] - times[k-1]
            # Go through all times
            # Calculate integral of density
            new_time = 0
            new_queue = 0
            for l in range(road.pad+1, len(rho[t])-road.pad):
                # Go through all interior points
                new_time += road.dx * max_dens *(rho[t][l] + rho[t][l-1]) / 2
            if queue:
                new_queue = queues[j][t] * max_dens

            road_travel += dt*(prev_time + prev_queue + new_time + new_queue) / 2
            prev_time = new_time
            prev_queue = new_queue

        total_travel += road_travel

    return total_travel

def test_smoothness_v1(v1, v2, cycle, T = 100):
    objectives = [0 for _ in range(len(v1))]
    for i, v in enumerate(v1):
        in_road = rd.Road(1, 100, 10, [v], [], flux_in=0.1)
        out_road = rd.Road(1, 100, 10, [v2], [], flux_in=0.1)
        traffic_light = tl.TrafficLightContinous(True, [0], [1], cycle)
        junction = jn.Junction([in_road, out_road], [0], [1], [[1.0]], [traffic_light], [])
        network = nw.RoadNetwork([in_road, out_road], [junction], T)
        densities, queues, _ = network.solve_cons_law()
        # print(objective_function(densities, network))
        objectives[i] = objective_function_queues(densities, queues, network)
    return objectives

def test_smoothness_v2(v1, v2, cycle, T = 100):
    objectives = [0 for _ in range(len(v2))]
    for i, v in enumerate(v2):
        in_road = rd.Road(1, 100, 10, [v1], [], flux_in=0.1)
        out_road = rd.Road(1, 100, 10, [v], [], flux_in=0.1)
        traffic_light = tl.TrafficLightContinous(True, [0], [1], cycle)
        junction = jn.Junction([in_road, out_road], [0], [1], [[1.0]], [traffic_light], [])
        network = nw.RoadNetwork([in_road, out_road], [junction], T)
        densities, queues, _ = network.solve_cons_law()
        # print(objective_function(densities, network))
        objectives[i] = objective_function_queues(densities, queues, network)
    return objectives

def test_smoothness_t1(v1, v2, t1, t2, T = 100):
    objectives = [0 for _ in range(len(t1))]
    for i, t in enumerate(t1):
        in_road = rd.Road(1, 100, 10, [v1], [], flux_in=0.1)
        out_road = rd.Road(1, 100, 10, [v2], [], flux_in=0.1)
        traffic_light = tl.TrafficLightContinous(True, [0], [1], [t, t2])
        junction = jn.Junction([in_road, out_road], [0], [1], [[1.0]], [traffic_light], [])
        network = nw.RoadNetwork([in_road, out_road], [junction], T)
        densities, queues, _ = network.solve_cons_law()
        # print(objective_function(densities, network))
        objectives[i] = objective_function_queues(densities, queues, network)
    return objectives

def test_smoothness_t2(v1, v2, t1, t2, T = 100):
    objectives = [0 for _ in range(len(t2))]
    for i, t in enumerate(t2):
        in_road = rd.Road(1, 100, 10, [v1], [], flux_in=0.1)
        out_road = rd.Road(1, 100, 10, [v2], [], flux_in=0.1)
        traffic_light = tl.TrafficLightContinous(True, [0], [1], [t1, t])
        junction = jn.Junction([in_road, out_road], [0], [1], [[1.0]], [traffic_light], [])
        network = nw.RoadNetwork([in_road, out_road], [junction], T)
        densities, queues, _ = network.solve_cons_law()
        # print(objective_function(densities, network))
        objectives[i] = objective_function_queues(densities, queues, network)
    return objectives


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    option = 4
    match option:
        case 1:
            objectives = test_smoothness_v1([float(i) for i in range(20, 50)], 30.0, [50.0,60.0], 1000.0)
            

            plt.plot([float(i) for i in range(20, 50)], [o.detach() for o in objectives])
            plt.show()
        case 2:
            objectives = test_smoothness_v2(30.0, [float(i) for i in range(20, 50)], [15.0,20.0], 100.0)
            plt.plot([float(i) for i in range(20, 50)], [o.detach() for o in objectives])
            plt.show()

        case 3:
            objectives = test_smoothness_t1(30.0, 30.0, [float(i) for i in range(10, 30)], 15.0, 100.0)
            plt.plot([float(i) for i in range(10, 30)], [o.detach() for o in objectives])
            plt.show()

        case 4:
            objectives = test_smoothness_t2(30.0, 30.0, 15.0, [float(i) for i in range(10, 30)],  100.0)
            plt.plot([float(i) for i in range(10, 30)], [o.detach() for o in objectives])
            plt.show()