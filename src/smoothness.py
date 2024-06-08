import torch
import generate_general_networks as generate
import FV_schemes as fv

def average_delay_time(bus_delays):
    avg_delay = torch.tensor(0.0)
    n_stops_reached = 0

    for delays in bus_delays.values():
        for delay in delays:
            if delay > 0.0:
                avg_delay = avg_delay + delay
                n_stops_reached += 1

    # Need some other way of determining the number of stops reached...
    # for now, assume the same number of stops reached for all iterations
    avg_delay = avg_delay #/ n_stops_reached
    # print(avg_delay)
    return avg_delay

def get_delay(network, delays, n_stops):
    tot_delay = 0
    for i in range(len(delays)):
        for j in range(len(delays[i])):
            tot_delay += delays[i][j]

    tot_delay.backward()
    # tot_delay = average_delay_time(delays) / n_stops
    # tot_delay.backward()

    speed_grads = network.get_speed_limit_grads()
    light_grads = network.get_traffic_light_grads()
    gradient = [s/3.6 for s in speed_grads] + light_grads

    return tot_delay.detach(), gradient

def get_throughput(network, densities):
    flux_out = 0
    times = list(densities[0].keys())
    for i, road in enumerate(network.roads):
        if not road.right:
            # Outgoing road
            for k in range(len(times)-1):
                t1 = times[k]
                t2 = times[k+1]
                # Find density at the end of the road
                end_rho1 = densities[i][t1][-1]
                end_rho2 = densities[i][t2][-1]

                flux1 =  fv.flux(end_rho1, road.gamma[0])
                flux2 =  fv.flux(end_rho2, road.gamma[0])
                # Minus to create a minimization problem
                flux_out -= (t2 - t1) * (flux1 + flux2) / 2

    flux_out.backward()
    speed_grad = network.get_speed_limit_grads()
    light_grad = network.get_traffic_light_grads()
    gradient = [s/3.6 for s in speed_grad] + light_grad

    return flux_out.detach(), gradient

# def get_total_travel_time(network, densities):
#     tot_int = 0
#     times = list(densities[0].keys())
#     dx = network.roads[0].dx

#     for i, road in enumerate(network.roads):
#         for k in range(len(times)-1):
#             t1 = times[k]
#             t2 = times[k+1]
#             # Find density at the end of the road
#             t1_int = torch.sum(densities[i][t1][1:-1]) + 0.5 * (densities[i][t1][-1] + densities[i][t1][0])
#             t2_int = torch.sum(densities[i][t2][1:-1]) + 0.5 * (densities[i][t2][-1] + densities[i][t2][0])

#             tot_int -= dx * (t2 - t1) * (t2_int + t1_int) / 2



#     tot_int.backward()
#     speed_grad = network.get_speed_limit_grads()
#     light_grad = network.get_traffic_light_grads()
#     gradient = [s/3.6 for s in speed_grad] + light_grad

#     return tot_int.detach(), gradient

def get_total_travel_time(network, densities, queues):
    tot_int = 0
    times = list(densities[0].keys())
    dx = network.roads[0].dx

    for i, road in enumerate(network.roads):
        for k in range(len(times)-1):
            t1 = times[k]
            t2 = times[k+1]
            # Find density at the end of the road
            t1_int = torch.sum(densities[i][t1][1:-1]) + 0.5 * (densities[i][t1][-1] + densities[i][t1][0])
            t2_int = torch.sum(densities[i][t2][1:-1]) + 0.5 * (densities[i][t2][-1] + densities[i][t2][0])
            q1 = queues[i][t1]
            q2 = queues[i][t2]
            tot_int += (t2 - t1) *(dx * (t2_int + t1_int) / 2 + (q1 + q2) / 2)

    tot_int.backward()
    speed_grad = network.get_speed_limit_grads()
    light_grad = network.get_traffic_light_grads()
    gradient = [s/3.6 for s in speed_grad] + light_grad

    return tot_int.detach(), gradient

def single_lane_step(T, N, speed_limits, controls, objective_type):
    network = generate.single_lane_network(T, N, speed_limits, controls, track_grad=True)

    match objective_type:
        case 0:
            # Bus delay as objective function
            _, _, _, delays, n_stops = network.solve_cons_law_counting()
            objective, gradient = get_delay(network, delays, n_stops)

        case 1:
            # Throughput as objective function
            densities, _, _, _, _ = network.solve_cons_law_counting()
            objective, gradient = get_throughput(network, densities)

        case 2:
            # Total travel time
            densities, queues, _, _, _ = network.solve_cons_law_counting()
            objective, gradient = get_total_travel_time(network, densities, queues)
    return objective, gradient

def single_lane_smooth(T, N, speed_limits_in, controls, increments, objective_type=0):
    objectives = []
    gradients = []

    for inc in increments:
        speed_limits = [torch.tensor(speed + inc) for speed in speed_limits_in]

        # print(speed_limits)
        # print(controls)
        # print(cycle)
        # print(objective_type)
        objective, grad = single_lane_step(T, N, speed_limits, controls, objective_type)

        objectives.append(objective)
        gradients.append(grad[0])
        
    return objectives, gradients

def single_junction_step(T, N, speed_limits, controls, cycle, objective_type):
    network = generate.single_junction_network(T, N, speed_limits, controls, cycle, track_grad=True)

    match objective_type:
        case 0:
            # Bus delay as objective function
            _, _, _, delays, n_stops = network.solve_cons_law_counting()
            objective, gradient = get_delay(network, delays, n_stops)

        case 1:
            # Throughput as objective function
            densities, _, _, _, _ = network.solve_cons_law_counting()
            objective, gradient = get_throughput(network, densities)

        case 2:
            # Total travel time
            densities, queues, _, _, _ = network.solve_cons_law_counting()
            objective, gradient = get_total_travel_time(network, densities, queues)

    return objective, gradient

def single_junction_smooth(T, N, speed_limits_in, controls, cycle_in, increments, param_idx = 0, 
                            objective_type = 0):
    objectives = []
    gradients = []

    for inc in increments:
        speed_limits = [[torch.tensor(speed)] for speed in speed_limits_in]
        cycle = [torch.tensor(cycle_in[0]), torch.tensor(cycle_in[1])]
        if param_idx < 2:
            speed_limits[param_idx] = [torch.tensor(speed_limits_in[param_idx]) + inc]
        else:
            cycle[param_idx - 2] = torch.tensor(cycle_in[param_idx - 2] + inc)

        # print(speed_limits)
        # print(controls)
        # print(cycle)
        # print(objective_type)
        objective, grad = single_junction_step(T, N, speed_limits, controls, cycle, objective_type)

        objectives.append(objective)
        gradients.append(grad[param_idx])
        
    return objectives, gradients

def two_two_step(T, N, speed_limits, controls, cycle, objective_type):
    network = generate.two_two_junction(T, N, speed_limits, controls, cycle, track_grad=True)

    match objective_type:
        case 0:
            # Bus delay as objective function
            _, _, _, delays, n_stops = network.solve_cons_law_counting()
            objective, gradient = get_delay(network, delays, n_stops)

        case 1:
            # Throughput as objective function
            densities, _, _, _, _ = network.solve_cons_law_counting()
            objective, gradient = get_throughput(network, densities)

        case 2:
            # Total travel time
            densities, queues, _, _, _ = network.solve_cons_law_counting()
            objective, gradient = get_total_travel_time(network, densities, queues)

    return objective, gradient

def two_two_junction_smooth(T, N, speed_limits_in, controls, cycle_in, increments, param_idx = 0, 
                            objective_type = 0):
    objectives = []
    gradients = []

    for inc in increments:
        speed_limits = [[torch.tensor(speed)] for speed in speed_limits_in]
        cycle = [torch.tensor(cycle_in[0]), torch.tensor(cycle_in[1])]
        if param_idx < 4:
            speed_limits[param_idx] = [torch.tensor(speed_limits_in[param_idx]) + inc]
        else:
            cycle[param_idx - 4] = torch.tensor(cycle_in[param_idx - 4] + inc)

        # print(speed_limits)
        # print(controls)
        # print(cycle)
        # print(objective_type)
        objective, grad = two_two_step(T, N, speed_limits, controls, cycle, objective_type)

        objectives.append(objective)
        if param_idx < 4:
            gradients.append(grad[param_idx])
        else:
            gradients.append(grad[param_idx])
        
    return objectives, gradients
        
if __name__ == "__main__":
    option = 2
    match option:
        case 0:
            import numpy as np
            T = 200
            N = 3
            controls = [[],[],[],[]]
            speeds = [[torch.tensor(40.0)], [torch.tensor(40.0)], [torch.tensor(40.0)], [torch.tensor(40.0)]]
            cycle = [torch.tensor(60.0), torch.tensor(60.0)]    
            # print(two_two_junction_smooth(T, N, speeds, controls, cycle, np.array([0.0]), 3, 0))
            # print(two_two_step(T, N, speeds, controls, cycle, 0))
            speeds = [40.0, 40.0, 40.0, 40.0]
            cycle = [60.0, 60.0]

            gradient = []
            obj, grad = two_two_junction_smooth(T, N, speeds, controls, cycle, np.array([0.0, 1.0]), 0, 2)
            gradient.append(grad)

            print(gradient)

        case 1:
            import numpy as np
            T = 100
            N = 3
            speeds = [40.0]
            increments = np.linspace(-5,5,3)
            print(single_lane_smooth(T, N, speeds, [], increments, 2))

        case 2:
            import numpy as np
            T = 200
            N = 3
            controls = [[],[],[],[]]
            speeds = [[torch.tensor(40.0)], [torch.tensor(40.0)], [torch.tensor(40.0)], [torch.tensor(40.0)]]
            cycle = [torch.tensor(60.0), torch.tensor(60.0)]    
            # print(two_two_junction_smooth(T, N, speeds, controls, cycle, np.array([0.0]), 3, 0))
            # print(two_two_step(T, N, speeds, controls, cycle, 0))
            speeds = [40.0, 40.0]
            cycle = [60.0, 60.0]

            gradient = []
            obj, grad = single_junction_smooth(T, N, speeds, controls, cycle, np.array([0.0, 1.0]), 0, 2)
            gradient.append(grad)

            print(gradient)



