import restarting_network as nw
import generate_general_networks as generate
import json

n_speeds = []
last_speed_idx = 0 # More accurately number of parameters related to the speed limit
n_cycles = []
upper_speeds = []
lower_speeds = []
upper_time = 200.0
lower_time = 10.0
control_points = []
config = None

def update_npeeds_ncycles_controls(speed_limits, cycle_times, new_control_points):
    global n_speeds
    global n_cycles
    global control_points
    global last_speed_idx

    n_speeds = []
    n_cycles = []
    speed_idx = 0
    for speeds in speed_limits:
        n_speeds.append(len(speeds))
        speed_idx += len(speeds)
    last_speed_idx = speed_idx

    for cycles in cycle_times:
        n_cycles.append(len(cycles))

    control_points = new_control_points


def update_limits(upper_speed_limit, lower_speed_limit, upper_cycle_time, lower_cycle_time):
    global upper_speeds
    global lower_speeds
    global upper_time
    global lower_time

    upper_speeds = upper_speed_limit
    lower_speeds = lower_speed_limit
    upper_time = upper_cycle_time
    lower_time = lower_cycle_time

def update_config(config_data):
    global config
    
    config = config_data


def load_bus_network(network_file, config_file):
    '''
    Function for initializing a bus network modelling kvadraturen
    with initial speed limits and speed limits as specified in the file
    filename. The grid spacing is also specified in the file
    '''
    f = open(network_file)
    data = json.load(f)
    f.close()
    T = data["T"]
    N = data["N"]
    speed_limits = data["speed_limits"] # Nested list
    lower_speed_limit = data["lower_speed_limit"] # Float
    upper_speed_limit = data["upper_speed_limit"] # Float
    control_points = data["control_points"] # Nested list
    cycle_times = data["cycle_times"] # Nested list
    lower_cycle_time = data["lower_cycle_time"] # Float
    upper_cycle_time = data["upper_cycle_time"] # Float

    update_npeeds_ncycles_controls(speed_limits, cycle_times, control_points)
    update_limits(upper_speed_limit, lower_speed_limit, upper_cycle_time, lower_cycle_time)
    
    f = open(config_file)
    data = json.load(f)
    f.close()
    update_config(data)

    return T, N, speed_limits, cycle_times


if __name__ == "__main__":
    option = 2

    match option:
        case 0:
            network_file = "optimization_cases/two_two_junction/network_file.json"
            config_file = "optimization_cases/two_two_junction/config_file.json"
            result_file = "optimization_results/general_optimization/two_two_junction_alt_start.json"
            
            T, N, speed_limits, cycle_times = load_bus_network(network_file, config_file)

            network = generate.two_two_junction(T, N, speed_limits, control_points, cycle_times[0], track_grad=False)
            densities, _, lengths, _ = network.solve_cons_law()
            times = list(densities[0].keys())

            for t in times:
                if 25 < t < 35:
                    print(t)
                    for i in range(len(lengths)):
                        print(lengths[i][t])
                    for i in range(len(densities)):
                        print(densities[i][t])
                    print()

                if 95 < t < 105:
                    print(t)
                    for i in range(len(lengths)):
                        print(lengths[i][t])
                    print()

        case 1:
            network_file = "optimization_cases/two_two_junction/network_file.json"
            config_file = "optimization_cases/two_two_junction/config_file.json"
            result_file = "optimization_results/general_optimization/two_two_junction_alt_start.json"
            
            T, N, speed_limits, cycle_times = load_bus_network(network_file, config_file)

            network = generate.two_two_junction(T, N, speed_limits, control_points, cycle_times[0], track_grad=False)
            network.solve_cons_law_counting()

        case 2:
            network_file = "optimization_cases/two_two_junction/network_file.json"
            config_file = "optimization_cases/two_two_junction/config_file.json"
            result_file = "optimization_results/general_optimization/two_two_junction_alt_start.json"
            
            T, N, speed_limits, cycle_times = load_bus_network(network_file, config_file)

            network = nw.GeneralRestartingRoadNetwork(T, N, speed_limits, control_points, cycle_times,
                                                      config, 2)
            densities, _, lengths, _, _, _, _ = network.solve_cons_law()

            times = list(densities[0].keys())

            for t in times:
                if 25 < t < 35:
                    print(t)
                    for i in range(len(lengths)):
                        print(lengths[i][t])
                    for i in range(len(densities)):
                        print(densities[i][t])
                    print()

                if 95 < t < 105:
                    print(t)
                    for i in range(len(lengths)):
                        print(lengths[i][t])
                    print()