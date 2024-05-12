###################################################
# Problem: Running out of memory if many iterations are performed
# - Need to release memory between iterations of optimization algorithm
# - Easiest way to do this is to put each iteration of the optimization algorithm inside
#   a dedicated function. When this function is exited, the memory will be released.

# No guarantee that global optimum found so multiple searches might be necessary
# - How to pick starting points?
# - Can parts of the search space be eliminated before? I.e. if the current 
#   iterate has been visited before, it will lead to the same solution, so 
#   algorithm should stop

#################################################################

# Importing necessary libraries:
import torch
import json
import generate_kvadraturen as gk
import numpy as np
import memory_profiler
import os
import json
import restarting_network as nw

# torch.autograd.set_detect_anomaly(True)


n_speeds = []
last_speed_idx = 0
n_cycles = []
upper_speeds = []
lower_speeds = []
upper_time = 200.0
lower_time = 10.0
control_points = []
config = None

################################
# Updating global values:
################################
# @memory_profiler.profile
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

# @memory_profiler.profile
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

################################
# Loading from file
################################
# @memory_profiler.profile
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

################################
# Converting from list of params to nested list
################################

# @memory_profiler.profile
def extract_params(speed_limits, cycle_times):
    params = []
    for speeds in speed_limits:
        for s in speeds:
            params.append(s)
    
    for cycles in cycle_times:
        for t in cycles:
            params.append(t)
    
    return np.array(params)

# @memory_profiler.profile
def get_speeds_cycles_from_params(params):
    idx = 0
    speed_limits = []
    cycle_times = []

    for i in range(len(n_speeds)):
        speed_limits.append([])
        for j in range(n_speeds[i]):
            speed_limits[i].append(params[idx])
            idx += 1

    for i in range(len(n_cycles)):
        cycle_times.append([])
        for j in range(n_cycles[i]):
            cycle_times[i].append(params[idx])
            idx += 1

    return speed_limits, cycle_times

# @memory_profiler.profile
def update_speeds_cycles(params, speed_structure, cycle_structure):
    idx = 0
    for i, speeds in enumerate(speed_structure):
        for j, s in enumerate(speeds):
            speed_structure[i][j] = params[idx]
            idx += 1
    
    for i, cycles in enumerate(cycle_structure):
        for j, t in enumerate(cycles):
            cycle_structure[i][j] = params[idx]
            idx += 1
    return speed_structure, cycle_structure


################################
# Helper function for gradient descent step
################################
# @memory_profiler.profile
def scale_gradient(gradient, prev_params, max_update):
    '''
    For now scale the gradient so that the largest updating step is equal to 20 km/h or 20 s. Note the gradient for the speeds
    is given in m/s, so the first step is to scale these elements to get km/h. Any elements that are sent outside the boundary 
    are set equal to boundary at a later step.

    Alternate step is to not allow the updating step to send any parameters not at the boundary outside the boundary.
    Both methods should be tested...
    '''

    # Tracking of gradient was turned on after dividing by 3.6
    # Account for this fact
    # Scale the first elements of gradient by 3.6
    for i in range(last_speed_idx):
        gradient[i] = gradient[i] / 3.6

    # Set elements on boundary equal to 0:
    for i in range(last_speed_idx):
        if prev_params[i] <= lower_speeds[i] and gradient[i] > 1.e-5:
            gradient[i] = 0
        elif prev_params[i] >= upper_speeds[i] and gradient[i] <- 1.e-5:
            gradient[i] = 0
        
    for i in range(last_speed_idx, len(gradient)):
        if prev_params[i] <= lower_time and gradient[i] >  1.e-5:
            gradient[i] = 0
        elif prev_params[i] >= upper_time and gradient[i] < - 1.e-5:
            gradient[i] = 0
        
    # Scale gradient either so that a new parameter reaches boundary, or so that the largest 
    # updating step is equal to the maximum updating step
    scaling_factor = np.inf
    pot_factor = np.inf

    for i in range(last_speed_idx):
        if np.abs(gradient[i]) < 1.e-5:
            continue
        elif gradient[i] < 0:
            # Going towards upper boundary
            # print(i, upper_speed - prev_params[i])
            pot_factor = (prev_params[i] - upper_speeds[i]) / gradient[i]
            # if pot_factor < 0:
            #     print(f"Scaling less than 0 at for component {i}, parameter: {prev_params[i]}")
        else:
            # Going towards lower boundary
            # print(i, prev_params[i] - lower_speed)
            pot_factor = (prev_params[i] - lower_speeds[i]) / gradient[i]
            # if pot_factor < 0:
            #     print(f"Scaling less than 0 at for component {i}, parameter: {prev_params[i]}")


        # pot_factor = min(max_update / np.abs(gradient[i]), pot_factor)
        # scaling_factor = min(pot_factor, scaling_factor)
        if np.abs(gradient[i]) > 1.e-5:
            scaling_factor = min(max_update / np.abs(gradient[i]), scaling_factor)

        # print(i, pot_factor, scaling_factor)

    for i in range(last_speed_idx, len(gradient)):
        if np.abs(gradient[i]) < 1.e-6:
            continue
        elif gradient[i] < 0:
            # Going towards upper boundary
            pot_factor = (prev_params[i] - upper_time) / gradient[i]
        else:
            # Going towards lower boundary
            pot_factor = (prev_params[i] - lower_time) / gradient[i]

        # pot_factor = min(max_update / np.abs(gradient[i]), pot_factor)
        # scaling_factor = min(pot_factor, scaling_factor)
        if np.abs(gradient[i]) > 1.e-5:
            scaling_factor = min(max_update / np.abs(gradient[i]), scaling_factor)


    if scaling_factor > 1.e6:
        scaling_factor = 0
    if scaling_factor < 0:
        raise ValueError("Scaling factor should be positive")
    # print(scaling_factor)
    # print(gradient * scaling_factor)
    scaled_gradient = [g * scaling_factor for g in gradient]
    return scaling_factor, scaled_gradient

# @memory_profiler.profile
def update_params(prev_params, scaled_grad):
    '''
    Tries to do the updating step
        new_params = prev_params - scaled_grad
    Any elements of new_params that are outside the allowed region will be set equal to the upper/lower limits
    '''
    new_params = prev_params - scaled_grad

    for i in range(last_speed_idx):
        if new_params[i] < lower_speeds[i]:
            new_params[i] = lower_speeds[i]
        elif new_params[i] > upper_speeds[i]:
            new_params[i] = upper_speeds[i]

    for i in range(last_speed_idx, len(scaled_grad)):
        if new_params[i] < lower_time:
            new_params[i] = lower_time
        elif new_params[i] > upper_time:
            new_params[i] = upper_time
    return new_params

# @memory_profiler.profile
def check_armijo(prev_objective, new_objective, prev_gradient, scaling_factor, c1 = 1e-6):
    return new_objective <= prev_objective - c1 * scaling_factor * np.dot(prev_gradient, prev_gradient)

################################
# Objective functions:
################################

# @memory_profiler.profile
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

################################
# Gradient descent functions
################################

# @memory_profiler.profile
def get_grad(objective, network):
    objective.backward()
    speed_limits = network.get_speed_limit_grads()
    traffic_lights = network.get_traffic_light_grads()
    gradient = speed_limits + traffic_lights
    return np.array(gradient)

# @memory_profiler.profile
def gradient_descent_first_step(T, N, speed_limits, cycle_times):
    '''
    First step of optimization approach. In this step there is no previous results to 
    compare with.
    '''
    
    # Create first network
    print("Creating the network...")
    restart_network = nw.RestartingRoadNetwork(T, N, speed_limits, control_points, cycle_times, config)

    print("Solving the conservation law...")
    _, _, _, bus_delays, n_stops_reached, speed_grads, light_grads = restart_network.solve_cons_law()
    objective = 0.0
    for i in range(len(bus_delays)):
        for delay in bus_delays[i]:
            objective += delay

    objective = objective / n_stops_reached
    gradient = speed_grads + light_grads

    print(f"Objective: {objective}")

    return gradient, objective

# @memory_profiler.profile
def create_network_and_calculate(T, N, new_params):
    # Creating the network:
    speed_limits, cycle_times = get_speeds_cycles_from_params(new_params)

    restart_network = nw.RestartingRoadNetwork(T, N, speed_limits, control_points, cycle_times, config)

    _, _, _, bus_delays, n_stops_reached, speed_grads, light_grads = restart_network.solve_cons_law()
    objective = 0.0
    for i in range(len(bus_delays)):
        for delay in bus_delays[i]:
            objective += delay

    objective = objective / n_stops_reached
    gradient = speed_grads + light_grads

    print(f"Objective: {objective}")


    return gradient, objective


# @memory_profiler.profile
def gradient_descent_step(prev_params, prev_gradient, prev_objective, T, N):
    '''
    Performs gradient step with armijo line search.
    Assumes that the first step is always so big that a necessary decrease is achieved
    If armijo condition is not satisfied, reduce step length until it is
    The first step length is chosen in such a way that the parameter being changed the 
    most is changed by some fixed number. This could i.e. be 10 or 20, both of which are 
    relevant for speed limits and cycle times. When scaling the gradient, elements too close
    to the boundary should not be used for the scaling (unless no other are non-zero).
    Needs to take in previous objective and previous gradient to be able to do line-search
    Should also take in upper/lower bounds of parameters
    Only do updating until a certain number of line search fails reached. Can i.e. stop iterating
    when the parameter being updated the most is updated by less than 1
    The type of objective function should be be specified as an argument
    '''
    armijo_fails = 0
    step_taken = False
    max_update = 20

    while not step_taken:
        # 1. Scale gradient so that parameter being updated most is updated by max_update
        # gradient = scale_gradient(gradient, upper_limits, lower_limits)
        print(f"Scaling the gradient to achieve a maximum updating of {max_update}...")
        scaling_factor, scaled_grad = scale_gradient(prev_gradient, prev_params, max_update)
        if np.linalg.norm(scaled_grad) < 1.e-6:
            print(f"Either boundary or stationary point reached")
            return prev_params, prev_gradient, prev_objective

        # print("New params:")
        # print(prev_params - scaled_grad)
        # # 2. Use previous gradient and prFTevious parameters to update parameters
        # new_params = update_params(old_params, gradient, upper_limits, lower_limits)
        print(f"Updating the parameters...\n")
        new_params = update_params(prev_params, scaled_grad)

        # 3. Create network using the new parameters and calculate the objective and gradient
        print(f"Creating the network with updated parameters and calculating the gradient/objective:")
        new_gradient, new_objective = create_network_and_calculate(T, N, new_params)
        
        # 4. Check armijo condition:
        print(f"Checking the Armijo condition...")
        if new_gradient is None:
            armijo_satisfied = False
        elif new_objective > prev_objective:
            armijo_satisfied = False
        else:
            armijo_satisfied = check_armijo(prev_objective, new_objective, prev_gradient, scaling_factor)

        if armijo_satisfied:
            # If armijo condition is satisfied, the new iterate can be returned
            return new_params, new_gradient, new_objective
        
        else:
            # If armijo condition not satisfied, increment the number of fails and 
            # decrease the maximum change in parameters
            print("Armijo condition failed! Need to reduce the updating step")
            armijo_fails += 1
            max_update = max_update * 0.5

        if armijo_fails > 5 or max_update < 1:
            # If either the number of fails are too many or the maximum update becomes too small
            # stop iterating and return the old iterate
            return prev_params, prev_gradient, prev_objective

# @memory_profiler.profile
def gradient_descent(network_file, config_file, result_file = "optimization_results/new_result.json", 
                     overwrite = False, max_iter = 100, tol = 1.e-4, debugging = True):
    '''
    Full method for the gradient descent algorithm. The funcion should take in a filename for 
    the initial configuration of the network. In addition the objetive type needs to be specified.

    Nothing inside this function should require gradient!
    Important that any references to parameters requiring gradient is not kept inside here
    '''

    # Check if the result_file exists:
    if os.path.isfile(result_file):
        print(f"File {result_file} already exists!")
        # File already exists
        if overwrite:
            # The file will be overwritten
            print(f"The file will be overwritten!")
            
        else:
            # Modify the file name slightly to prevent overwriting
            result_file = result_file[:-5]+"_copy.json"
            print(f"Saving results to {result_file} instead")
            
    else:
        print(f"{result_file} does not exists!")
        print(f"Saving results to {result_file}")


    # 1. Load configuration from filename
    print("Loading from file")
    T, N, speed_limits, cycle_times = load_bus_network(network_file, config_file)
    if debugging:
        T = 60

    # 1.1 Map from speed_limits and cycle_times to one parameter list
    print("Extracting parameters")
    prev_params = extract_params(speed_limits, cycle_times) # Turns the nested lists into one combined np.array

    # 2. Do first simulation using the initial configuration
    print("\n------------------------------------------------------")
    print("Starting the first step of the gradient descent algorithm:")
    prev_grad, prev_objective = gradient_descent_first_step(T, N, speed_limits, cycle_times)

    # 2.0 Initialize result file:
    result_dict = {
        "network_file" : network_file,
        "config_file" : config_file,
        "ad_method" : "backward",
        "upper_speeds" : upper_speeds,
        "lower_speeds" : lower_speeds,
        "upper_time" : upper_time,
        "lower_time" : lower_time,
        "n_speeds" : n_speeds,
        "last_speed_idx" : last_speed_idx,
        "n_cycles" : n_cycles,
        "control_points" : control_points,
        "parameters" : [
            list(prev_params)
        ],
        "gradients" : [
            list(prev_grad)
        ],
        "objectives" : [
            float(prev_objective)
        ]
    }

    with open(result_file, "w") as outfile:
        outfile.write(json.dumps(result_dict, indent=4))

    # 2.1 Store results from first iteration
    params_history = [prev_params]
    grad_history = [prev_grad]
    objective_history = [prev_objective]

    # 2.2 Print results of first simulation:
    #       Give option for detailed print, and less detailed print
    print("Initial step of the gradient descent algorithm finished!")
    print(f"Objective value: {prev_objective}")
    print(f"Gradient (before scaling): {prev_grad}")
    print("------------------------------------------------------\n")
    
    # 3. Perform gradient descent steps until some criteria is reached
    curr_iter = 0
    error = tol + 1
    if debugging:
        max_iter = 1

    while curr_iter < max_iter and error > tol:

        # 3.1 Do the gradient descnent iteration
        print("\n------------------------------------------------------")
        print(f"Performing step {curr_iter+1} of the algorithm:")
        new_params, new_grad, new_objective = gradient_descent_step(prev_params, prev_grad, prev_objective, T, N)
        print(new_objective)

        # 3.2 Print results from iteration
        #       Give option for detailed print, and less detailed print
        print(f"Iteration {curr_iter+1} finished!")
        print(f"Next set of parameters:{new_params}")
        print(f"New objective value: {new_objective}")
        print(f"new gradient (before scaling): {new_grad}")

        # 3.3 Calculate error and increment number of iterations
        curr_iter += 1
        error = np.linalg.norm(new_params - prev_params)
        print(f"The normed difference between the previous set of parameters and the new set of parameter")
        print(f"is equal to {error}.")
        print("------------------------------------------------------\n")
        

        # 3.4 Store the new results before doing the next iteration
        params_history.append(new_params)
        grad_history.append(new_grad)
        objective_history.append(new_objective)
        prev_params, prev_grad, prev_objective = new_params, new_grad, new_objective

        # 3.5 Save results to result file
        print("Saving results to file...")
        new_result_dict = {}
        with open(result_file, "r") as openfile:
            prev_result_dict = json.load(openfile)
            new_result_dict["network_file"] = prev_result_dict["network_file"]
            new_result_dict["config_file"] = prev_result_dict["config_file"]
            new_result_dict["ad_method"] = prev_result_dict["ad_method"]
            new_result_dict["upper_speeds"] = prev_result_dict["upper_speeds"]
            new_result_dict["lower_speeds"] = prev_result_dict["lower_speeds"]
            new_result_dict["upper_time"] = prev_result_dict["upper_time"]
            new_result_dict["lower_time"] = prev_result_dict["lower_time"]
            new_result_dict["n_speeds"] = prev_result_dict["n_speeds"]
            new_result_dict["last_speed_idx"] = prev_result_dict["last_speed_idx"]
            new_result_dict["n_cycles"] = prev_result_dict["n_cycles"]
            new_result_dict["control_points"] = prev_result_dict["control_points"]

            old_params = prev_result_dict["parameters"]
            old_params.append(list(prev_params))
            new_result_dict["parameters"] = old_params

            old_grads = prev_result_dict["gradients"]
            old_grads.append(list(prev_grad))
            new_result_dict["gradients"] = old_grads

            old_objectives = prev_result_dict["objectives"]
            old_objectives.append(float(prev_objective))
            new_result_dict["objectives"] = old_objectives
        
        with open(result_file, "w") as outfile:
            outfile.write(json.dumps(new_result_dict, indent=4))

    # 4. Stopping criteria reached. Return results from simulation
    print("Stopping criteria reached! Exiting the algorithm...")
    return params_history, grad_history, objective_history


if __name__ == "__main__":
    option = 1
    match option:
        case 0:
            # Run small example with e18
            network_file = "kvadraturen_networks/with_e18/network_1_2.json"
            config_file = "kvadraturen_networks/with_e18/config_1_1.json"
            result_file = "optimization_results/network12_config11_restart.json"
            gradient_descent(network_file, config_file, result_file,
                             overwrite=False, debugging=False)
            
        case 1:
            # Run larger example with e18
            network_file = "kvadraturen_networks/with_e18/network_2_1.json"
            config_file = "kvadraturen_networks/with_e18/config_2_1.json"
            result_file = "optimization_results/network21_config21_restart.json"
            gradient_descent(network_file, config_file, result_file,
                             overwrite=False, debugging=False)
            
        case 2:
            # Run larger example with e18 with different starting point
            network_file = "kvadraturen_networks/with_e18/network_2_2.json"
            config_file = "kvadraturen_networks/with_e18/config_2_1.json"
            result_file = "optimization_results/network22_config21_restart.json"
            gradient_descent(network_file, config_file, result_file,
                             overwrite=False, debugging=False)
            
        case 3:
            # Run larger example with e18 with different starting point and different config
            network_file = "kvadraturen_networks/with_e18/network_2_2.json"
            config_file = "kvadraturen_networks/with_e18/config_2_2.json"
            result_file = "optimization_results/network22_config22_restart.json"
            gradient_descent(network_file, config_file, result_file,
                             overwrite=False, debugging=False)

        case 4:
            network_file = "kvadraturen_networks/with_e18/network_3.json"
            config_file = "kvadraturen_networks/with_e18/config_3_1.json"
            result_file = "optimization_results/network3_config31_restart.json"
            gradient_descent(network_file, config_file, result_file,
                             overwrite=False, debugging=False)
            
        case 5:
            network_file = "kvadraturen_networks/with_e18/network_4.json"
            config_file = "kvadraturen_networks/with_e18/config_4_1.json"
            result_file = "optimization_results/network4_config41_restart.json"
            gradient_descent(network_file, config_file, result_file,
                             overwrite=False, debugging=False)