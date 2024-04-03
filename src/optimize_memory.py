#################################################################
# Problem: Running out of memory if many iterations are performed
# Need to release memory between iterations
# One fix -> instead of creating new networks, just reset densities
# and update parameters

# Multiple searches
# Eliminate parts of domain already visited naively/greedily



# Check that queue works as it should
# How to check: If no flux is allowed to leave the system, then 
# the integral of the road and the queue should be the same for different 
# speeds.
#################################################################

# Full iterative approach of setting up road network and then iteratively 
# choosing new speed limits
import time as time
# pyswarm - used to get approximation of exact solution

import loading_json as load
# import road_network as rn
import road as rd
import junction as jn
import traffic_lights as tl
import network as rn
import torch
import FV_schemes as fv
import matplotlib.pyplot as plt

def jacobi(objective, params):
    j = torch.zeros(len(params))

    for i in range(len(params)):
        derivative =  torch.autograd.grad(objective, params[i], create_graph=True, allow_unused=True)[0]
        if derivative:
            j[i] = derivative

    return j

def hessian(grad, params):
    H = torch.zeros((len(params), len(params)))

    for i in range(len(params)):
        for j in range(len(params)):
            derivative =  torch.autograd.grad(grad[i], params[j], create_graph=True, allow_unused=True)[0]
            if derivative:
                H[i,j] = derivative

    return H


def out_flux(history, network):
    '''
    Calculates the sum of the time integral of the flux out of each road
    Integral calculated as a linear interpolation
    '''
    total_out = 0
    for j in range(len(history)):
        # Going through each road
        rho = history[j]
        # print(j)
        # print(rho)
        padding = network.roads[j].pad
        gamma = network.roads[j].gamma
        max_dens = network.roads[j].max_dens

        # Go through each of the roads and calculate integral of flux out of road
        # Calculate integral as linear interpolation
        times = list(rho.keys())
        for i in range(1, len(times)):
            gamma1 = network.roads[j].get_gamma(times[i])
            gamma2 = network.roads[j].get_gamma(times[i-1])

            dt = times[i] - times[i-1]
            right = fv.flux(rho[times[i]][-padding-1], gamma1) * max_dens
            left = fv.flux(rho[times[i-1]][-padding-1], gamma2) * max_dens

            total_out += dt*(right + left) / 2

    return -total_out



def point_flux(history, network):
    '''
    Calculates the flux at one point in time and space for one road
    '''

    time_interval = [network.T/3, 2*network.T/3]
    road_idx = 0
    pnt_idx = 20
    total_out = 0

    rho = history[road_idx]
    gamma = network.roads[road_idx].gamma
    max_dens = network.roads[road_idx].max_dens
    # print("Gamma")
    # print(gamma)

    times = list(rho.keys())
    sub_times = [t for t in times if time_interval[0] <= t <= time_interval[1]]

    for i in range(1, len(sub_times)):
        dt = sub_times[i] - sub_times[i-1]
        gamma1 = network.roads[road_idx].get_gamma(times[i])
        gamma2 = network.roads[road_idx].get_gamma(times[i-1])

        right = fv.flux(rho[times[i]][pnt_idx], gamma1) * max_dens
        left = fv.flux(rho[times[i-1]][pnt_idx], gamma2) * max_dens
        total_out += dt * (right + left) / 2

    return -total_out

def total_travel_time(history, network):
    '''
    Calculates the total travel time on each road, and sums the results together
    '''
    total_travel = 0

    for j in range(len(history)):
        # Go through each road
        road = network.roads[j]
        rho = history[j]
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


def total_travel_time_queue(history,queues,network):
    '''
    Calculates the total travel time on each road, and sums the results together
    '''
    total_travel = 0

    for j in range(len(history)):
        # Go through each road
        road = network.roads[j]
        rho = history[j]
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

def exit_flux(history, network):            
    total_out = 0
    for j in range(len(history)):
        # Going through each road
        if not network.roads[j].right:
            # Right boundary not attached to junction
            rho = history[j]
            # print(j)
            # print(rho)
            padding = network.roads[j].pad
            gamma = network.roads[j].gamma
            max_dens = network.roads[j].max_dens

            # Go through each of the roads and calculate integral of flux out of road
            # Calculate integral as linear interpolation
            times = list(rho.keys())
            for i in range(1, len(times)):
                gamma1 = network.roads[j].get_gamma(times[i])
                gamma2 = network.roads[j].get_gamma(times[i-1])

                dt = times[i] - times[i-1]
                right = fv.flux(rho[times[i]][-padding-1], gamma1) * max_dens
                left = fv.flux(rho[times[i-1]][-padding-1], gamma2) * max_dens

                total_out += dt*(right + left) / 2
    return total_out


def total_travel_flux_out(history, network):
    '''
    '''
    pass

def project(params, Lower, Upper):
    out = params.clone()
    for i in range(len(Lower)):
        out[i] = max(Lower[i], min(Upper[i], params[i]))
    return out

def project_int(params, Lower, Upper):
    out = params.clone()
    for i in range(len(Lower)):
        out[i] = int(max(Lower[i], min(Upper[i], params[i])))
    return out

def initialize_road_network(T, roads, junctions, all_params, limiter = "minmod"):
    # T, roads, junctions = read_json(filename)

    loaded_roads = []
    loaded_junctions = []
    index = 0
    for l, r in enumerate(roads):
        n = len(r["Vmax"])
        if "max_dens" in r:
            loaded_roads.append(rd.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=[all_params[i] for i in range(index, index+n)],
                    control_points=r["ControlPoints"],
                    scheme=r['Scheme'], limiter=limiter,
                    initial = r["Init_distr"], inflow = r["Inflow"],
                    max_dens=r["max_dens"],
                    left_pos=r['left_pos'], right_pos=r['right_pos'],
                    flux_in=r["f_in"]))
        else:
            loaded_roads.append(rd.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=[all_params[i] for i in range(index, index+n)],
                        control_points=r["ControlPoints"],
                        scheme=r['Scheme'], limiter=limiter,
                        initial = r["Init_distr"], inflow = r["Inflow"],
                        left_pos=r['left_pos'], right_pos=r['right_pos'],
                        flux_in=r["f_in"]))
        index += n

    
    for j in junctions:
        loaded_traffic_lights = []
        for light in j["trafficlights"]:
            # Go through all traffic lights in junction
            if light["StartingState"] == "Green":
                start = False
            else:
                start = True
            n = len(light["Cycle"])
            loaded_traffic_lights.append(tl.TrafficLightContinous(start, light["Entering"],
                                        light["Leaving"], [all_params[i] for i in range(index, index+n)]))
            index += n
        
        loaded_coupled = []
        for light in j["coupled"]:
            ##########################################
            # Need to figure out which index is in question
            ##########################################


            # Go through all coupled traffic lights in junction
            if light["StartingState"] == "Green":
                start = True
            else:
                start = False
            n = len(light["Cycle"])
            loaded_coupled.append(tl.CoupledTrafficLightContinuous(start, light["aEntering"],
                                    light["aLeaving"], light["bEntering"],
                                    light["bLeaving"], [all_params[i] for i in range(index, index+n)]))
            index += n
            
        loaded_junctions.append(jn.Junction(roads = [loaded_roads[i] for i in j["j_roads"]],
                                        entering = j["entering"], leaving = j["leaving"],
                                        distribution=j["distribution"], 
                                        trafficlights=loaded_traffic_lights,
                                        coupled_trafficlights=loaded_coupled))
        
    network = rn.RoadNetwork(loaded_roads, loaded_junctions, T)

    return loaded_roads, loaded_junctions, network


def first_step(T, roads, junctions, all_params, objective_type, limiter = "minmod"):
    '''
    First step of optimization approach
    '''
    _, _, network = initialize_road_network(T, roads, junctions, all_params, limiter)
    # Solve conservation law
    history, queues = network.solve_cons_law()

    # Calculate objective function to minimize
    match objective_type:
        case 0:
            objective = out_flux(history, network)
        case 1:
            objective = point_flux(history, network)
        case 2:
            objective = total_travel_time(history, network)
        case 3:
            objective = exit_flux(history, network)
        case 4:
            objective = total_travel_time_queue(history, queues, network)
    
    # Storing parameters in a list
    params = []
    for road in network.roads:
        # Going through each road
        for v in road.Vmax:
            # Going through each speed for each road
            params.append(v)

    for j in network.junctions:
        for light in j.trafficlights:
            for c in light.cycle:
                params.append(c)

        for light in j.coupled_trafficlights:
            # Going through all coupled traffic lights
            for c in light.cycle:
                # Going through all parameters in the cycle
                params.append(c)

    # Calculating gradient wrt parameters
    grad = jacobi(objective, params)
    print(f"Gradient : {grad}")

    # Doesn't change parameters so no need to return parameters
    return grad.detach(), objective.detach()


def gradient_step(prev_grad, prev_all_params, upper_bounds, lower_bounds,
                  prev_objective, tol, T, roads, junctions, objective_type,
                  c1, max_step=10, limiter = "minmod"):
    '''
    Performs gradient step with armijo line search
    '''
    gradient_step_taken = False
    # Set elements of gradient to 0 if boundary already reached
    for i in range(len(prev_grad)):
        if prev_all_params[i] >= upper_bounds[i] and prev_grad[i] < 0:
            prev_grad[i] = 0
        elif prev_all_params[i] <= lower_bounds[i] and prev_grad[i] > 0:
            prev_grad[i] = 0

    if torch.norm(prev_grad) == 0:
        # Cannot update any more
        print("Cannot do any updating")

        new_all_params = prev_all_params
        new_grad = prev_grad
        new_objective = prev_objective
        gradient_step_taken = True
    else:
        # Some parameters left that can be optimized
        # First guess on step length
        alpha = max_step / max([abs(g) for g in prev_grad])
        new_objective = None
        new_all_params = None
        new_grad = None

    armijo_fails = 0
    while not gradient_step_taken:
        # Perform untill maximum number of backtracks raeached or untill armijo condition is satisfied
        new_all_params = project(prev_all_params - alpha * prev_grad,
                                 lower_bounds, upper_bounds)
        
        _, _, network = initialize_road_network(T, roads, junctions, new_all_params, limiter)
        history, queues = network.solve_cons_law()
        match objective_type:
            case 0:
                new_objective = out_flux(history, network)
            case 1:
                new_objective = point_flux(history,network)
            case 2:
                new_objective = total_travel_time(history, network)
            case 3:
                new_objective = exit_flux(history, network)
            case 4:
                new_objective = total_travel_time_queue(history, queues, network)

        params = []
        for road in network.roads:
            for v in road.Vmax:
                params.append(v)

        for j in network.junctions:
            for light in j.trafficlights:
                for c in light.cycle:
                    params.append(c)

            for light in j.coupled_trafficlights:
                # Going through all coupled traffic lights
                for c in light.cycle:
                    # Going through all parameters in the cycle
                    params.append(c)

        new_grad = jacobi(new_objective, params)
        print(f"New gradient : {new_grad}")

        # Check wolfe-armijo conditions
        i_satisfied = new_objective <= prev_objective - alpha*c1 * torch.dot(prev_grad, prev_grad)
        
        if i_satisfied:
            armijo_fails = 0
            gradient_step_taken = True
        else:
            armijo_fails += 1
            print("Armijo failed")
            print(f"Previous gradient: {prev_grad} new gradient: {new_grad}")
            print(f"Previous objective: {prev_objective}, new objective {new_objective}")
            print(f"Previous parameters: {prev_all_params} New speeds: {new_all_params}")

            alpha = 0.5 * alpha

            if armijo_fails > 5:
                # Step size has become too small, stop iteration
                print("Armijo failed more than 5 times!")
                print("Should probably stop iterating now.")
                gradient_step_taken = True
                new_all_params = prev_all_params
                new_grad = prev_grad
                new_objective = prev_objective
    
    # Gradient step is performed
    return new_all_params, new_grad, new_objective

def gradient_step_integer(prev_grad, prev_all_params, upper_bounds, lower_bounds,
                  prev_objective, tol, T, roads, junctions, objective_type,
                  c1, max_step=10, limiter = "minmod"):
    '''
    Performs gradient step with armijo line search
    '''
    gradient_step_taken = False
    # Set elements of gradient to 0 if boundary already reached
    for i in range(len(prev_grad)):
        if prev_all_params[i] >= upper_bounds[i] and prev_grad[i] < 0:
            prev_grad[i] = 0
        elif prev_all_params[i] <= lower_bounds[i] and prev_grad[i] > 0:
            prev_grad[i] = 0

    if torch.norm(prev_grad) == 0:
        # Cannot update any more
        print("Cannot do any updating")

        new_all_params = prev_all_params
        new_grad = prev_grad
        new_objective = prev_objective
        gradient_step_taken = True
    else:
        # Some parameters left that can be optimized
        # First guess on step length
        alpha = max_step / max([abs(g) for g in prev_grad])
        new_objective = None
        new_all_params = None
        new_grad = None

    armijo_fails = 0
    while not gradient_step_taken:
        # Perform untill maximum number of backtracks raeached or untill armijo condition is satisfied
        new_all_params = project_int(prev_all_params - alpha * prev_grad,
                                 lower_bounds, upper_bounds)
        
        _, _, network = initialize_road_network(T, roads, junctions, new_all_params, limiter)
        history, queues = network.solve_cons_law()
        match objective_type:
            case 0:
                new_objective = out_flux(history, network)
            case 1:
                new_objective = point_flux(history,network)
            case 2:
                new_objective = total_travel_time(history, network)
            case 3:
                new_objective = exit_flux(history, network)
            case 4:
                new_objective = total_travel_time_queue(history, queues, network)

        params = []
        for road in network.roads:
            for v in road.Vmax:
                params.append(v)

        for j in network.junctions:
            for light in j.trafficlights:
                for c in light.cycle:
                    params.append(c)

            for light in j.coupled_trafficlights:
                # Going through all coupled traffic lights
                for c in light.cycle:
                    # Going through all parameters in the cycle
                    params.append(c)

        new_grad = jacobi(new_objective, params)
        print(f"New gradient : {new_grad}")

        # Check wolfe-armijo conditions
        i_satisfied = new_objective <= prev_objective - alpha*c1 * torch.dot(prev_grad, prev_grad)
        
        if i_satisfied:
            armijo_fails = 0
            gradient_step_taken = True
        else:
            armijo_fails += 1
            print("Armijo failed")
            print(f"Previous gradient: {prev_grad} new gradient: {new_grad}")
            print(f"Previous objective: {prev_objective}, new objective {new_objective}")
            print(f"Previous parameters: {prev_all_params} New speeds: {new_all_params}")

            alpha = 0.5 * alpha

            if armijo_fails >= 5:
                # Step size has become too small, stop iteration
                print("Armijo failed 5 times!")
                print("Should probably stop iterating now.")
                gradient_step_taken = True
                new_all_params = prev_all_params
                new_grad = prev_grad
                new_objective = prev_objective
    
    # Gradient step is performed
    return new_all_params, new_grad, new_objective

def gradient_step_integer2(prev_grad, prev_all_params, upper_bounds, lower_bounds,
                  prev_objective, tol, T, roads, junctions, objective_type,
                  c1, max_step=10, limiter = "minmod", armijo_step = 0):
    '''
    Performs gradient step with armijo line search
    '''
    gradient_step_taken = False
    # Set elements of gradient to 0 if boundary already reached
    for i in range(len(prev_grad)):
        if prev_all_params[i] >= upper_bounds[i] and prev_grad[i] < 0:
            prev_grad[i] = 0
        elif prev_all_params[i] <= lower_bounds[i] and prev_grad[i] > 0:
            prev_grad[i] = 0

    if torch.norm(prev_grad) == 0:
        # Cannot update any more
        print("Cannot do any updating")

        new_all_params = prev_all_params
        new_grad = prev_grad
        new_objective = prev_objective
        return new_all_params, new_grad, new_objective, True
    else:
        # Some parameters left that can be optimized
        # First guess on step length
        alpha = max_step / max([abs(g) for g in prev_grad])
        new_objective = None
        new_all_params = None
        new_grad = None

    # Perform untill maximum number of backtracks raeached or untill armijo condition is satisfied
    alpha = 0.5**(armijo_step) * alpha
    new_all_params = project_int(prev_all_params - alpha * prev_grad,
                                lower_bounds, upper_bounds)
    

    _, _, network = initialize_road_network(T, roads, junctions, new_all_params, limiter)
    history, queues = network.solve_cons_law()
    match objective_type:
        case 0:
            new_objective = out_flux(history, network)
        case 1:
            new_objective = point_flux(history,network)
        case 2:
            new_objective = total_travel_time(history, network)
        case 3:
            new_objective = exit_flux(history, network)
        case 4:
            new_objective = total_travel_time_queue(history, queues, network)

    params = []
    for road in network.roads:
        for v in road.Vmax:
            params.append(v)

    for j in network.junctions:
        for light in j.trafficlights:
            for c in light.cycle:
                params.append(c)

        for light in j.coupled_trafficlights:
            # Going through all coupled traffic lights
            for c in light.cycle:
                # Going through all parameters in the cycle
                params.append(c)

    new_grad = jacobi(new_objective, params)
    print(f"New gradient : {new_grad}")

    # Check wolfe-armijo conditions
    i_satisfied = new_objective <= prev_objective - alpha*c1 * torch.dot(prev_grad, prev_grad)
    
    if not i_satisfied:
        print("Armijo failed")
        print(f"Previous gradient: {prev_grad} new gradient: {new_grad}")
        print(f"Previous objective: {prev_objective}, new objective {new_objective}")
        print(f"Previous parameters: {prev_all_params} New speeds: {new_all_params}")
        return None, None, None, False
    
    # Gradient step and armijo condition is satisfied
    return new_all_params, new_grad, new_objective, True

    
def optimize_parameters(filename, objective_type, tol=0.1, maxiter=10, c1 = 1e-4, c2=0.9, limiter = "minmod",
                        light_lower = 10, light_upper = 120, max_step = 10):
    '''
    Same procedure as optimize gradient procedure

    Difference is that it is split into more functions so that memory will be reallocated
    '''

    ################################################
    # STEP 1: Loading in file and storing parameters
    ################################################
    
    print('\n######################################')
    print(f"Trying to optimize the parameters in the road network specified in {filename}.")
    print('######################################\n')


    T, roads, junctions = load.read_json(filename)

    Vmax = []
    light_times = []
    Vlower = []
    VUpper = []
    for r in roads:
        for i in range(len(r['Vmax'])):
            Vmax.append(r['Vmax'][i])
            Vlower.append(r['low'][i])
            VUpper.append(r['high'][i])

    lightUpper = []
    lightLower = []
    for j in junctions:
        for light in j["trafficlights"]:
            for c in light["Cycle"]:
                lightUpper.append(light_upper)
                lightLower.append(light_lower)
                light_times.append(c)

        for coupled in j["coupled"]:
            for c in coupled["Cycle"]:
                lightUpper.append(light_upper)
                lightLower.append(light_lower)
                light_times.append(c)

    all_params = Vmax + light_times
    all_params = torch.tensor(all_params)
    upper_bounds = VUpper + lightUpper
    lower_bounds = Vlower + lightLower

    # Do first iteration
    # First parameter list equal to all_params
    prev_grad, prev_objective = first_step(T, roads, junctions, all_params, objective_type, limiter)
    prev_all_params = all_params

    print('\n######################################')
    print("First iteration:")
    print(f"All parameters: {all_params}")
    print(f"Objective value: {prev_objective}")
    print(f"Gradient: {prev_grad}")
    print('######################################\n')

    error = tol + 1 # Chosen so that while loop will be entered
    curr_iter = 0
    total_params = [prev_all_params.detach()]
    all_objectives = [prev_objective.detach()]
    all_grads = [prev_grad.detach()]


    while curr_iter < maxiter and error > tol:
        # Repeat optimization loop until error reached or until max number of iterations reached
        # Don't need to send tol as argument to gradient_step...
        new_all_params, new_grad, new_objective = gradient_step(prev_grad, prev_all_params, upper_bounds, 
                                                                lower_bounds, prev_objective, tol, T, 
                                                                roads, junctions, objective_type,
                                                                c1, max_step=max_step, limiter = limiter)
        
        print('######################################')
        print(f"Iteration {curr_iter+1}:")
        print(f"All parameters: {new_all_params}")
        print(f"Objective value: {new_objective}")
        print(f"Gradient: {new_grad}")

        curr_iter += 1
        total_params.append(new_all_params.detach())
        all_objectives.append(new_objective.detach())
        all_grads.append(new_grad.detach())
        error = torch.norm((total_params[curr_iter] - total_params[curr_iter - 1]) / total_params[0])
        print(f"Current error: {error}")
        print('######################################\n')

        prev_all_params = new_all_params.detach()
        prev_grad = new_grad.detach()
        prev_objective = new_objective.detach()

    # At this point either maximum number of iterations or the maximum error is reached
    print("Stopping criteria reached")
    return total_params, all_objectives, all_grads


def optimize_parameters_integers(filename, objective_type, tol=0.1, maxiter=10, c1 = 1e-4, c2=0.9, limiter = "minmod",
                                 light_lower = 10, light_upper = 120, max_step = 10):
    '''
    Same procedure as optimize gradient procedure

    Difference is that it is split into more functions so that memory will be reallocated

    Very similar to above procedure, but parameters are forced to be integers at each step
    '''

    ################################################
    # STEP 1: Loading in file and storing parameters
    ################################################
    
    print('\n######################################')
    print(f"Trying to optimize the parameters in the road network specified in {filename}.")
    print('######################################\n')


    T, roads, junctions = load.read_json(filename)

    Vmax = []
    light_times = []
    Vlower = []
    VUpper = []
    for r in roads:
        for i in range(len(r['Vmax'])):
            Vmax.append(r['Vmax'][i])
            Vlower.append(r['low'][i])
            VUpper.append(r['high'][i])

    lightUpper = []
    lightLower = []
    for j in junctions:
        for light in j["trafficlights"]:
            for c in light["Cycle"]:
                lightUpper.append(light_upper)
                lightLower.append(light_lower)
                light_times.append(c)

        for coupled in j["coupled"]:
            for c in coupled["Cycle"]:
                lightUpper.append(light_upper)
                lightLower.append(light_lower)
                light_times.append(c)

    all_params = Vmax + light_times
    all_params = torch.tensor(all_params)
    upper_bounds = VUpper + lightUpper
    lower_bounds = Vlower + lightLower

    # Do first iteration
    # First parameter list equal to all_params
    prev_grad, prev_objective = first_step(T, roads, junctions, all_params, objective_type, limiter)
    prev_all_params = all_params

    print('\n######################################')
    print("First iteration:")
    print(f"All parameters: {all_params}")
    print(f"Objective value: {prev_objective}")
    print(f"Gradient: {prev_grad}")
    print('######################################\n')

    error = tol + 1 # Chosen so that while loop will be entered
    curr_iter = 0
    total_params = [prev_all_params.detach()]
    all_objectives = [prev_objective.detach()]
    all_grads = [prev_grad.detach()]


    while curr_iter < maxiter and error > tol:
        # Repeat optimization loop until error reached or until max number of iterations reached
        # Don't need to send tol as argument to gradient_step...
        new_all_params, new_grad, new_objective = gradient_step_integer(prev_grad, prev_all_params, upper_bounds, 
                                                                lower_bounds, prev_objective, tol, T, 
                                                                roads, junctions, objective_type,
                                                                c1, max_step=max_step, limiter = limiter)
        
        print('######################################')
        print(f"Iteration {curr_iter+1}:")
        print(f"All parameters: {new_all_params}")
        print(f"Objective value: {new_objective}")
        print(f"Gradient: {new_grad}")

        curr_iter += 1
        total_params.append(new_all_params.detach())
        all_objectives.append(new_objective.detach())
        all_grads.append(new_grad.detach())
        error = torch.norm((total_params[curr_iter] - total_params[curr_iter - 1]) / total_params[0])
        print(f"Current error: {error}")
        print('######################################\n')

        prev_all_params = new_all_params.detach()
        prev_grad = new_grad.detach()
        prev_objective = new_objective.detach()

    # At this point either maximum number of iterations or the maximum error is reached
    print("Stopping criteria reached")
    return total_params, all_objectives, all_grads

def optimize_parameters_integers2(filename, objective_type, tol=0.1, maxiter=10, c1 = 1e-4, c2=0.9, limiter = "minmod",
                                 light_lower = 10, light_upper = 120, max_step = 10):
    '''
    Same procedure as optimize gradient procedure

    Difference is that it is split into more functions so that memory will be reallocated

    Very similar to above procedure, but parameters are forced to be integers at each step
    '''

    ################################################
    # STEP 1: Loading in file and storing parameters
    ################################################
    
    print('\n######################################')
    print(f"Trying to optimize the parameters in the road network specified in {filename}.")
    print('######################################\n')


    T, roads, junctions = load.read_json(filename)

    Vmax = []
    light_times = []
    Vlower = []
    VUpper = []
    for r in roads:
        for i in range(len(r['Vmax'])):
            Vmax.append(r['Vmax'][i])
            Vlower.append(r['low'][i])
            VUpper.append(r['high'][i])

    lightUpper = []
    lightLower = []
    for j in junctions:
        for light in j["trafficlights"]:
            for c in light["Cycle"]:
                lightUpper.append(light_upper)
                lightLower.append(light_lower)
                light_times.append(c)

        for coupled in j["coupled"]:
            for c in coupled["Cycle"]:
                lightUpper.append(light_upper)
                lightLower.append(light_lower)
                light_times.append(c)

    all_params = Vmax + light_times
    all_params = torch.tensor(all_params)
    upper_bounds = VUpper + lightUpper
    lower_bounds = Vlower + lightLower

    # Do first iteration
    # First parameter list equal to all_params
    prev_grad, prev_objective = first_step(T, roads, junctions, all_params, objective_type, limiter)
    prev_all_params = all_params

    print('\n######################################')
    print("First iteration:")
    print(f"All parameters: {all_params}")
    print(f"Objective value: {prev_objective}")
    print(f"Gradient: {prev_grad}")
    print('######################################\n')

    error = tol + 1 # Chosen so that while loop will be entered
    curr_iter = 0
    total_params = [prev_all_params.detach()]
    all_objectives = [prev_objective.detach()]
    all_grads = [prev_grad.detach()]


    while curr_iter < maxiter and error > tol:
        # Repeat optimization loop until error reached or until max number of iterations reached
        # Don't need to send tol as argument to gradient_step...
        step_taken = False
        armijo_step = 0
        while not step_taken:
            new_all_params, new_grad, new_objective, armijo_state = gradient_step_integer2(prev_grad, prev_all_params, upper_bounds, 
                                                                    lower_bounds, prev_objective, tol, T, 
                                                                    roads, junctions, objective_type,
                                                                    c1, max_step=max_step, limiter = limiter,
                                                                    armijo_step = armijo_step)
            if armijo_state:
                step_taken = True
            else:
                armijo_step += 1
            
            if armijo_step >= 5:
                print("Armijo failed 5 times!")
                print("Should probably stop iterating now.")
                step_taken = True
                new_all_params = prev_all_params
                new_grad = prev_grad
                new_objective = prev_objective
        
        print('######################################')
        print(f"Iteration {curr_iter+1}:")
        print(f"All parameters: {new_all_params}")
        print(f"Objective value: {new_objective}")
        print(f"Gradient: {new_grad}")

        curr_iter += 1
        total_params.append(new_all_params.detach())
        all_objectives.append(new_objective.detach())
        all_grads.append(new_grad.detach())
        error = torch.norm((total_params[curr_iter] - total_params[curr_iter - 1]) / total_params[0])
        print(f"Current error: {error}")
        print('######################################\n')

        prev_all_params = new_all_params.detach()
        prev_grad = new_grad.detach()
        prev_objective = new_objective.detach()

    # At this point either maximum number of iterations or the maximum error is reached
    print("Stopping criteria reached")
    return total_params, all_objectives, all_grads


# def dummy_func():
    # speed_times, objectives, final_speeds = optimize_gradient_armijo2('networks/2-2_coupledlights.json',4, tol=0.01, maxiter=3)

if __name__ == "__main__":
    ##############################################
    # TODO: Add method for saving results to json
    ##############################################

    variation = 10
    match variation:
        case 0:
            # Make optimize_parameters able to take in different staring point
            # 1-1 road network
            # Optimizing the control parameters of the traffic light
            all_params, objectives, grads = optimize_parameters_integers2('networks/1-1trafficLight.json',4, tol=0.01, maxiter=20)
            print("All parameters:")
            print(all_params)

            print("Objectives:")
            print(objectives)

            print("Gradients:")
            print(grads)

        case 1:
            # Optimization approach where parameters are forced to be integers at each step
            all_params, objectives, grads = optimize_parameters_integers('networks/1-1trafficLight.json',4, tol=0.01, maxiter=20)
            print("All parameters:")
            print(all_params)

            print("Objectives:")
            print(objectives)

            print("Gradients:")
            print(grads)

            print("All parameters:")
            print(all_params)

        case 2:
            # Optimize parameters in a junction with 2 incoming roads and 2 outgoing roads
            all_params, objectives, grads = optimize_parameters('networks/2-2_coupledlights.json',4, tol=0.01, maxiter=20)

            print("Objectives:")
            print(objectives)

            print("Gradients:")
            print(grads)

            print("All parameters:")
            print(all_params)

        case 3:
            # Optimize parameters in a junction with 2 incoming roads and 2 outgoing roads
            all_params, objectives, grads = optimize_parameters('networks/1-1.json',4, tol=0.01, maxiter=20, 
                                                                         light_lower = 100, light_upper = 100)

            print("Objectives:")
            print(objectives)

            print("Gradients:")
            print(grads)

            print("All parameters:")
            print(all_params)

        case 4:
            # Sometimes continuous optimization works best, sometimes integer optimization works best
            # Optimize parameters in a junction with 2 incoming roads and 2 outgoing roads
            all_params, objectives, grads = optimize_parameters_integers('networks/1-1_variable.json',4, tol=0.01, maxiter=20,
                                                                         light_lower = 100, light_upper = 100, max_step = 2)

            print("Objectives:")
            print(objectives)

            print("Gradients:")
            print(grads)

            print("All parameters:")
            print(all_params)
        
        case 5:
            # Sometimes continuous optimization works best, sometimes integer optimization works best
            # Optimize parameters in a junction with 2 incoming roads and 2 outgoing roads
            all_params, objectives, grads = optimize_parameters('networks/1-1_full.json',4, tol=0.001, maxiter=40,
                                                                         light_lower = 10, light_upper = 120, max_step = 5)

            print("Objectives:")
            print(objectives)

            print("Gradients:")
            print(grads)

            print("All parameters:")
            print(all_params)

        case 6:
            T, roads, junctions = load.read_json('networks/1-1.json')
            Vmax = []
            light_times = []
            # Vlower = []
            # VUpper = []
            for r in roads:
                for i in range(len(r['Vmax'])):
                    Vmax.append(r['Vmax'][i])
                    # Vlower.append(r['low'][i])
                    # VUpper.append(r['high'][i])

            # lightUpper = []
            # lightLower = []
            for j in junctions:
                for light in j["trafficlights"]:
                    for c in light["Cycle"]:
                        # lightUpper.append(120.0)
                        # lightLower.append(10.0)
                        light_times.append(c)

                for coupled in j["coupled"]:
                    for c in coupled["Cycle"]:
                        # lightUpper.append(120.0)
                        # lightLower.append(10.0)
                        light_times.append(c)

            all_params = Vmax + light_times
            all_params = torch.tensor(all_params)

            _, _, network = initialize_road_network(T, roads, junctions, all_params, "minmod")
            # Solve conservation law
            history, queues = network.solve_cons_law()

            print(queues[0])
            # print(history)
        
        case 7:
            # Optimize parameters in a junction with 2 incoming roads and 2 outgoing roads
            all_params, objectives, grads = optimize_parameters_integers('networks/2-2coupled.json',4, tol=0.001, maxiter=40,
                                                                         light_lower = 10, light_upper = 120, max_step = 10)

            print("Objectives:")
            print(objectives)

            print("Gradients:")
            print(grads)

            print("All parameters:")
            print(all_params)
        case 8:
            T, roads, junctions = load.read_json('networks/2-2coupled.json')
            Vmax = []
            light_times = []
            # Vlower = []
            # VUpper = []
            for r in roads:
                for i in range(len(r['Vmax'])):
                    Vmax.append(r['Vmax'][i])
                    # Vlower.append(r['low'][i])
                    # VUpper.append(r['high'][i])

            # lightUpper = []
            # lightLower = []
            for j in junctions:
                for light in j["trafficlights"]:
                    for c in light["Cycle"]:
                        # lightUpper.append(120.0)
                        # lightLower.append(10.0)
                        light_times.append(c)

                for coupled in j["coupled"]:
                    for c in coupled["Cycle"]:
                        # lightUpper.append(120.0)
                        # lightLower.append(10.0)
                        light_times.append(c)

            all_params = Vmax + light_times
            all_params = torch.tensor(all_params)

            _, _, network = initialize_road_network(T, roads, junctions, all_params, "minmod")
            # Solve conservation law
            history, queues = network.solve_cons_law()

            print(queues[0])
            print("\n##############################################################\n")
            print(queues[2])

        case 9:
             # Optimize parameters in a junction with 2 incoming roads and 2 outgoing roads
            all_params, objectives, grads = optimize_parameters_integers('networks/2-2coupled.json',4, tol=0.001, maxiter=40,
                                                                         light_lower = 10, light_upper = 120, max_step = 10)

            print("Objectives:")
            print(objectives)

            print("Gradients:")
            print(grads)

            print("All parameters:")
            print(all_params)

        case 10:
             # Optimize parameters in a junction with 2 incoming roads and 2 outgoing roads
            all_params, objectives, grads = optimize_parameters_integers2('networks/complex4.json',4, tol=0.001, maxiter=40,
                                                                         light_lower = 10, light_upper = 120, max_step = 10)

            print("Objectives:")
            print(objectives)

            print("Gradients:")
            print(grads)

            print("All parameters:")
            print(all_params)
# Maybe try to be even more clever when choosing starting point
# F.ex. if armijo failed once (or maybe more), try next guess to be smaller than 10
# I.e. start by setting maximum increase to 10 (20?). If armijo fails try 5 (10?). Maybe try even 
# lower, but maybe okay to stop there
# If iteration goes through, then increase step size back to 10 (20?). 
# 2.3449


'''
1. Case: 
Both ended up having queus... bad
Coupled light cycle:
    120, 20

Road 1: 
    Speed limit: 80
    Init density: 0.1
    Inflow: 0.4
    End queue length: 1.6877

Road 2:
    Speed limit: 50
    Init density: 0.1
    Inflow: 0.1
    End queue length: 2.4054


2. Case: 
No queus - not desired either

Road 1: 
    Speed limit: 80
    Init density: 0.1
    Inflow: 0.2
    End queue length: 0

Road 2:
    Speed limit: 50
    Init density: 0.05
    Inflow: 0.05
    End queue length: 0

2. Case: 
No queus - not desired either

Road 1: 
    Speed limit: 80
    Init density: 0.1
    Inflow: 0.2
    End queue length: 0.4017

Road 2:
    Speed limit: 50
    Init density: 0.05
    Inflow: 0.05
    End queue length: 0.3651


0.3843 and 0 with cycle 100, 100

'''