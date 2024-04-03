#################################################################
# Problem: Running out of memory if many iterations are performed
# Need to release memory between iterations
# One fix -> instead of creating new networks, just reset densities
# and update parameters

# Multiple searches
# Eliminate parts of domain already visited naively/greedily
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

    
def gradient_descent_step():
    pass

def project(speeds, Lower, Upper):
    out = torch.zeros_like(speeds)
    for i in range(len(out)):
        out[i] = max(Lower[i], min(Upper[i], speeds[i]))

    return out

def project2(params, Lower, Upper):
    out = params.clone()
    for i in range(len(Lower)):
        out[i] = max(Lower[i], min(Upper[i], params[i]))
    return out


def initialize_road_network(T, roads, junctions, vmaxes):
    # T, roads, junctions = read_json(filename)

    loaded_roads = []
    loaded_junctions = []
    index = 0
    for l, r in enumerate(roads):
        n = len(r["Vmax"])
        if "max_dens" in r:
            loaded_roads.append(rd.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=[vmaxes[i] for i in range(index, index+n)],
                        control_points=r["ControlPoints"],
                        scheme=r['Scheme'], limiter="minmod",
                        initial = r["Init_distr"], inflow = r["Inflow"],
                        max_dens=r["max_dens"]))
        else:
            loaded_roads.append(rd.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=[vmaxes[i] for i in range(index, index+n)],
                        control_points=r["ControlPoints"],
                        scheme=r['Scheme'], limiter="minmod",
                        initial = r["Init_distr"], inflow = r["Inflow"]))
        index += n

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
                start = False
            else:
                start = True
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


def optimize_gradient_armijo(filename, objective_type, tol = 0.1, maxiter=10, c1 = 1e-4, c2=0.9):
    '''
    Take in some configuration of traffic network organized in a json file
    This configuration should include roads with their initial funcitons, length, 
    starting guess for speed limits, scheme to be used and initial funcitons
    It should include all junctions, and distribution in junction, as well as any traffic lights
    Should include number of steps in x-direction either for each road or same for all roads
    The total time of the simulation should also be included
    
    Junctions should remain fixed for each iteration of scheme
    Roads should mostly stay the same, with the exception of initial speed limit
    Network will be created in same way for each iteration

    Goal is to minimize some objective function

    ----------------------------------------------------------
    Approach:

    1.  Initialize all roads, Junctions and Network
    2.  Solve the conservation law for this road network until time T given in config file
    3.  Calculate derivatives of all parameters (first only speed limits, and later for traffic
        lights as well).
    4.  Get new guess for parameters, using backtracking gradient ascent
        4.1 For the steplength, the first guess will be set so that the speed limit that 
            changes the most will change with 5km/t. Then wolfe-armijo condition to check it is 
            okay.
    5.  Go back to step 1, with new parameters - repeat untill "close enough" to optimal solution
        Note that this solution might only be locally optimal

    ----------------------------------------------------------


    This function supports a number of different objective functions
    objective_type = 0  ->  Calculate the time interval over all times
                            of the flux out of each road
    objective_type = 1  ->  Calculate the time interval over some small time interval
                            of the flux out of each road
    '''

    # Load in starting point from json file
    T, roads, junctions = load.read_json(filename)

    # Storing starting speeds and bounds on speeds
    Vmax = []
    Vlower = []
    VUpper = []
    for r in roads:
        for i in range(len(r['Vmax'])):
            Vmax.append(r['Vmax'][i])
            Vlower.append(r['low'][i])
            VUpper.append(r['high'][i])

    Vmax = torch.tensor(Vmax)
    print(f"Starting speed limits {Vmax}")
    print(f"Upper limits {VUpper}")
    print(f"Lower limits {Vlower}")

    # Vmax = torch.tensor(Vmax)

    error = tol + 1
    curr_iter = 0
    speeds = []
    objectives = []
    while error > tol and curr_iter < maxiter:
        # Perform gradient descent step until tolerance reached
        if curr_iter == 0:
            # Initilaizing variables

            # Initialize system and 
            loaded_roads, loaded_junctions, network = initialize_road_network(T, roads, junctions, Vmax)
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
                


            # Storing parameters in a list
            params = []
            for road in network.roads:
                # Going through each road
                for v in road.Vmax:
                    # Going through each speed for each road
                    params.append(v)

            # Calculating gradient wrt parameters
            grad = jacobi(objective, params)
            print(f"Gradient : {grad}")

            speeds.append(torch.tensor([v.detach() for v in Vmax]))

            objectives.append(objective.detach())


        print(f"Iteration {curr_iter+1}")
        print(f"Current objective value: {objective}")

        

        # Performing gradient descent step
        step_taken = False
        # First guess of steplength so that speed limit changes by 5
        for i in range(len(grad)):
            if Vmax[i] >= VUpper[i] and grad[i] < 0:
                # Cannot update this speed with 5 km/h
                grad[i] = 0

            elif Vmax[i] <= Vlower[i] and grad[i] > 0:
                # Cannot update this speed with 5 km/h
                grad[i] = 0
        
        print(Vmax)
        if torch.norm(grad) == 0:
            # Cannot do any updating
            print("Cannot do any updating")

            new_Vmax = Vmax
            new_history = history
            new_grad = grad
            new_objective = objective
            new_loaded_roads, new_loaded_junctions, new_network = loaded_roads, loaded_junctions, network
            step_taken = True

        alpha = 10 / max([abs(g) for g in grad])
        print(alpha * grad)

        new_loaded_roads, new_loaded_junctions, new_network = None, None, None
        new_history, new_objective = None, None
        new_params, new_grad = None, None

        armijo_fails = 0
        while not step_taken:
            # Check if steplength is okay
            new_Vmax = Vmax - alpha * grad
            #new_Vmax = project(new_Vmax, Vlower, VUpper)
            if torch.norm(new_Vmax - Vmax) < tol:
                step_taken = True
            new_Vmax = project(new_Vmax, Vlower, VUpper)
            # Need to do simulation at next iterate:

            # Initialize new road system, calculate objective function and gradient
            new_loaded_roads, new_loaded_junctions, new_network = initialize_road_network(T, roads, junctions, new_Vmax)
            new_history, new_queues = new_network.solve_cons_law()

            match objective_type:
                case 0:
                    new_objective = out_flux(new_history, new_network)
                case 1:
                    new_objective = point_flux(new_history,new_network)
                case 2:
                    new_objective = total_travel_time(new_history, new_network)
            
            new_params = []
            for road in new_network.roads:
                for v in road.Vmax:
                    new_params.append(v)
            new_grad = jacobi(new_objective, new_params)
            print(f"Gradient : {new_grad}")

            # Check wolfe-armijo conditions
            i_satisfied = new_objective <= objective - alpha*c1 * torch.dot(grad, grad) # Grad*grad should be norm of grad
            # ii not needed since first guess is taken to be rather large
            #ii_satisfied = True # inner product of gradients <= c2*norm of old gradient

            if i_satisfied: #and ii_satisfied:
                armijo_fails = 0
                # Perform step
                step_taken = True
            else:
                armijo_fails += 1
                print("Armijo failed")
                print(f"Previous gradient: {grad} new gradient: {new_grad}")
                print(f"Previous objective: {objective}, new objective {new_objective}")
                print(f"Previous speeds: {Vmax} New speeds: {new_Vmax}")
                alpha = 0.5 * alpha

                if armijo_fails >= 5:
                    # Step size has become too small, stop iteration
                    print("Armijo failed 5 times!")
                    print("Should probably stop iterating now.")
                    step_taken = True
                    new_Vmax = Vmax
                    new_history = history
                    new_grad = grad
                    new_objective = objective
                    new_loaded_roads, new_loaded_junctions, new_network = loaded_roads, loaded_junctions, network
                    step_taken = True

        # Gradient descent step taken
        # Update variables
        Vmax = project(new_Vmax, Vlower, VUpper) # Project onto bounding rectangle
        loaded_roads, loaded_junctions, network = new_loaded_roads, new_loaded_junctions, new_network
        history = new_history
        grad = new_grad
        objective = new_objective

        speeds.append(Vmax.detach())
        objectives.append(objective.detach())

        curr_iter += 1

        error = torch.norm(speeds[curr_iter] - speeds[curr_iter-1]) / torch.norm(speeds[1])
        print(error)
    final_speeds = []
    for s in speeds[-1]:
        final_speeds.append(int(s))

    return speeds, objectives, final_speeds



def initialize_road_network2(T, roads, junctions, all_params, limiter="minmod"):
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
                    left_pos=r['left_pos'], right_pos=r['right_pos']))
        else:
            loaded_roads.append(rd.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=[all_params[i] for i in range(index, index+n)],
                        control_points=r["ControlPoints"],
                        scheme=r['Scheme'], limiter=limiter,
                        initial = r["Init_distr"], inflow = r["Inflow"],
                        left_pos=r['left_pos'], right_pos=r['right_pos']))
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

def optimize_gradient_armijo2(filename, objective_type, tol = 0.1, maxiter=10, c1 = 1e-4, c2=0.9, limiter="minmod"):
    '''
    Take in some configuration of traffic network organized in a json file
    This configuration should include roads with their initial funcitons, length, 
    starting guess for speed limits, scheme to be used and initial funcitons
    It should include all junctions, and distribution in junction, as well as any traffic lights
    Should include number of steps in x-direction either for each road or same for all roads
    The total time of the simulation should also be included
    
    Junctions should remain fixed for each iteration of scheme
    Roads should mostly stay the same, with the exception of initial speed limit
    Network will be created in same way for each iteration

    Goal is to minimize some objective function

    ----------------------------------------------------------
    Approach:

    1.  Initialize all roads, Junctions and Network
    2.  Solve the conservation law for this road network until time T given in config file
    3.  Calculate derivatives of all parameters (first only speed limits, and later for traffic
        lights as well).
    4.  Get new guess for parameters, using backtracking gradient ascent
        4.1 For the steplength, the first guess will be set so that the speed limit that 
            changes the most will change with 5km/t. Then wolfe-armijo condition to check it is 
            okay.
    5.  Go back to step 1, with new parameters - repeat untill "close enough" to optimal solution
        Note that this solution might only be locally optimal

    ----------------------------------------------------------


    This function supports a number of different objective functions
    objective_type = 0  ->  Calculate the time interval over all times
                            of the flux out of each road
    objective_type = 1  ->  Calculate the time interval over some small time interval
                            of the flux out of each road
    '''

    # Load in starting point from json file
    T, roads, junctions = load.read_json(filename)

    # Storing starting speeds and bounds on speeds
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
                lightUpper.append(100.0)
                lightLower.append(10.0)
                light_times.append(c)

        for coupled in j["coupled"]:
            for c in coupled["Cycle"]:
                lightUpper.append(100.0)
                lightLower.append(10.0)
                light_times.append(c)

    all_params = Vmax + light_times
    all_params = torch.tensor(all_params)
    upperBounds = VUpper + lightUpper
    lowerBounds = Vlower + lightLower
    
    # Vmax = torch.tensor(Vmax)
    print(f"Starting speed limits {Vmax}")
    print(f"Upper limits {VUpper}")
    print(f"Lower limits {Vlower}")

    light_times = torch.tensor(light_times)
    print(f"Starting cycle times {light_times}")
    print(f"Upper limits {lightUpper}")
    print(f"Lower limits {lightLower}")

    print(f"All parameters: {all_params}")


    # Vmax = torch.tensor(Vmax)

    error = tol + 1
    curr_iter = 0
    speeds = []
    cycle_times = []
    speed_times = []

    objectives = []
    while error > tol and curr_iter < maxiter:
        # Perform gradient descent step until tolerance reached
        if curr_iter == 0:
            # Initilaizing variables

            # Initialize system and 
            loaded_roads, loaded_junctions, network = initialize_road_network2(T, roads, junctions, all_params, limiter)
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


            # speeds.append(torch.tensor([v.detach() for v in Vmax]))
            # cycle_times.append(torch.tensor([t.detach() for t in light_times]))

            speed_times.append(torch.tensor([p.detach() for p in all_params]))
            objectives.append(objective.detach())


        print(f"Iteration {curr_iter+1}")
        print(f"Current objective value: {objective}")

        

        # Performing gradient descent step
        step_taken = False
        for i in range(len(grad)):
            if all_params[i] >= upperBounds[i] and grad[i] < 0:
                grad[i] = 0
            elif all_params[i] <= lowerBounds[i] and grad[i] > 0:
                grad[i] = 0
        
        print(all_params)
        print(lowerBounds)
        print(upperBounds)
        if torch.norm(grad) == 0:
            # Cannot do any updating
            print("Cannot do any updating")

            new_all_params = all_params
            new_history = history
            new_grad = grad
            new_objective = objective
            new_loaded_roads, new_loaded_junctions, new_network = loaded_roads, loaded_junctions, network
            step_taken = True
        else:

            alpha = 10 / max([abs(g) for g in grad])
            print(alpha * grad)
            # print(alpha)
            new_loaded_roads, new_loaded_junctions, new_network = None, None, None
            new_history, new_objective = None, None
            new_params, new_grad = None, None

        armijo_fails = 0
        while not step_taken:
            # Check if steplength is okay
            new_all_params = all_params - alpha * grad
            #new_Vmax = project(new_Vmax, Vlower, VUpper)
            if torch.norm(new_all_params - all_params) < tol:
                step_taken = True
            new_all_params = project2(new_all_params, lowerBounds, upperBounds)
            # Need to do simulation at next iterate:

            # Initialize new road system, calculate objective function and gradient
            new_loaded_roads, new_loaded_junctions, new_network = initialize_road_network2(T, roads, junctions, new_all_params, limiter)
            new_history, new_queues = new_network.solve_cons_law()

            match objective_type:
                case 0:
                    new_objective = out_flux(new_history, new_network)
                case 1:
                    new_objective = point_flux(new_history,new_network)
                case 2:
                    new_objective = total_travel_time(new_history, new_network)
                case 3:
                    new_objective = exit_flux(new_history, new_network)
                case 4:
                    new_objective = total_travel_time_queue(new_history, new_queues, new_network)
            
            new_params = []
            for road in new_network.roads:
                for v in road.Vmax:
                    new_params.append(v)

            for j in new_network.junctions:
                for light in j.trafficlights:
                    for c in light.cycle:
                        new_params.append(c)

                for light in j.coupled_trafficlights:
                    # Going through all coupled traffic lights
                    for c in light.cycle:
                        # Going through all parameters in the cycle
                        new_params.append(c)

            new_grad = jacobi(new_objective, new_params)
            print(f"Gradient : {new_grad}")

            # Check wolfe-armijo conditions
            i_satisfied = new_objective <= objective - alpha*c1 * torch.dot(grad, grad) # Grad*grad should be norm of grad
            # ii not needed since first guess is taken to be rather large
            #ii_satisfied = True # inner product of gradients <= c2*norm of old gradient

            if i_satisfied: #and ii_satisfied:
                armijo_fails = 0
                # Perform step
                step_taken = True
            else:
                armijo_fails += 1
                print("Armijo failed")
                print(f"Previous gradient: {grad} new gradient: {new_grad}")
                print(f"Previous objective: {objective}, new objective {new_objective}")
                print(f"Previous parameters: {all_params} New speeds: {new_all_params}")
                
                alpha = 0.5 * alpha

                if armijo_fails >= 5:
                    # Step size has become too small, stop iteration
                    print("Armijo failed 5 times!")
                    print("Should probably stop iterating now.")
                    step_taken = True
                    new_all_params = all_params
                    new_history = history
                    new_grad = grad
                    new_objective = objective
                    new_loaded_roads, new_loaded_junctions, new_network = loaded_roads, loaded_junctions, network
                    step_taken = True

        # Gradient descent step taken
        # Update variables
        all_params = project2(new_all_params, lowerBounds, upperBounds) # Project onto bounding rectangle
        loaded_roads, loaded_junctions, network = new_loaded_roads, new_loaded_junctions, new_network
        history = new_history
        grad = new_grad
        objective = new_objective

        speed_times.append(all_params.detach())
        objectives.append(objective.detach())

        curr_iter += 1

        error = torch.norm(speed_times[curr_iter] - speed_times[curr_iter-1]) / torch.norm(speed_times[1])
        print(error)
    print("Main loop exited")
    # print(time.time())
    final_speeds = []
    for s in speed_times[-1]:
        # print(s)
        final_speeds.append(torch.round(s,decimals=0))
    print("Speeds initialized")
    # print(time.time())
    return speed_times, objectives, final_speeds


def dummy_func():
    speed_times, objectives, final_speeds = optimize_gradient_armijo2('networks/2-2_coupledlights.json',4, tol=0.01, maxiter=3)

if __name__ == "__main__":
    variation = 10
    match variation:
        case 0:
            # Simple 2-2 junction with 
            speed_times, objectives, final_speeds = optimize_gradient_armijo2('networks/2-2_coupledlights.json', 3, tol=0.01, maxiter=20)
            print(objectives)
            print(speed_times)
            print(final_speeds)

        case 1:
            speeds, objectives = optimize_gradient_armijo('configs_variable/single_lane.json', 1, tol = 0.1, maxiter=10, c1 = 1e-4, c2=0.9)
            print(speeds, objectives)

        case 2:
            # 60.5245
            #torch.autograd.set_detect_anomaly(True)
            speeds, objectives, final = optimize_gradient_armijo('networks/1-1.json', 2, tol = 0.005, maxiter=20, c1 = 1e-4, c2=0.9)

            print(speeds)
            print(objectives)
            print(final)

        case 3:
            # 66.12872314453125
            T, roads, junctions = load.read_json('networks/1-1.json')

            #first = torch.tensor([80.0000, 67.2727, 75.0000, 50.0000, 40.0000, 35.0000])
            #second = torch.tensor([80.0000, 80., 80., 80.0000, 80.0000, 80.0000, 80.0])
            second = torch.tensor([30, 30, 30, 30, 30.0000])

            #_, _, network = initialize_road_network(T, roads, junctions, first, ControlPoints)
            #history = network.solve_cons_law()
            _, _, network = initialize_road_network(100, roads, junctions, second)
            for road in network.roads:
                print(road.Vmax)
            history, queues = network.solve_cons_law()

            import plotting as plot
            fig, ax = plot.plot_results(history, 1, network)
            plt.show()

            # #objective = point_flux(history,network)
            objective = total_travel_time(history,network)

            # #print(objective)
            print(objective)

        case 4:
            # Simple 1-1 junction with fixed speed limits only looking at traffic lights
            # Problem: Tend to optimize 1 of the cycle times before the other time. The second
            # time is also changed slightly in the direction of the change of the first
            # Problem arises if dominant cycle time reaches upper/lower bound. Then the 
            # less dominant may be changed in the wrong direction so that no updating is performed.

            speed_times, objectives, final_speeds = optimize_gradient_armijo2('networks/1-1trafficLight.json',4, tol=0.01, maxiter=15)
            print(objectives)
            print(speed_times)
            print(final_speeds)

        case 5:
            T, roads, junctions = load.read_json('networks/1-1trafficLight.json')

            all_params = torch.tensor([50., 50., 100., 10.0])
    
            loaded_roads, loaded_junctions, network = initialize_road_network2(1000, roads, junctions, all_params)

            history, queues = network.solve_cons_law()
            print(len(history[0].keys()))
            objective = total_travel_time_queue(history, queues, network)
            print(objective)


        case 6:
            # Simple 1-1 junction with fixed speed limits only looking at traffic lights
            # Problem: Tend to optimize 1 of the cycle times before the other time. The second
            # time is also changed slightly in the direction of the change of the first
            # Problem arises if dominant cycle time reaches upper/lower bound. Then the 
            # less dominant may be changed in the wrong direction so that no updating is performed.
            speed_times, objectives, final_speeds = optimize_gradient_armijo2('networks/2-2_coupledlights.json',4, tol=0.01, maxiter=10)
            # print("End time")
            # print(time.time())

            print(objectives)
            print(speed_times)
            print(final_speeds)
            print("Done")

        case 7:
            times = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

            for t in times:
                T, roads, junctions = load.read_json('networks/2-2_coupledlights.json')

                all_params = torch.tensor([50.0000, 50.0000, 50.0000, 50.0000, t, t])
        
                loaded_roads, loaded_junctions, network = initialize_road_network2(1000, roads, junctions, all_params)

                # # Solve conservation law
                # for j in network.junctions:
                #     for light in j.trafficlights:
                #         print(light.starting_state)
                #         for c in light.cycle:
                #             print(c)
                history, queues = network.solve_cons_law()
                print(len(history[0].keys()))
                objective = total_travel_time_queue(history, queues, network)
                print(f" Objective value with cycle times equal to {t}: {objective}")


        case 8:
            T, roads, junctions = load.read_json('networks/1-1.json')
            
            all_params = torch.tensor([30., 30.])
    
            loaded_roads, loaded_junctions, network = initialize_road_network2(T, roads, junctions, all_params)

            history, queues = network.solve_cons_law()

            objective = total_travel_time_queue(history, queues, network)
            print(f" Objective value: {objective}")


        case 9:
            speed_times, objectives, final_speeds = optimize_gradient_armijo2('networks/1-1.json',4, tol=0.01, maxiter=15)
            print(objectives)
            print(speed_times)
            print(final_speeds)

        case 10:
            for i in range(5):
                print(f"Iteration number {i+1}")
                dummy_func()