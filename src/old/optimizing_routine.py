# Full iterative approach of setting up road network and then iteratively 
# choosing new speed limits

# pyswarm - used to get approximation of exact solution

import loading_json as load
import road_network as rn
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
        rho = history[j]
        # print(j)
        # print(rho)
        padding = network.roads[j].pad
        gamma = network.roads[j].gamma

        # Go through each of the roads and calculate integral of flux out of road
        # Calculate integral as linear interpolation
        times = list(rho.keys())
        for i in range(1, len(times)):
            idx1 = network.get_index(times[i])
            idx2 = network.get_index(times[i-1])

            dt = times[i] - times[i-1]
            right = fv.flux(rho[times[i]][-padding-1], gamma[idx1])
            left = fv.flux(rho[times[i-1]][-padding-1], gamma[idx2])

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
    # print("Gamma")
    # print(gamma)

    times = list(rho.keys())
    sub_times = [t for t in times if time_interval[0] <= t <= time_interval[1]]

    for i in range(1, len(sub_times)):
        dt = sub_times[i] - sub_times[i-1]
        idx1 = network.get_index(times[i])
        idx2 = network.get_index(times[i-1])

        right = fv.flux(rho[times[i]][pnt_idx], gamma[idx1])
        left = fv.flux(rho[times[i-1]][pnt_idx], gamma[idx2])
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
        times = list(rho.keys())    
        road_travel = 0

        prev_time = 0
        for l in range(road.pad+1, len(rho[0]) - road.pad):
            prev_time += road.dx * (rho[0][l] + rho[0][l-1]) / 2

        for k in range(1, len(times)):
            t = times[k]
            dt = times[k] - times[k-1]
            # Go through all times
            # Calculate integral of density
            new_time = 0
            for l in range(road.pad+1, len(rho[t])-road.pad):
                # Go through all interior points
                new_time += road.dx * (rho[t][l] + rho[t][l-1]) / 2
            
            road_travel += dt*(prev_time + new_time) / 2
            prev_time = new_time

        total_travel += road_travel

    return total_travel


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

def initialize_road_network(T, roads, junctions, vmaxes, control_points):
    '''
    Initializing a road network given a time, roads and junctions as well as the speed limits
    vmaxes.
    Could probably instead just update densities and speed limits of existing system...

    Need to be updated to take in vmaxes that may vary for each road
    '''

    n_speeds = len(control_points) + 1 # Number of different speed limits for each road

    loaded_roads = []
    loaded_junctions = []
    for l, r in enumerate(roads):
        # print(Vmax[l])
        idx = l * n_speeds
        loaded_roads.append(rn.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=[float(vmaxes[i]) for i in range(idx, idx+n_speeds)],
                    scheme=r['Scheme'], limiter="minmod",
                    initial = r["Init_distr"], inflow = r["Inflow"]))

    for j in junctions:
        loaded_junctions.append(rn.Junction(roads = [loaded_roads[i] for i in j["j_roads"]],
                                        entering = j["entering"], leaving = j["leaving"],
                                        distribution=j["distribution"], redlights=[]))
    network = rn.RoadNetwork(loaded_roads, loaded_junctions, T, control_points)

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
    T, roads, junctions, ControlPoints = load.read_json(filename)
    n = len(ControlPoints)

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
            loaded_roads, loaded_junctions, network = initialize_road_network(T, roads, junctions, Vmax, ControlPoints)
            # Solve conservation law
            history = network.solve_cons_law()

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
        alpha = 5 / max([abs(g) for g in grad])
        # print(alpha)
        new_loaded_roads, new_loaded_junctions, new_network = None, None, None
        new_history, new_objective = None, None
        new_params, new_grad = None, None

        armijo_fails = 0
        while not step_taken:
            # Check if steplength is okay
            new_Vmax = Vmax - alpha * grad
            new_Vmax = project(new_Vmax, Vlower, VUpper)
            if torch.norm(new_Vmax - Vmax) < tol:
                step_taken = True
            #new_Vmax = project(new_Vmax, Vlower, VUpper)
            # Need to do simulation at next iterate:

            # Initialize new road system, calculate objective function and gradient
            new_loaded_roads, new_loaded_junctions, new_network = initialize_road_network(T, roads, junctions, new_Vmax, ControlPoints)
            new_history = new_network.solve_cons_law()

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
                    new_params.append(road.Vmax)
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
                print(f"Previous objective: {objective}, new objective {new_objective}")
                print(f"Previous speeds: {Vmax} New speeds: {new_Vmax}")
                alpha = 0.5 * alpha

                if armijo_fails >= 5:
                    # Step size has become too small, stop iteration
                    step_taken = True

        # Gradient descent step taken
        # Update variables
        Vmax = project(new_Vmax, Vlower, VUpper) # Project onto bounding rectangle
        loaded_roads, loaded_junctions, network  = new_loaded_roads, new_loaded_junctions, new_network
        history = new_history
        grad = new_grad
        objective = new_objective

        speeds.append(Vmax.detach())
        objectives.append(objective.detach())

        curr_iter += 1

        error = torch.norm(speeds[curr_iter] - speeds[curr_iter-1])
    final_speeds = []
    for s in speeds[-1]:
        final_speeds.append(int(s))

    return speeds, objectives, final_speeds



if __name__ == "__main__":

    variation = 2
   
    match variation:
        case 0:
            pass
        case 1:
            speeds, objectives = optimize_gradient_armijo('configs_variable/single_lane.json', 1, tol = 0.1, maxiter=10, c1 = 1e-4, c2=0.9)
            print(speeds, objectives)

        case 2:
            speeds, objectives, final = optimize_gradient_armijo('configs_variable/simple_1-1.json', 2, tol = 0.1, maxiter=20, c1 = 1e-4, c2=0.9)

            print(speeds)
            print(objectives)
            print(final)

        case 3:
            T, roads, junctions, ControlPoints = load.read_json('configs_variable/simple_1-1.json')
            n = len(ControlPoints)
            #first = torch.tensor([80.0000, 67.2727, 75.0000, 50.0000, 40.0000, 35.0000])
            second = torch.tensor([80.0000, 80., 80., 50.0000, 40.0000, 35.0000])

            #_, _, network = initialize_road_network(T, roads, junctions, first, ControlPoints)
            #history = network.solve_cons_law()
            _, _, network2 = initialize_road_network(T, roads, junctions, second, ControlPoints)
            history2 = network2.solve_cons_law()

            #objective = point_flux(history,network)
            objective2 = point_flux(history2,network2)

            #print(objective)
            print(objective2)
