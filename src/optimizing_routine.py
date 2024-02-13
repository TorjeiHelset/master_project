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
            dt = times[i] - times[i-1]
            right = fv.flux(rho[times[i]][-padding-1], gamma)
            left = fv.flux(rho[times[i-1]][-padding-1], gamma)

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
        right = fv.flux(rho[times[i]][pnt_idx], gamma)
        left = fv.flux(rho[times[i-1]][pnt_idx], gamma)
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

def initialize_road_network(T, roads, junctions, vmaxes):
    '''
    Initializing a road network given a time, roads and junctions as well as the speed limits
    vmaxes.
    Could probably instead just update densities and speed limits of existing system...
    '''

    loaded_roads = []
    loaded_junctions = []
    for l, r in enumerate(roads):
        # print(Vmax[l])
        loaded_roads.append(rn.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=float(vmaxes[l]),
                    scheme=r['Scheme'], limiter="minmod",
                    initial = r["Init_distr"], inflow = r["Inflow"]))

    for j in junctions:
        loaded_junctions.append(rn.Junction(roads = [loaded_roads[i] for i in j["j_roads"]],
                                        entering = j["entering"], leaving = j["leaving"],
                                        distribution=j["distribution"], redlights=[]))
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
        Vmax.append(torch.tensor(r['Vmax']))
        Vlower.append(torch.tensor(r['low']))
        VUpper.append(torch.tensor(r['high']))

    print(f"Upper limits {VUpper}")
    print(f"Lower limits {Vlower}")

    Vmax = torch.tensor(Vmax)

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
            history = network.solve_cons_law()

            # Calculate objective function to minimize
            match objective_type:
                case 0:
                    objective = out_flux(history, network)
                case 1:
                    objective = point_flux(history, network)
                case 2:
                    objective = total_travel_time(history, network)
                case 3:
                    objective = network.roads[0].Vmax**2

            # Storing parameters in a list
            params = []
            for road in network.roads:
                params.append(road.Vmax)

            # Calculating gradient wrt parameters
            grad = jacobi(objective, params)
            

            speeds.append(Vmax.detach())
            objectives.append(objective.detach())


        print(f"Iteration {curr_iter+1}")
        print(f"Current objective value: {objective}")

        

        # Performing gradient descent step
        step_taken = False
        # First guess of steplength so that speed limit changes by 5
        alpha = 5 / max([abs(g) for g in grad])
        
        new_loaded_roads, new_loaded_junctions, new_network = None, None, None
        new_history, new_objective = None, None
        new_params, new_grad = None, None

        while not step_taken:
            # Check if steplength is okay
            new_Vmax = Vmax - alpha * grad
            # Need to do simulation at next iterate:

            # Initialize new road system, calculate objective function and gradient
            new_loaded_roads, new_loaded_junctions, new_network = initialize_road_network(T, roads, junctions, new_Vmax)
            new_history = new_network.solve_cons_law()

            match objective_type:
                case 0:
                    new_objective = out_flux(new_history, new_network)
                case 1:
                    new_objective = point_flux(new_history,new_network)
                case 2:
                    new_objective = total_travel_time(new_history, new_network)
                case 3:
                    new_objective = new_network.roads[0].Vmax**2
            
            new_params = []
            for road in new_network.roads:
                new_params.append(road.Vmax)
            new_grad = jacobi(new_objective, new_params)
            #print(new_grad)

            # Check wolfe-armijo conditions
            i_satisfied = new_objective <= objective - c1 * torch.dot(grad, grad) # Grad*grad should be norm of grad
            # ii not needed since first guess is taken to be rather large
            #ii_satisfied = True # inner product of gradients <= c2*norm of old gradient

            if i_satisfied: #and ii_satisfied:
                # Perform step
                step_taken = True
            else:
                print("Armijo failed")
                alpha = 0.5 * alpha

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
    
    for j in range(len(speeds[-1])):
        speeds[-1][j] = int(speeds[-1][j])

    return speeds, objectives



def optimize_speed_limits(filename, objective_type, tol = 1e-4, maxiter = 10):
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

    ----------------------------------------------------------
    Approach:

    1.  Initialize all roads, Junctions and Network
    2.  Solve the conservation law for this road network until time T given in config file
    3.  Calculate derivatives of all parameters (first only speed limits, and later for traffic
        lights as well).
    4.  Get new guess for parameters, maybe using newton method
    5.  Go back to step 1, with new parameters - repeat untill "close enough" to optimal solution
        Note that this solution might only be locally optimal

    ----------------------------------------------------------


    This function supports a number of different objective functions
    objective_type = 0  ->  Calculate the time interval over all times
                            of the flux out of each road
    objective_type = 1  ->  Calculate the time interval over some small time interval
                            of the flux out of each road
    '''
    T, roads, junctions = load.read_json(filename)

    Vmax = []
    Vlower = []
    VUpper = []

    for r in roads:
        Vmax.append(torch.tensor(r['Vmax']))
        Vlower.append(torch.tensor(r['low']))
        VUpper.append(torch.tensor(r['high']))

    print(f"Upper limits {VUpper}")
    print(f"Lower limits {Vlower}")

    Vmax = torch.tensor(Vmax)

    error = tol + 1
    curr_iter = 0
    speeds = []
    objectives = []
    while error > tol and curr_iter < maxiter:
        # Perform gradient descent step untill

        # Initialize system
        loaded_roads, loaded_junctions, network = initialize_road_network(T, roads, junctions, Vmax)
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
            case 3:
                objective = - network.roads[0].Vmax**2
            
        print(f"Iteration {curr_iter+1}")
        print(f"Current objective value: {objective}")

        # Storing previous speed and objective function
        speeds.append(Vmax.detach())
        objectives.append(objective.detach())

        params = []
        for road in network.roads:
            params.append(road.Vmax)

        
        # Calculate gradient and hessian
        grad = jacobi(objective, params)
        #print(grad)
        H = hessian(grad, params)
        h = torch.linalg.solve(H, grad)

        # For now just do gradient descent step

        Vmax = Vmax - h
        print(Vmax)
        # What steplength to use?
        #Vmax = Vmax.clone().detach() - grad / abs(objective) # What steplength?

        # Check that Vmax is inside boundary
        for l in range(len(Vmax)):
            Vmax[l] = torch.min(VUpper[l], torch.max(Vlower[l], Vmax[l]))

        

        curr_iter += 1

    return speeds, objectives

def dummy_obj_func(a,b,c,d):
    return a**2 * ( c - d*c) - c**2 + d**2 + b
    
                            
def plot_objective_against_speed(objective_type, filename, nsteps = 10):
    T, roads, junctions = load.read_json(filename)
    Vlower = []
    VUpper = []

    for r in roads:
        Vlower.append(torch.tensor(r['low']))
        VUpper.append(torch.tensor(r['high']))

    Vmaxes = torch.linspace(Vlower[0], VUpper[0], nsteps)
    print(Vmaxes)
    objectives = torch.zeros_like(Vmaxes)
    grads = torch.zeros_like(Vmaxes)
    Hessians = torch.zeros_like(Vmaxes)

    for i in range(len(Vmaxes)):
        V= Vmaxes[i]
        loaded_roads = []
        loaded_junctions = []
        for l, r in enumerate(roads):
            # print(Vmax[l])
            loaded_roads.append(rn.Road(b=r['b'], L = r['L'], N = r['N'], Vmax=V,
                        scheme=r['Scheme'], limiter="minmod",
                        initial = r["Init_distr"], inflow = r["Inflow"]))

        for j in junctions:
            loaded_junctions.append(rn.Junction(roads = [loaded_roads[i] for i in j["j_roads"]],
                                            entering = j["entering"], leaving = j["leaving"],
                                            distribution=j["distribution"], redlights=[]))
        network = rn.RoadNetwork(loaded_roads, loaded_junctions, T)

        # Solve conservation law
        history = network.solve_cons_law()

        match objective_type:
            case 0:
                objective = out_flux(history, network)
            case 1:
                objective = point_flux(history, network)
            case 2:
                objective = total_travel_time(history, network)
            case 3:
                objective = - network.roads[0].Vmax**2

        params = []
        for road in network.roads:
            params.append(road.Vmax)
        grad = jacobi(objective, params)
        H = hessian(grad, params)

        objectives[i] = objective.detach()
        grads[i] = grad[0].detach()
        Hessians[i] = H[0,0].detach()
    
    return Vmaxes, objectives, grads, Hessians


if __name__ == "__main__":

    variation = 2
   
    match variation:
        case 0:
            speeds, objectives, grads, Hs = plot_objective_against_speed(3, 'configs/single_lane.json', nsteps = 3)
            plt.plot(speeds, objectives)
            ds = speeds[1] - speeds[0]
            for i in range(len(speeds)):
                s = speeds[i]
                plt.plot([s-ds/2, s+ds/2], [objectives[i] - grads[i]*ds/2, objectives[i] + grads[i]*ds/2], 'r')
            #plt.plot(speeds, Hs)


            plt.show()
        case 1:
            speeds, objectives = optimize_gradient_armijo('configs/single_lane.json', 2, tol = 0.1, maxiter=5, c1 = 1e-4, c2=0.9)
            print(speeds, objectives)

        case 2:
            speeds, objectives = optimize_gradient_armijo('configs/simple_1-1.json', 0, tol = 0.01, maxiter=3, c1 = 1e-4, c2=0.9)
            print(speeds)
            print(objectives)