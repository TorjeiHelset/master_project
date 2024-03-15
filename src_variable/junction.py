import torch
import FV_schemes as fv
import optimize_flux as opt
import numpy as np
import scipy.integrate as integrate

# Maybe want to use torch.jit.script to speed up code, but not possible for member functions since self 
# is an argument
# In that case, it would be necessary to define a function outside of the class, call it 
# from the member function and send more of the member variables as arguments to the 
# function

def h_00(t):
    return 2*t**3 - 3*t**2 + 1

def h_10(t):
    return t**3 - 2*t**2 + t

def h_01(t):
    return -2*t**3 + 3*t**2

def h_11(t):
    return t**3 - t**2

def p_1(rho, h0, rho_m, hmax, m0):
    t_1 = rho/rho_m
    return h_00(t_1)*h0 + h_10(t_1)*rho_m*m0 + h_01(t_1)*hmax

def p_2(rho, h1, rho_m, hmax, m2):
    t_2 = rho/rho_m
    return h_00(t_2)*hmax + h_01(t_2)*h1 + h_11(t_2)*(1-rho_m)*m2


def priority_fnc(rho, h0=0.6, hmax=0.9, h1=0.6, rho_m=0.6):
    '''
    Cubic Hermite interpolation of the points (0, h0), (rho_m, hmax), (1, h1)
    with tangent slopes (hmax - h0)/rho_m, 0 and (h1 - hmax)/(1 - rho_m) at the three points
    '''

    if rho < rho_m:
        return p_1(rho, h0, rho_m, hmax, (hmax - h0)/rho_m)
    else:
        return p_2(rho, h1, rho_m, hmax, (h1 - hmax)/(1 - rho_m))
    

def short_stick_prob(x, d):
    '''
    Probability of a point x in [0,1] being hit by a stick of length d <= 0.5
    '''
    if x <= d:
        return x / (1 - d)
    elif x <= 1 - d:
        return d / (1 - d)
    else:
        return (1 - x) / (1 - d)
    
def long_stick_prob(x, d):
    '''
    Probability of a point x in [0,1] being hit by a stick of length d > 0.5
    '''
    if x <= 1 - d:
        return x / (1 - d)
    elif x <= d:
        return 1
    else:
        return (1 - x) / (1 - d)
    
def stick_prob(x,d):
    '''
    Probability of a point x in [0,1] being hit by a stick of length d (> 0)
    '''
    if d <= 0.5:
        return short_stick_prob(x,d)
    else:
        return long_stick_prob(x,d)
    
def two_short_explicit(d1, d2):
    '''
    Given two sticks of length d1 and d2, 0 < d1 <= d2 <= 0.5, 
    this function returns the explicit value of the integral
    int(0,1) 1 - p(x,d1) * p(x,d2) dx
    '''
    return 1 / ((1-d1)*(1-d2)) * (1/3*d1**3 + d1**2*d2 - d1**2 + 2*d1*d2**2 - 3*d1*d2 + d1 - d2**2 + d2)

def trapezoidal_rule(f, a=0, b=1, n=10):
    '''
    Assuming equidistant spacing
    '''
    dx = (b-a) / n
    xk = torch.linspace(a, b, n+1)
    int = torch.tensor(0.0)
    int += (f(xk[0]) + f(xk[-1]))
    for i in range(1,n):
        int += 2 * f(xk[i])
    int = int * dx / 2
    return int


def two_stick_quadrature(d1, d2):
    '''
    Replace with manually implemented trapezoidal(?) rule
    '''
    # return integrate.quad(lambda x: 1 - (1-stick_prob(x,d1))*(1-stick_prob(x,d2)), 0, 1)[0]
    return trapezoidal_rule(lambda x : 1 - (1-stick_prob(x,d1))*(1-stick_prob(x,d2)),
                            0, 1, 4)

def n_stick_quadrature(d_list):
    '''
    all lengths in d in d_list should satisfy 0 < d < 1
    Replace with manually implemented trapezoidal(?) rule
    '''
    try:
        # return integrate.quad(lambda x: 1 - torch.prod(torch.tensor([1-stick_prob(x,d) for d in d_list])), 0, 1)[0]
        return trapezoidal_rule(lambda x: 1 - torch.prod(torch.tensor([1-stick_prob(x,d) for d in d_list])),
                                0, 1, 4)
    except:
        # return integrate.quad(lambda x: 1 - np.prod([1-stick_prob(x,d) for d in d_list]), 0, 1)[0]
        return trapezoidal_rule(lambda x: 1 - np.prod(torch.tensor([1-stick_prob(x,d) for d in d_list])),
                                0, 1, 4)

class Junction:
    # Allow for roads to have different flux function
    roads = None # All roads that pass through Junction
    entering = None # Index of roads going in
    leaving = None # Index of roads going out
    priority = None # Assume priority equal for all roads
    distribution = None # Assume all roads have same distribution

    trafficlights = None # Define all traffic lights in junction
    coupled_trafficlights = None # Define all coupled traffic lights in junction

    duty_to_gw = False
    priorities = None
    crossing_connections = None
    max_crossing_connections = 0
    
    def __init__(self, roads, entering, leaving, distribution, trafficlights, coupled_trafficlights,
                 duty_to_gw=False, priorities=None, crossing_connections=None):
        # Make sure roads are either entering or leaving
        assert set(entering).isdisjoint(leaving)
        # Make sure all roads actually cross junction
        assert len(roads) == len(entering) + len(leaving)

        # Check that all roads have correct position of junction
        # junction_pos = roads[entering[0]].right_pos
        # for i in entering:
        #     assert roads[i].right_pos == junction_pos
            
        # for j in leaving:
        #     assert roads[j].left_pos == junction_pos


        # Make sure distribution is of correct dimension

        assert len(distribution[0]) == len(leaving)
        # Make sure distribution sums to 1
        # assert abs(sum(distribution) -1 ) <= 1e-4
        for i in range(len(distribution)):
            # For every incoming road all of the flux should be distributed
            # Distribution changed to be 2d array
            assert abs(sum(distribution[i]) -1) <= 1e-4

        for trafficlight in trafficlights:
            # Check that traffic light only contains roads in junction
            assert set(trafficlight.entering).issubset(set(entering))
            assert set(trafficlight.leaving).issubset(set(leaving))

        for coupled in coupled_trafficlights:
            # Check that the coupled traffic light only contains roads in junction
            assert set(coupled.a_entering).issubset(set(entering))
            assert set(coupled.b_entering).issubset(set(entering))
            assert set(coupled.a_leaving).issubset(set(leaving))
            assert set(coupled.b_leaving).issubset(set(leaving))

        self.roads = roads
        self.entering = entering
        for i in self.entering:
            self.roads[i].right = True

        self.leaving = leaving
        for j in self.leaving:
            self.roads[j].left = True

        self.distribution = distribution
        self.priority = [1/len(entering)] * len(entering)

        self.trafficlights = trafficlights
        self.coupled_trafficlights = coupled_trafficlights
        self.road_in = [self.roads[i] for i in self.entering]
        self.road_out = [self.roads[i] for i in self.leaving]


        self.duty_to_gw = duty_to_gw
        # Add check on format of priorities and crossing_connections
        self.priorities = priorities
        if self.duty_to_gw:
            self.max_priority = max([max(row) for row in priorities])
        self.crossing_connections = crossing_connections
        try:
            self.max_crossing_connections = max([max([len(row) for row in row_list]) for row_list in crossing_connections])
        except:
            self.max_crossing_connections = 0
            
    def divide_flux(self, t):
        '''
        When comparing the fluxes from different roads, use 
        gamma * f(rho) instead of f(rho)
        idx is index of speed limit to be used


        Optimization problems:
        
        The function uses a lot of for loops, which could potentially be avoided
        Should probably split into seperate functions to find out where exactly the time is 
        being spent

        The optimization part - find_parameters uses the most of the time
        This is difficult to optimize
        '''

        # Some of these probably also don't need to be created every time...
        rho_in = [road.rho[-road.pad] for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        max_dens_in = [road.max_dens for road in self.road_in]
        rho_out = [road.rho[road.pad-1] for road in self.road_out]
        gamma_out = [road.gamma[road.idx] for road in self.road_out]
        max_dens_out = [road.max_dens for road in self.road_out]

        # Can below code be optimized in any meaningfull way(?)
        # Would maybe make it very unreadable...

        n = len(self.entering)
        m = len(self.leaving)
        active = torch.ones((n,m))
        for light in self.trafficlights:
            for i in range(n):
                for j in range(m):
                    if self.entering[i] in light.entering and self.leaving[j] in light.leaving:
                        active[i,j] = light.activation_func(t)
        
        for light in self.coupled_trafficlights:
            for i in range(n):
                for j in range(m):
                    if self.entering[i] in light.a_entering and self.leaving[j] in light.a_leaving:
                        active[i,j] = light.a_activation(t)

                    if self.entering[i] in light.b_entering and self.leaving[j] in light.b_leaving:
                        active[i,j] = light.b_activation(t)

        ####################################################
        # Instead of returning fluxes, return beta parameters
        # Use beta parameters to calculate fluxes
        #####################################################
        beta, _ = opt.find_parameters(rho_in, rho_out, self.distribution, gamma_in, gamma_out, active,
                                        max_dens_in, max_dens_out)
        fluxes = torch.zeros((n, m))
        for i in range(n):
            for j in range(m):
                fluxes[i,j] = min(active[i,j]*self.distribution[i][j]*max_dens_in[i] * fv.D(rho_in[i].clone(), gamma_in[i]),
                                  beta[i][j]*max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j]))

        fluxes_in = [0]*n
        fluxes_out = [0]*m

        for i in range(n):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_in[i] = sum([fluxes[i][j] for j in range(m)]) / max_dens_in[i]

        for j in range(m):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_out[j] = sum([fluxes[i][j] for i in range(n)]) / max_dens_out[j]
        
        return fluxes_in, fluxes_out
    
    def calculate_activation(self, n, m, t):
        active = torch.ones((n,m))

        for light in self.trafficlights:
            for i in range(n):
                if self.entering[i] in light.entering:
                    for j in range(m):
                        if self.leaving[j] in light.leaving:
                            active[i,j] = light.activation_func(t)

        
        for light in self.coupled_trafficlights:
            for i in range(n):
                if self.entering[i] in light.a_entering:
                    for j in range(m):
                        if self.leaving[j] in light.a_leaving:
                            active[i,j] = light.a_activation(t)

                if self.entering[i] in light.b_entering:
                    for j in range(m):
                        if self.leaving[j] in light.b_leaving:
                            active[i,j] = light.b_activation(t)
        return active
    
    def calculate_demand(self, rho_in, gamma_in, active, max_dens_in, n, m):
        fluxes = torch.zeros((n, m))

        # Calculate the desired flux from each road i to road j
        # Scales the flux with the maximum density of the incoming road
        # as well as the activation of potential traffic lights and the
        # distribution of the traffic
        for i in range(n):
            # move D to be here to reduce the number of calls
            # is clone needed?
            # Change below to be a function of all of rho that return a new
            # tensor of length n
            i_flux = fv.D(rho_in[i].clone(), gamma_in[i])            
            for j in range(m):
                fluxes[i,j] = active[i,j]*self.distribution[i][j]*max_dens_in[i] * i_flux
        return fluxes
    
    def calculate_priority_params(self, rho_in, n, m, h0=0.6, hmax=0.9, h1=0.6, rho_m=0.6):
        '''
        Calculates the priority parameters, given the parameters of the priority function, 
        the densities of the incoming roads and a prirority list of the roads.
        Ideally there should be a priority list for each outgoing road

        Any edges with priority 0 in the priority list correspond to illegal edges - set
        the priority parameter to 0 for these edges

        Should not take in the actual rho_in, but rather the 

        Parameters of the priority function could potentially be optimized
        '''
        priority_list = self.priorities
        priority_params = [[0 for _ in range(len(self.leaving))] for _ in range(len(self.entering))]

        for j in range(m):
            non_zero = [i for i in range(n) if priority_list[i][j] != 0]
            if len(non_zero) == 0:
                # No legal edges into road j -> This should never happen, maybe throw an error
                continue

            nz_priorities = [priority_list[k][j] for k in non_zero]
            sorted_indexes = [x for _, x in sorted(zip(nz_priorities, non_zero))]
            if len(sorted_indexes) > 1:
                # More than one one legal edge into road j -> need prirority parameters
                density_in = self.distribution[sorted_indexes[0]][j] * rho_in[sorted_indexes[0]]
                priority_params[sorted_indexes[0]][j] = priority_fnc(density_in, h0, hmax,
                                                                    h1, rho_m)

                for i, idx in enumerate(sorted_indexes[1:-1]):
                    # Go through the indexes in decreasing order
                    # Don't need the last index -> set to 1 - sum(previous)
                    density_in = self.distribution[idx][j] * rho_in[idx]
                    nu_factor = priority_fnc(density_in, h0, hmax, h1, rho_m)
                    priority_params[idx][j] = (1 - sum([priority_params[l][j] for l in sorted_indexes[:i+1]])) * nu_factor

                priority_params[sorted_indexes[-1]][j] = 1 - sum([priority_params[l][j] for l in sorted_indexes[:-1]])
            
            elif len(sorted_indexes) == 1:
                # Only one legal edge into road j -> set priority parameter to 1
                priority_params[sorted_indexes[0]][j] = 1

        return priority_params
    
    def calculate_xi_two_crossing(self, actual_fluxes, crossing_connections, max_flux_in):
        # Two crossing connections
        # I'm using scipy quadrature rule here
        # This is maybe not supported by torch.autograd, so might
        # have to implement this myself...
        d1 = actual_fluxes[crossing_connections[0][0],crossing_connections[0][1]] / max_flux_in[crossing_connections[0][0]]
        d2 = actual_fluxes[crossing_connections[1][0],crossing_connections[1][1]] / max_flux_in[crossing_connections[1][0]]
        if d1 == 0 and d2 == 0:
            return 1
        elif d1 == 0:
            return 1 - d2
        elif d2 == 0:
            return 1 - d1
        elif d1 == 1 or d2 == 1:
            return 0 # also maybe missing gradient
        else:
            if d1 <= 0.5 and d2 <= 0.5:
                if d1 < d2:
                    return 1 - two_short_explicit(d1, d2)
                else:
                    return 1 - two_short_explicit(d2, d1)
            else:
                return 1 - two_stick_quadrature(d1, d2)
        
    
    def calculate_xi_n_crossing(self, actual_fluxes, crossing_connections, max_flux_in):
        # Calculate the lengths of the sticks
        # Sticks of length 0 can be ignored
        # If any sticks have length 1, then xi = 0
        lengths = []
        for i_c, j_c in crossing_connections:
            length = actual_fluxes[i_c,j_c] / max_flux_in[i_c]
            if length == 1:
                return 0
            elif length == 0:
                continue
            lengths.append(length)
        
        if len(lengths) == 0:
            return 1
        else:
            # Calculate using quadrature rule
            return 1 - n_stick_quadrature(lengths)
    
    def calculate_upper_bound(self, actual_fluxes, crossing_connections, i, max_flux_in,
                              demand_ij, epsilon=0.1):
        '''
        Calculates the upper bound determined by the flux on the crossing connections
        Upper bound is never lower than epsilon*max_flux_in[i], i.e. a small percentage of the 
        maximum flux, right now default is at 10%
        '''
        cloned_fluxes = actual_fluxes.clone()
        if len(crossing_connections) == 1:
            # Only one crossing connection
            xi = 1 - cloned_fluxes[crossing_connections[0][0],crossing_connections[0][1]]/max_flux_in[crossing_connections[0][0]]

        elif len(crossing_connections) == 2:
            xi = self.calculate_xi_two_crossing(cloned_fluxes, crossing_connections, max_flux_in)
        else:
            xi = self.calculate_xi_n_crossing(cloned_fluxes, crossing_connections, max_flux_in)
        epsilon = epsilon * max_flux_in[i]
        return torch.min(demand_ij, epsilon + xi*(max_flux_in[i] - epsilon))

    
    def calculate_fluxes(self, demand, capacities, priorities, n, m, max_flux_in, 
                         max_dens_in, max_dens_out):
        #crossing_connections = self.crossing_connections
        actual_fluxes = torch.zeros((n, m))

        assigned_fluxes = []
        upper_bounds = demand.clone()

        for n_crossing in range(self.max_crossing_connections + 1):
            for i in range(n):
                for j in range(m):
                    if len(self.crossing_connections[i][j]) == n_crossing and (i,j) not in assigned_fluxes:
                        # Connection has not been assigned a flux, and it has the correct number of crossing connections
                        if len(self.crossing_connections[i][j]) > 0:
                            # Update upper bound
                            upper_bounds[i,j] = self.calculate_upper_bound(actual_fluxes, 
                                                                           self.crossing_connections[i][j],
                                                                           i, max_flux_in, demand[i,j])
                        # Calculate actual flux
                        # upper_bound[i,j] should be less or equal to demand[i,j] and so
                        # it should not be necessary to include upper_bounds[i,j] below
                        # demand_sum = torch.tensor(0.0)
                        upper_bound_sum = torch.tensor(0.0)
                        for l in range(n):
                            if l != i:
                                # demand_sum += demand[l,j]
                                # upper_bound_sum += upper_bounds[l,j]
                                upper_bound_sum += torch.min(upper_bounds[l,j].clone(), demand[l,j].clone())
                        # interior_max = torch.max(capacities[j] - demand_sum,
                        #                                 capacities[j] - upper_bound_sum)
                        interior_max = capacities[j] - upper_bound_sum
                        supply_max = torch.max(priorities[i][j]*capacities[j], interior_max)
                        demand_max = torch.min(upper_bounds[i,j].clone(), demand[i,j].clone())
                        actual_fluxes[i,j] = torch.min(demand_max, supply_max.clone())
                        
                        # actual_fluxes[i,j] = torch.min(torch.min(upper_bounds[i,j], demand[i,j]), 
                        #                             torch.max(priorities[i][j]*capacities[j],
                        #                                 torch.max(
                        #                                 capacities[j] - demand_sum,
                        #                                 capacities[j] - upper_bound_sum
                        #                                 )))
                            

                        assigned_fluxes.append((i,j))
        
        # Fluxes on edges i,j should now be calculated
        # Calculate fluxes in and fluxes out
        fluxes_in = [0]*n
        fluxes_out = [0]*m

        for i in range(n):
            # Sum over all connections out of road i and scale with max density
            fluxes_in[i] = torch.sum(actual_fluxes[i]) / max_dens_in[i]
        
        for j in range(m):
            # Sum over all connections into road j and scale with max density
            fluxes_out[j] = torch.sum(actual_fluxes[:,j]) / max_dens_out[j]

        return fluxes_in, fluxes_out
            
    # @torch.jit.script
    def divide_flux_wo_opt(self, t):
        rho_in = [road.rho[-road.pad] for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        max_dens_in = torch.tensor([road.max_dens for road in self.road_in])
        rho_out = [road.rho[road.pad-1] for road in self.road_out]
        gamma_out = [road.gamma[road.idx] for road in self.road_out]
        max_dens_out = torch.tensor([road.max_dens for road in self.road_out])


        # If sentences can be moved outside of j for loop and split into two if-sentences
        # Would reducde the number of times the if-sentence is evaluated
        n = len(self.entering)
        m = len(self.leaving)
        active = torch.ones((n,m))

        # Probably quicker - need to check that the functionality is the same!
        for light in self.trafficlights:
            for i in range(n):
                if self.entering[i] in light.entering:
                    for j in range(m):
                        if self.leaving[j] in light.leaving:
                            active[i,j] = light.activation_func(t)

        
        for light in self.coupled_trafficlights:
            for i in range(n):
                if self.entering[i] in light.a_entering:
                    for j in range(m):
                        if self.leaving[j] in light.a_leaving:
                            active[i,j] = light.a_activation(t)

                if self.entering[i] in light.b_entering:
                    for j in range(m):
                        if self.leaving[j] in light.b_leaving:
                            active[i,j] = light.b_activation(t)
        
        # fluxes[i,j] is the flux from road i to road j
        fluxes = torch.zeros((n, m))

        # Calculate the desired flux from each road i to road j
        for i in range(n):
            # move D to be here to reduce the number of calls
            i_flux = fv.D(rho_in[i].clone(), gamma_in[i])            
            for j in range(m):
                fluxes[i,j] = active[i,j]*self.distribution[i][j]*max_dens_in[i] * i_flux
        
        
        cloned_fluxes = fluxes.clone()
    
        for j in range(m):
            capacity = max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j])

            # Update the flux from all roads into road j
            sum_influx = torch.sum(fluxes[:,j])
            if sum_influx > capacity:
                # If the sum of the fluxes is larger than the capacity, scale down the fluxes
                cloned_fluxes[:,j] = fluxes[:,j] * capacity / sum_influx
                # for i in range(n):
                #     cloned_fluxes[i,j] = fluxes[i,j] * capacity / sum_influx


        fluxes_in = [0]*n
        fluxes_out = [0]*m


        for i in range(n):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_in[i] = torch.sum(cloned_fluxes[i]) / max_dens_in[i]


        for j in range(m):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_out[j] = torch.sum(cloned_fluxes[:,j]) / max_dens_out[j]
        
        
        return fluxes_in, fluxes_out
    
    def divide_flux_wo_opt_duty_to_gw(self, t):
        rho_in = [road.rho[-road.pad] for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        max_dens_in = torch.tensor([road.max_dens for road in self.road_in])
        rho_out = [road.rho[road.pad-1] for road in self.road_out]
        gamma_out = [road.gamma[road.idx] for road in self.road_out]
        max_dens_out = torch.tensor([road.max_dens for road in self.road_out])


        # If sentences can be moved outside of j for loop and split into two if-sentences
        # Would reducde the number of times the if-sentence is evaluated
        n = len(self.entering)
        m = len(self.leaving)
        active = torch.ones((n,m))

        # Probably quicker - need to check that the functionality is the same!
        for light in self.trafficlights:
            for i in range(n):
                if self.entering[i] in light.entering:
                    for j in range(m):
                        if self.leaving[j] in light.leaving:
                            active[i,j] = light.activation_func(t)

        
        for light in self.coupled_trafficlights:
            for i in range(n):
                if self.entering[i] in light.a_entering:
                    for j in range(m):
                        if self.leaving[j] in light.a_leaving:
                            active[i,j] = light.a_activation(t)

                if self.entering[i] in light.b_entering:
                    for j in range(m):
                        if self.leaving[j] in light.b_leaving:
                            active[i,j] = light.b_activation(t)
        
        # fluxes[i,j] is the flux from road i to road j
        fluxes = torch.zeros((n, m))

        # Calculate the desired flux from each road i to road j
        for i in range(n):
            # move D to be here to reduce the number of calls
            i_flux = fv.D(rho_in[i].clone(), gamma_in[i])            
            for j in range(m):
                fluxes[i,j] = active[i,j]*self.distribution[i][j]*max_dens_in[i] * i_flux
        # print("Fluxes:")
        # print(fluxes)
        # print(f"T = {t}")
        # print(f"Desired fluxed before scaling and duty to give way: {fluxes}")

        actual_fluxes = torch.zeros((n, m))
        remaining_capacities = [max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j]) for j in range(m)]
        # print(f"Capacities of outgoing roads: {remaining_capacities}")
        
        for p in range(1, self.max_priority+1):
            # Go through all priorities
            for j in range(m):
                # Go through all roads leaving the junction
                if abs(remaining_capacities[j]) < 1e-10:
                    # No capacity left on road j
                    continue
                for i in range(n):
                    # Go through all roads entering the junction
                    if self.priorities[i][j] == p:
                        # Connnection i->j has priority p and should be updated
                        # All crossing connections should already be updated
                        if len(self.crossing_connections[i][j]) == 0:
                            # No crossing connections, use remaining out capacity to update flux
                            # print(f"Connection {i}->{j} has priority {p} and no crossing connections")
                            try:
                                actual_fluxes[i,j] = torch.min(remaining_capacities[j], fluxes[i,j])
                            except:
                                actual_fluxes[i,j] = min(remaining_capacities[j], fluxes[i,j])
                            # print(f"Actual flux from road {i} to road {j}: {actual_fluxes[i,j]}")
                            remaining_capacities[j] = remaining_capacities[j] - actual_fluxes[i,j]
                            # print(f"Remaining capacity of road {j}: {remaining_capacities[j]}")

                        else:
                            # There is at least one crossing connection that should be used to update
                            # the flux
                            # print(f"Connection {i}->{j} has priority {p} and at least crossing connections")
                            total_crossing_flux = torch.tensor(0.0)
                            for i_c, j_c in self.crossing_connections[i][j]:
                                total_crossing_flux = total_crossing_flux + actual_fluxes[i_c,j_c]

                            upper_bound = torch.max(torch.tensor(0.0),
                                                    max_dens_in[i]*fv.D(torch.tensor(1.0), gamma_in[i]) - total_crossing_flux)
                            # print(f"Total flux that needs to be crossed: {total_crossing_flux}")
                            # print(f"Maximum outflux from road {i}: {max_dens_in[i]*fv.D(torch.tensor(1.0), gamma_in[i])}")
                            # print(f"Upper bound for road {i} to road {j}: {upper_bound}")
                            try:
                                actual_fluxes[i,j] = torch.min(remaining_capacities[j], fluxes[i,j], upper_bound)
                            except:
                                actual_fluxes[i,j] = min(remaining_capacities[j], fluxes[i,j], upper_bound)
            #                 print(f"Actual flux from road {i} to road {j}: {actual_fluxes[i,j]}")
            #                 remaining_capacities[j] = remaining_capacities[j] - actual_fluxes[i,j]
            #                 print(f"Remaining capacity of road {j}: {remaining_capacities[j]}")
                            
            # print("-----------------------------\n")


        fluxes_in = [0]*n
        fluxes_out = [0]*m

        for i in range(n):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_in[i] = torch.sum(actual_fluxes[i]) / max_dens_in[i]

        for j in range(m):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_out[j] = torch.sum(actual_fluxes[:,j]) / max_dens_out[j]
        
        # print(f"Fluxes in: {fluxes_in}")
        # print(f"Fluxes out: {fluxes_out}")
        # print("****************************************************\n\n")
        if abs(fluxes_in[2] - fluxes_out[2]) > 1e-4:
            print(f"In and out fluxes to road 2 are not equal!")
            print(f"Flux in: {fluxes_in[2]}, Flux out: {fluxes_out[2]}")
            print(f"Time: {t}")
        return fluxes_in, fluxes_out
    
    def divide_flux_wo_opt_right_of_way(self, t):
        '''
        Calculate how the flux is being distributed across a junction when there is merging or crossing traffic
        In this case we need a priority function to determine the percentage of traffic going to the different outgoing roads
        Each junction has n incoming roads and m outgoing roads. 

        Whenever there is more than one outgoing road, a distribution matrix needs to be specified
        Whenever there is more than one incoming road, some priority list needs to be specified.
        This prirority list and the densities of the incoming roads will be used to find prirotiy
        parameters.
        '''

        rho_in = [road.rho[-road.pad] for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        max_flux_in = [fv.fmax(gamma) for gamma in gamma_in]
        max_dens_in = torch.tensor([road.max_dens for road in self.road_in])
        rho_out = [road.rho[road.pad-1] for road in self.road_out]
        gamma_out = [road.gamma[road.idx] for road in self.road_out]
        max_dens_out = torch.tensor([road.max_dens for road in self.road_out])

        n = len(self.entering)
        m = len(self.leaving)

        # Steps of the algorithm:

        # 1. Calculate how much of the demand is limited by traffic lights
        activation = self.calculate_activation(n, m, t)

        # 2. Calculate the actual desired flux from each road i to road j
        demand = self.calculate_demand(rho_in, gamma_in, activation, max_dens_in, n, m)

        # 3. Calculate the capacity of each road j
        # Scaled with the maximum density of the outgoing road
        capacities = [max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j]) for j in range(m)]

        # 4. Determine a set of prirority parameters for each outgoing road j
        # This needs to be permuted back to original order!!
        priorities = self.calculate_priority_params(rho_in, n, m)
        
        # 5. Determine the actual flux between road i and road j:
        # 5.1 Start with roads with no crossing connections
        # 5.2 For roads with one crossing connection calculate upper bound
        # 5.3 Calculate actual flux from road i to road j with one crossing connection
        # 5.4 Repeat for roads with more crossing connections
        # 6. Sum over fluxes from i to j to get fluxes in and fluxes out
        fluxes_in, fluxes_out = self.calculate_fluxes(demand, capacities, priorities, n, m, 
                                                      max_flux_in, max_dens_in, max_dens_out)

        return fluxes_in, fluxes_out

    def apply_bc(self, dt, t):
        '''
        Calculate how flux is divided among roads
        To this end gamma of each road is important
        Actual flux is gamma*f(rho) instad of just f(rho)
        ...
        '''
        #--------------------------------
        # Dividing flux need to somehow take into account gamma of each road
        # --------------------------------

        # Need to change divide_flux3 so that the fluxes depend on parameter, i.e. are differentiable
        fluxes_in, fluxes_out = self.divide_flux(t)

        # What happens if flux in/out is 0 -> Right now it looks like density can reach negative values, whic
        # is unphysical...
        
        # outputed fluxes should now be gamma*f(rho)


        #---------------------------------------------
        # Note not exactly correct: using time tn+1 to do updating, but should actually use
        # time tn
        # Maybe not a very big problem ????
        # Solution: save out/in flux at previous step somewhere
        #           not very difficult so should maybe just do this
        #---------------------------------------------

        # Can this be done without for loop?
        # Most of the time save is inside divide_flux anyways

        # These two codeblocks take up over half of the running time of the entire code
        # It is very worthwhile to try to vectorize these if possible bc this is where most of the time save is
        # Don't spend time on this rn...

        for i, flux in enumerate(fluxes_in):
            road = self.road_in[i]
            left, in_mid = road.rho[-road.pad-1], road.rho[-road.pad]
            s = torch.max(torch.abs(fv.d_flux(left, road.gamma[road.idx])), torch.abs(fv.d_flux(in_mid, road.gamma[road.idx])))
            left_f = fv.flux(left.clone(), road.gamma[road.idx])
            mid_f = fv.flux(in_mid.clone(), road.gamma[road.idx])
            left_flux = 0.5 * (left_f + mid_f) - 0.5 * s * (in_mid - left)

            # Don't multiply with gamma in denominator because flux is already multiplied with
            # gamma
            road.rho[-road.pad] = road.rho[-road.pad] - dt/road.dx * (flux - left_flux)
            if road.pad > 1:
                road.rho[-road.pad+1] = road.rho[-road.pad]
        
        for i, flux in enumerate(fluxes_out):
            road = self.road_out[i]

            right, out_mid = road.rho[road.pad], road.rho[road.pad-1]
            s = torch.max(torch.abs(fv.d_flux(out_mid, road.gamma[road.idx])), torch.abs(fv.d_flux(right, road.gamma[road.idx])))
            mid_f = fv.flux(out_mid.clone(), road.gamma[road.idx])
            right_f = fv.flux(right.clone(), road.gamma[road.idx])
            right_flux = 0.5 * (mid_f + right_f) - 0.5 * s * (right - out_mid)

            road.rho[road.pad-1] = road.rho[road.pad-1] - dt / road.dx * (right_flux - flux)
            if road.pad > 1:
                road.rho[0] = road.rho[1]

    # @torch.jit.script
    def apply_bc_wo_opt(self, dt, t):
        '''
        To save time, assume that the flux is distributed equally among roads
        Could alternatively make the user specify the distribution
        '''
        # Also not necessary to create every time, instead store as member
        # variable!
        if self.duty_to_gw:
            # fluxes_in, fluxes_out = self.divide_flux_wo_opt_duty_to_gw(t)
            fluxes_in, fluxes_out = self.divide_flux_wo_opt_right_of_way(t)
        else:
            fluxes_in, fluxes_out = self.divide_flux_wo_opt(t)

        # Ideally want to reduce the number of calls to d_flux and flux
        # The code below is rather slow, but difficult to optimize
        
        for i, flux in enumerate(fluxes_in):
            road = self.road_in[i]
            left, in_mid = road.rho[-road.pad-1], road.rho[-road.pad]
            
            s = torch.max(torch.abs(fv.d_flux(left, road.gamma[road.idx])), torch.abs(fv.d_flux(in_mid, road.gamma[road.idx])))
            left_f = fv.flux(left.clone(), road.gamma[road.idx])
            mid_f = fv.flux(in_mid.clone(), road.gamma[road.idx])
            left_flux = 0.5 * (left_f + mid_f) - 0.5 * s * (in_mid - left)

            # Don't multiply with gamma in denominator because flux is already multiplied with
            # gamma
            road.rho[-road.pad] = road.rho[-road.pad] - dt/road.dx * (flux - left_flux)
            if road.pad > 1:
                road.rho[-road.pad+1] = road.rho[-road.pad]
        
        for i, flux in enumerate(fluxes_out):
            road = self.road_out[i]

            right, out_mid = road.rho[road.pad], road.rho[road.pad-1]
            s = torch.max(torch.abs(fv.d_flux(out_mid, road.gamma[road.idx])), torch.abs(fv.d_flux(right, road.gamma[road.idx])))
            mid_f = fv.flux(out_mid.clone(), road.gamma[road.idx])
            right_f = fv.flux(right.clone(), road.gamma[road.idx])
            right_flux = 0.5 * (mid_f + right_f) - 0.5 * s * (right - out_mid)
            # if i == 2:
            #     print(f"Dividing flux on outgoing road {i}")
            #     print(f"Incoming flux: {flux}, Outgoing flux: {right_flux}")
            #     print(f"Density on road before: {road.rho[road.pad-1]}")
            road.rho[road.pad-1] = road.rho[road.pad-1] - dt / road.dx * (right_flux - flux)
            # if i == 2:
            #     print(f"Density on road after: {road.rho[road.pad-1]}")
            if road.pad > 1:
                road.rho[0] = road.rho[1]

    def get_next_control_point(self, t):
        '''
        Given a time t, this function returns the next time where a jump occurs
        It should also maybe return some of the points in the jump itself to capture
        the full change in state


        For some reason gets stuck whenever t1 or t2 are float ... Why????
        '''
        if torch.is_tensor(t):
            t = t.detach()

        control_point = t + 100 # Just set to some value so that actual control points will be considered

        for light in self.trafficlights:
            t1 =  torch.round(light.cycle[0].detach(), decimals=0)
            t2 =  torch.round(light.cycle[1].detach(), decimals=0)
            jump_time = 10
            period_time = t % (t1 + t2)
            if period_time < jump_time/2:
                control_point = min(control_point, t-period_time+jump_time/2)
            elif period_time < jump_time:
                control_point = min(control_point, t-period_time+jump_time)
            elif period_time < t1:
                control_point = min(control_point, t-period_time+t1)
            elif period_time < t1 + jump_time/2:
                control_point = min(control_point, t-period_time+t1+jump_time/2)
            elif period_time < t1 + jump_time:
                control_point = min(control_point, t-period_time+t1+jump_time)
            else:
                control_point = min(control_point, t-period_time+t1+t2)

        for light in self.coupled_trafficlights:
            # Maybe add some extra point here if two activation functions are used
            t1 =  torch.round(light.cycle[0].detach(), decimals=0)
            t2 =  torch.round(light.cycle[1].detach(), decimals=0)
            jump_time = 10
            period_time = t % (t1 + t2)
            if period_time < jump_time/2:
                control_point = min(control_point, t-period_time+jump_time/2)
            elif period_time < jump_time:
                control_point = min(control_point, t-period_time+jump_time)
            elif period_time < t1:
                control_point = min(control_point, t-period_time+t1)
            elif period_time < t1 + jump_time/2:
                control_point = min(control_point, t-period_time+t1+jump_time/2)
            elif period_time < t1 + jump_time:
                control_point = min(control_point, t-period_time+t1+jump_time)
            else:
                control_point = min(control_point, t-period_time+t1+t2)
            
        return control_point

    def get_activation(self, t, id_1, id_2):
        '''
        Given a time t, this function returns the activation of roads with id_1 and id_2
        '''
        # First check if both id_1 and id_2 are a part of the junction
        id_1_in = False
        in_idx = -1
        for i in self.entering:
            if self.roads[i].id == id_1:
                id_1_in = True
                in_idx = i
                break
        
        id_2_in = False
        out_idx = -1
        for i in self.leaving:
            if self.roads[i].id == id_2:
                id_2_in = True
                out_idx = i
                break
        
        if not id_1_in or not id_2_in:
            return False, 0.0
        
        else:
            # id_1 is entering into the junction, id_2 is leaving
            # Assume that this specific crossing is either combined with a traffic light or a 
            # coupled traffic light, but not both
            activation = 1.0
            for light in self.trafficlights:
                if in_idx in light.entering and out_idx in light.leaving:
                    activation = light.activation_func(t)
                    # print("Regular traffic light")
                    return True, activation

            for light in self.coupled_trafficlights:
                if in_idx in light.a_entering and out_idx in light.a_leaving:
                    activation = light.a_activation(t)
                    # print("Coupled traffic light, state a")
                    return True, activation

                if in_idx in light.b_entering and out_idx in light.b_leaving:
                    activation = light.b_activation(t)
                    return True, activation
        return True, activation
    
    def get_speed(self, t, id_1, id_2):
        '''
        Need to reevaluate how the flux is divided among the roads
        -> Change this to avoid reevaluations...
        '''
        idx_1 = 0
        idx_2 = 0
        for i, road in enumerate(self.road_in):
            if road.id == id_1:
                idx_1 = i
                break
        for i, road in enumerate(self.road_out):
            if road.id == id_2:
                idx_2 = i
                break
        

        rho_in = [road.rho[-road.pad] for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        max_dens_in = torch.tensor([road.max_dens for road in self.road_in])
        rho_out = [road.rho[road.pad-1] for road in self.road_out]
        gamma_out = [road.gamma[road.idx] for road in self.road_out]
        max_dens_out = torch.tensor([road.max_dens for road in self.road_out])

        n = len(self.entering)
        m = len(self.leaving)
        active = torch.ones((n,m))

        for light in self.trafficlights:
            for i in range(n):
                if self.entering[i] in light.entering:
                    for j in range(m):
                        if self.leaving[j] in light.leaving:
                            active[i,j] = light.activation_func(t)

        
        for light in self.coupled_trafficlights:
            for i in range(n):
                if self.entering[i] in light.a_entering:
                    for j in range(m):
                        if self.leaving[j] in light.a_leaving:
                            active[i,j] = light.a_activation(t)

                if self.entering[i] in light.b_entering:
                    for j in range(m):
                        if self.leaving[j] in light.b_leaving:
                            active[i,j] = light.b_activation(t)

        fluxes = torch.zeros((n, m))

        # Calculate the desired flux from each road i to road j
        for i in range(n):
            # move D to be here to reduce the number of calls
            i_flux = fv.D(rho_in[i].clone(), gamma_in[i])            
            for j in range(m):
                # print(f"Desired flux from road {i} to road {j}: {self.distribution[i][j]* i_flux * max_dens_in[i]}")
                # print(f"Flux scaled with traffic light from road {i} to road {j}: {active[i,j]*self.distribution[i][j] * i_flux * max_dens_in[i]}")
                fluxes[i,j] = active[i,j]*self.distribution[i][j]*max_dens_in[i] * i_flux
        
        cloned_fluxes = fluxes.clone()
        # calculate the capacity of each road j
        for j in range(m):
            # print(f"Capacity of road {j}: {max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j])}")
            capacity = max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j])

            # Update the flux from all roads into road j
            sum_influx = torch.sum(fluxes[:,j])
            # print(f"Total sum of fluxes into road {j}: {sum_influx}")
            if sum_influx > capacity:
                # If the sum of the fluxes is larger than the capacity, scale down the fluxes
                cloned_fluxes[:,j] = fluxes[:,j] * capacity / sum_influx # is this the inplace in question?
                
        flux = cloned_fluxes[idx_1, idx_2] / max_dens_in[idx_1]
        speed = flux / rho_in[idx_1] * self.road_in[idx_1].L
        return speed
    