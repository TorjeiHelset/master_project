import torch
import FV_schemes as fv
import numpy as np

#--------------------------------------------------------
# Functions used for calculating the upper bound when crossing multiple other lanes
# Can potentially be moved to separate file
#--------------------------------------------------------

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
        return torch.tensor(1.0)
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
    Replace with manually implemented trapezoidal rule
    '''
    return trapezoidal_rule(lambda x : 1 - (1-stick_prob(x,d1))*(1-stick_prob(x,d2)),
                            0, 1, 4)

def n_stick_quadrature(d_list):
    '''
    all lengths in d in d_list should satisfy 0 < d < 1
    Evaluate integral using trapezoidal rule

    
    '''
    try:
        # return trapezoidal_rule(lambda x: 1 - torch.prod(torch.tensor([1-stick_prob(x,d) for d in d_list])),
        #                         0, 1, 4)
        return trapezoidal_rule(lambda x : 1.0 - torch.prod(torch.stack([1.-stick_prob(x,d) for d in d_list])),
                                0,1,4)
    except:
        # This should probably never happen - maybe throw an exception...
        return trapezoidal_rule(lambda x: 1 - np.prod(torch.tensor([1-stick_prob(x,d) for d in d_list])),
                                0, 1, 4)
    
class Junction:
    # Allow for roads to have different flux function
    # These member variables are all initialized in the __init__ function, and 
    # so probably do not need to be created here...

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
        # Making sure the variables are initialized correctly:
        # Make sure roads are either entering or leaving
        assert set(entering).isdisjoint(leaving)
        # Make sure all roads actually cross junction
        assert len(roads) == len(entering) + len(leaving)
        # Make sure distribution is of correct dimension
        assert len(distribution[0]) == len(leaving)
        for i in range(len(distribution)):
            # For every incoming road all of the flux should be distributed, i.e. summing to 1
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

        # Initialize member variables:
        self.roads = roads
        self.entering = entering
        for i in self.entering:
            self.roads[i].right = True

        self.leaving = leaving
        for j in self.leaving:
            self.roads[j].left = True

        self.distribution = torch.tensor(distribution)
        self.priority = [1/len(entering)] * len(entering)

        self.trafficlights = trafficlights
        self.coupled_trafficlights = coupled_trafficlights
        self.road_in = [self.roads[i] for i in self.entering]
        self.road_out = [self.roads[i] for i in self.leaving]

        # for road in self.road_in:
        #     if road.id == "e18_w_2bw":
        #         print(f"e18_w_2bw is entering a junction!")

        # for road in self.road_out:
        #     if road.id == "e18_w_2bw":
        #         print(f"e18_w_2bw is leaving a junction!")

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
        
        self.n = len(self.entering)
        self.m = len(self.leaving)

    def get_next_control_point(self, t):
        '''
        Given a time t, this function returns the next time where a jump occurs
        It should also maybe return some of the points in the jump itself to capture
        the full change in state


        For some reason gets stuck whenever t1 or t2 are float ... Why????
        Is the above still the case?
        Should a torch tensor be returned?
        Use detach() or not?
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

    def check_roads_contained(self, id_1, id_2):
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
            return False
        else:
            return True
        
    def get_activation(self, t, id_1, id_2):
        '''
        Given a time t, this function returns the activation of roads with id_1 and id_2
        If either id_1 or id_2 is not a part of the junction, False, 0.0 is returned

        Should also take into account the density on the outgoing road
        If density higher than ? then bus should not travel past
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
            activation = torch.tensor(1.0)
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
    
    def calculate_activation(self, t):
        '''
        Method for calculating the activations between all possible connections
        between incoming road i and outgoing road j.
        Activation set to 1 for all connections, and index [i,j] is updated if a traffic 
        light specifies the allowed traffic between road i and road j

        Is this possible to do without requiring several for-loops?
        Yes! Each traffic light can return an activation matrix, and these can be summed
        Is this actually faster?
        '''
        # ADDING 0-TENSOR and inplace updating of this tensor - likely okay
        active = torch.ones((self.n,self.m))

        for light in self.trafficlights:
            for i in range(self.n):
                if self.entering[i] in light.entering:
                    for j in range(self.m):
                        if self.leaving[j] in light.leaving:
                            active[i,j] = light.activation_func(t)

        
        for light in self.coupled_trafficlights:
            for i in range(self.n):
                if self.entering[i] in light.a_entering:
                    for j in range(self.m):
                        if self.leaving[j] in light.a_leaving:
                            active[i,j] = light.a_activation(t)

                if self.entering[i] in light.b_entering:
                    for j in range(self.m):
                        if self.leaving[j] in light.b_leaving:
                            active[i,j] = light.b_activation(t)
        return active
    
    def calculate_demand(self, active):
        '''
        Calculate the demand of each connection i,j given the activation
        on each connection determined by the traffic lights
        '''
        # ADDING 0-TENSOR
        demands = torch.zeros((self.n, self.m))
        for i, road in enumerate(self.road_in):
            demand = road.demand()
            # Remove for loop below:
            # for j, alpha in enumerate(self.distribution[i]):
            #     demands[i,j] = active[i,j] * alpha * demand
            demands[i,:] = active[i,:] * self.distribution[i] * demand
        return demands
    
    def calculate_priority_params(self, rho_in, h0=0.6, hmax=0.9, h1=0.6, rho_m=0.6):
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

        for j in range(self.m):
            # Find all legal connections leading to outgoing road j
            non_zero = [i for i in range(self.n) if priority_list[i][j] != 0]
            if len(non_zero) == 0:
                # No legal edges into road j -> This should never happen, maybe throw an error 
                # For now just continue
                continue

            nz_priorities = [priority_list[k][j] for k in non_zero]
            sorted_indexes = [x for _, x in sorted(zip(nz_priorities, non_zero))]

            if len(sorted_indexes) > 1:
                # More than one one legal edge into road j -> need priority parameters
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
        '''
        Function for calculating the combined contribution of two crossing connections
        '''
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
                # Have only calculated explicit solution in the case where both d1 and d2 are smaller than 0.5
                # Expression also depends on which of the two sticks is smaller
                if d1 < d2:
                    return 1 - two_short_explicit(d1, d2)
                else:
                    return 1 - two_short_explicit(d2, d1)
            else:
                return 1 - two_stick_quadrature(d1, d2)
        
    def calculate_xi_n_crossing(self, actual_fluxes, crossing_connections, max_flux_in):
        '''
        Function for calculating the combined contribution of n crossing connections
        Any connections with no flux can be ignored
        If any connections have maximal flux, the combined contribution is equal to 1
        The remaining part of the unit interval is 0 -> set xi = 0
        '''
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
        In the general case, a quadrature rule is used to calculate the combined contribution of crossing connections
        '''
        cloned_fluxes = actual_fluxes.clone()
        if len(crossing_connections) == 1:
            # Only one crossing connection - no integral needed
            xi = 1 - cloned_fluxes[crossing_connections[0][0],crossing_connections[0][1]]/max_flux_in[crossing_connections[0][0]]

        elif len(crossing_connections) == 2:
            # Two crossing connections - might be able to use explicit solution for integral
            xi = self.calculate_xi_two_crossing(cloned_fluxes, crossing_connections, max_flux_in)
        else:
            # Need to use quadrature rule
            xi = self.calculate_xi_n_crossing(cloned_fluxes, crossing_connections, max_flux_in)

        # xi should be somewhere in the interval [0,1]
        epsilon = epsilon * max_flux_in[i]
        return torch.minimum(demand_ij, epsilon + xi*(max_flux_in[i] - epsilon))
    
    def calculate_fluxes(self, demand, capacities, priorities, max_flux_in):
        '''
        Function for calculating (approximating) the flux across a general junction
        '''
        actual_fluxes = torch.zeros((self.n, self.m))

        assigned_fluxes = []
        upper_bounds = demand.clone()
        # ADDING 0-Tensor
        # CLONING
        for n_crossing in range(self.max_crossing_connections + 1):
            # Go through all connections in an increasing order of n_crossings starting with 0
            for i in range(self.n):
                # Go through every incoming road
                for j in range(self.m):
                    # Go through every outgoing road
                    if len(self.crossing_connections[i][j]) == n_crossing and (i,j) not in assigned_fluxes:
                        # Connection has not been assigned a flux, and it has the correct number of crossing connections
                        if len(self.crossing_connections[i][j]) > 0:
                            # There are some crossing connections, and so an upper bound needs to be calculated
                            # Not actually necessary to take in the entire actual_fluxes. Could instead use only the elements of
                            # actual_fluxes corresponding to crossing connections with higher priority
                            upper_bounds[i,j] = self.calculate_upper_bound(actual_fluxes, 
                                                                           self.crossing_connections[i][j],
                                                                           i, max_flux_in, demand[i,j])
                        # Calculate actual flux
                        # upper_bound[i,j] should be less or equal to demand[i,j] and so
                        # it should not be necessary to include upper_bounds[i,j] below
                        # demand_sum = torch.tensor(0.0)
                        # upper_bound_sum = torch.tensor(0.0)
                        # for l in range(self.n):
                        #     if l != i:
                        #         upper_bound_sum += torch.min(upper_bounds[l,j].clone(), demand[l,j].clone())
                        mask = torch.ones(self.n, dtype=torch.bool)
                        mask[i] = False
                        upper_bound_sum = torch.sum(upper_bounds[mask,j])

                        interior_max = capacities[j] - upper_bound_sum
                        supply_max = torch.max(priorities[i][j]*capacities[j], interior_max)
                        # demand_max = torch.min(upper_bounds[i,j].clone(), demand[i,j].clone())
                        # actual_fluxes[i,j] = torch.min(demand_max, supply_max.clone())
                        # actual_fluxes[i,j] = torch.min(upper_bounds[i,j].clone(), supply_max.clone())
                        actual_fluxes[i,j] = torch.minimum(upper_bounds[i,j].clone(), supply_max)
                        assigned_fluxes.append((i,j))

        # Can for loops be removed below - most likely yes!
        # Fluxes on edges i,j should now be calculated
        # Calculate fluxes in and fluxes out
        fluxes_in = [0]*self.n
        fluxes_out = [0]*self.m
        for i in range(self.n):
            # Sum over all connections out of road i
            fluxes_in[i] = torch.sum(actual_fluxes[i])
        for j in range(self.m):
            # Sum over all connections into road j
            fluxes_out[j] = torch.sum(actual_fluxes[:,j])
        return fluxes_in, fluxes_out

    def divide_flux(self, t):
        # 1. Calculate how much of the demand is limited by traffic lights
        # ADDING 0-TENSOR
        activation = self.calculate_activation(t)

        # ADDING 0-TENSOR
        # 2. Calculate the actual desired flux from each road i to road j
        demand = self.calculate_demand(activation)

        # 3. Calculate the capacity of the roads
        # Scaled with the maximum density of the outgoing road
        # LIST OF TENSORS
        capacities = [road.supply() for road in self.road_out]

        # fluxes[i,j] is theself. flux from road i to road j
        # ADDING 0-Tensor
        fluxes = torch.zeros((self.n, self.m))

        # This can likely be done without for loop...
        for j in range(self.m):
            # Update the flux from all roads into road j
            sum_influx = torch.sum(demand[:,j])
            if sum_influx > capacities[j]:
                # If the sum of the fluxes is larger than the capacity, scale down the fluxes
                fluxes[:,j] = demand[:,j] * capacities[j] / sum_influx
            else:
                fluxes[:,j] = demand[:,j]

        # For loops below can likely be removed
        # LIST OF TENSORS
        fluxes_in = [0]*self.n
        fluxes_out = [0]*self.m

        for i in range(self.n):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_in[i] = torch.sum(fluxes[i])


        for j in range(self.m):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_out[j] = torch.sum(fluxes[:,j])
        
        
        return fluxes_in, fluxes_out

    def divide_flux_right_of_way(self, t):
        '''
        Calculate how the flux is being distributed across a junction when there is merging or crossing traffic
        In this case we need a priority function to determine the percentage of traffic going to the different outgoing roads
        Each junction has n incoming roads and m outgoing roads. 

        Whenever there is more than one outgoing road, a distribution matrix needs to be specified
        Whenever there is more than one incoming road, some priority list needs to be specified.
        This prirority list and the densities of the incoming roads will be used to find prirotiy
        parameters.
        '''

        # LIST OF TENSORS
        # rho_in = [road.rho[-road.pad] for road in self.road_in]
        rho_in = [road.rho[-road.pad].clone() for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        # max_flux_in = [fv.fmax(gamma) for gamma in gamma_in] # not multiplied with max_dens!
        max_flux_in = [fv.fmax(gamma)*road.max_dens for gamma,road in zip(gamma_in, self.road_in)]

        # Steps of the algorithm:
        # 1. Calculate how much of the demand is limited by traffic lights
        activation = self.calculate_activation(t)

        # 2. Calculate the actual desired flux from each road i to road j
        demand = self.calculate_demand(activation)

        # 3. Calculate the capacity of each road j
        # Scaled with the maximum density of the outgoing road
        capacities = [road.supply() for road in self.road_out]

        # 4. Determine a set of prirority parameters for each outgoing road j
        # This needs to be permuted back to original order!!
        priorities = self.calculate_priority_params(rho_in)
        
        # 5. Determine the actual flux between road i and road j:
        # 5.1 Start with roads with no crossing connections
        # 5.2 For roads with one crossing connection calculate upper bound
        # 5.3 Calculate actual flux from road i to road j with one crossing connection
        # 5.4 Repeat for roads with more crossing connections
        # 6. Sum over fluxes from i to j to get fluxes in and fluxes out
        fluxes_in, fluxes_out = self.calculate_fluxes(demand, capacities, priorities, max_flux_in)

        return fluxes_in, fluxes_out

    def apply_bc(self, dt, t):
        '''
        To save time, assume that the flux is distributed equally among roads
        Could alternatively make the user specify the distribution
        '''

        if self.duty_to_gw:
            # fluxes_in, fluxes_out = self.divide_flux_wo_opt_duty_to_gw(t)
            fluxes_in, fluxes_out = self.divide_flux_right_of_way(t)
        else:
            fluxes_in, fluxes_out = self.divide_flux(t)

        # Here, for loops are necessary
        # LIST OF TENSORS
        for i, road in enumerate(self.road_in):
            road.update_right_boundary(fluxes_in[i] / road.max_dens, dt, t)
        
        for j, road in enumerate(self.road_out):
            road.update_left_boundary(fluxes_out[j] / road.max_dens, dt, t)
    
    def get_speed(self, t, id_1, id_2):
        '''
        Need to recalculate how the flux is divided among the roads
        -> Change this to avoid recalculations...
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
        
        # rho_in = [road.rho[-road.pad] for road in self.road_in]
        rho_in = [road.rho[-road.pad].clone() for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        max_dens_in = torch.tensor([road.max_dens for road in self.road_in])
        # rho_out = [road.rho[road.pad-1] for road in self.road_out]
        rho_out = [road.rho[road.pad-1].clone() for road in self.road_out]
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
    