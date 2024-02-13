import torch
import initial_and_bc as ibc
import FV_schemes as fv
import matplotlib.pyplot as plt
import networkx as nx
# import sys
# import json

# def maximize_flux(roads, entering, leaving):
#     '''
#     Sets condition on flux in and out of system by maximizing the total
#     flux through junction
#     '''
#     road_in = [roads[i] for i in entering]
#     road_out = [roads[i] for i in leaving]
#     # print("Pad and rho at end for debugging purposes")
#     # print([road.pad for road in road_in])
#     # print([road.rho[-road.pad-1:] for road in road_in])
#     rho_in = [road.rho[-road.pad] for road in road_in]
#     V_max_in = [road.Vmax for road in road_in]
#     rho_out = [road.rho[road.pad-1] for road in road_out]
#     V_max_out = [road.Vmax for road in road_out]
#     _, _, fluxes, _ = of.find_parameters(rho_in, rho_out, V_max_in, V_max_out)

#     n = len(rho_in)
#     m = len(rho_out)

#     flux_in = [sum([fluxes[i][j] for j in range(m)]) for i in range(n)]
#     # print("Flux in")
#     # print(flux_in)
#     flux_out = [sum([fluxes[i][j] for i in range(n)]) for j in range(m)]
#     # print("Flux Out")
#     # print(flux_out)
#     return flux_in, flux_out

class Road:
    '''
    Single laned road going from a to b
    N is the number of internal nodes used in the FV scheme
    rho is density of cars
    Vmax: speed limit on road
    scheme: different FV schemes
        - 0: Lax-Friedrich
        - 1: Rusanov
        - 2: Lax-Wendroff
        - 3: 2. order scheme with Rusanov flux and SSP RK for updating
        - 4: 2. order in space scheme with Rusanov flux Euler updating

    Need to somehow specify boundary conditions
        - Can maximize flux across junction using distribution and priority parameters

    a = 0 for all roads,
    Length of road is a part of scaling parameter alpha
    To ensure that condition over junction makes sense, maybe make sure L is the same for
    all roads, and that dx is the same for roads.
    Let x go from 0 to b for some b not equal in general for all roads

    At the very least, make sure the above are same for roads in junction
    '''
    b = None
    # N = None
    dx = None
    rho = None
    L = None # Should be the same for all roads in system - instead let x go from 0 to a
    Vmax = None
    gamma = None # Will be updated when simulation time is set
    scheme = None
    pad = None
    # N_full = None
    limiter = None
    left = False
    right = False
    inflow = None


    def __init__(self, b, L, N, Vmax = 1., scheme = 3, limiter = "minmod", initial = lambda x: torch.zeros_like(x), inflow = -1):
        '''
        Initializes a road given a set of parameters
        
        Input:
            L:          Length unit constant for all roads in system
            N:          Number of internal nodes for a length unit - constant for all roads
            b:          Number of length units road consists of
            Vmax:       Speed limit of road
            Scheme:     Which FV scheme to be used in simulation. 0 correspond to Lax Friedrich,
                        1 - Rusanov scheme, 2 - Lax-Wendroff, 3 - SSP_RK (2.order in time and space)
                        4 - Euler (2.order in space, 1.order in time)
            Limiter:    If the scheme is 2.order in space -> use a piecewise linear instead of 
                        piecewise constant reconstruction. Limiter defines how reconstruction is
                        built
            initial:    Initial density of cars
            inflow:     BC on left edge of roads leading in to system

        '''
        self.b = b
        self.L = L 
        self.dx = 1 / (N + 1) # Should be the same for all roads
        # Denominator = 1 since x goes from 0 to b (0 to 1 on a unit length)

        # Vmax is parameter that optimization should done wrt.
        # Therefore we set requires_grad to True
        self.Vmax = torch.tensor([float(Vmax / 3.6)], requires_grad=True)
    
        # Depending on the order of scheme, the padding might differ
        match scheme:
            case 0 | 1 | 2:
                self.pad = 1
                self.scheme = scheme
            case 3 | 4:
                self.pad = 2
                self.scheme = scheme
            case _:
                self.pad = 2
                self.scheme = 3

        # When order of scheme is determined, the grid can be defined
        # For each length unit we have N internal nodes
        # We also have 2*self.pad boundary nodes.
        # Finally we have b-1 connecting nodes between each length unit of road

        # In total we have
        N_full = N * b + 2*self.pad + (b-1)

        j = torch.linspace(-(self.pad-1), b*(N+1) + (self.pad-1), N_full)
        # If first order, x goes from 0 to bL
        # If higher order x goes from -dx * order to bL + dx*order
        # Defining the system in such a way means that all internal points in actual
        # road correspond to internal nodes in FV scheme
        x =  (j + 1/2) * self.dx 

        # Determine initial denisty according to some density function
        # Note that x < 0 at some points for higher order schemes
        self.rho = initial(x)

        # limiter is used if the scheme is of second order
        self.limiter = limiter
        # Inflow determines boundary condition in to road
        self.inflow = inflow

        # For now it is assumed that all flux is allowed to exit road
        # Should maybe extend to limit the allowed flux that can exit road

    def calculate_gamma(self, T):
        '''
        Calculate dimensionless parameter gamma using total 
        time of simulation
        This parameter does not depend on the length of the road, only the length of a 
        unit. 
        This is useful for determining the flux across a junction
        '''
        #self.gamma = self.Vmax * T / self.L 
        self.gamma = self.Vmax / self.L 


    def max_dt(self):
        '''
        Calculate the largest allowed timestep for a FV scheme on this road according
        to some CFL condition

        CFL condition from Mishra
        During analysis 0.5 was used in inequality, but it was claimed that also 1
        is fine
        -> Use 1 in ineq unless otherwise specified
        '''
        strict = False
        if strict:
            CFL = 0.5
        else:
            CFL = 1. 
        
        # Must reformulate this condition with scaled equation
        return CFL * self.dx / (torch.max(torch.tensor([torch.abs(fv.d_flux(self.rho[j], self.gamma)) for j in range(len(self.rho))])))

    def solve_internally(self, dt):
        # print("Solving internally")
        # print(self.scheme)
        # Solve conservation law for internal nodes on road
        #print(self.rho, self.dx, self.limiter, self.Vmax, dt)
        # Check if scheme is 1. or 2. order
        match self.scheme:
            case 0:
                # Lax-Friedrich scheme
                F = fv.LxF_flux(self.rho, self.dx, dt, self.gamma)
                # print("Internal Flux")
                # print(F)
                self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
            case 1:
                # Rusanov scheme
                F = fv.Rusanov_Flux(self.rho, self.gamma)
                # print("Internal Flux")
                # print(F)
                self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
            case 2:
                F = fv.Lax_Wendroff_Flux(self.rho, self.dx, dt, self.gamma)
                # print("Internal Flux")
                # print(F)
                self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
            case 3:
                # 2. order in time and space
                ###########
                # SSP_RK and Euler must be redefined to take in gamma instead of vmax
                ###########
                self.rho = fv.SSP_RK(self.rho, self.dx, self.limiter, dt, self.gamma)
            case 4:
                # 2. order in space, 1. order in time
                self.rho = fv.Euler(self.rho, self.dx, self.limiter, dt, self.gamma)

    def apply_bc(self, t):
        # Apply boundary conditions to edges not attached to any junction

        # For now set right boundary elements equal to closest interior
        if not self.right:
            # Right boundary not attached to junction
            self.rho[-self.pad:] = self.rho[-self.pad-1]
        
        if not self.left:
            # Set some influx to left boundary
            if self.inflow < 0:
                inflow = max(0, torch.sin(torch.tensor(2*torch.pi*t)))
            else:
                inflow = self.inflow
            self.rho[:self.pad] = inflow

class RedLight:
    '''
    Should probably add coupling between red lights
    I.e. if one light turns red, the other turns green with some delay
    '''
    entering = []
    leaving = []
    stop_times = [] # On the form [[start_1, end1], [start2, end2], ...]
    # If t in [start_i, end_i] red light is active

    def __init__(self, entering, leaving, stop_times):
        # Make sure roads are either entering or leaving
        assert set(entering).isdisjoint(leaving)
        self.entering = entering
        self.leaving = leaving
        self.stop_times = stop_times

    def active(self, t):
        red = False 
        for interval in self.stop_times:
            if t >= interval[0] and t <= interval[1]:
                red = True
                break
        return red
    
class CoupledRedLight:
    '''
    Class for coupled redlights
    Should contain several redlights, and their relation to eachother
    '''
    on_redlights = []
    off_redlights = []
    stop_times = []

    def __init__(self):
        pass
    
    # Spend some time figuring out how to set this class up, and how it should be used 
    # in the simulation


class Junction:
    # Allow for roads to have different flux function
    roads = None # All roads that pass through Junction
    entering = None # Index of roads going in
    leaving = None # Index of roads going out
    priority = None # Assume priority equal for all roads
    distribution = None # Assume all roads have same distribution

    redlights = None # Define all redlights in junction

    def __init__(self, roads, entering, leaving, distribution, redlights):
        # Make sure roads are either entering or leaving
        assert set(entering).isdisjoint(leaving)
        # Make sure all roads actually cross junction
        assert len(roads) == len(entering) + len(leaving)
        # Make sure distribution is of correct dimension
        assert len(distribution) == len(leaving)
        # Make sure distribution sums to 1
        assert abs(sum(distribution) -1 ) <= 1e-4

        for redlight in redlights:
            # Check that redlight only contains roads in junction
            assert set(redlight.entering).issubset(set(entering))
            assert set(redlight.leaving).issubset(set(leaving))

        self.roads = roads
        self.entering = entering
        for i in self.entering:
            self.roads[i].right = True

        self.leaving = leaving
        for j in self.leaving:
            self.roads[j].left = True

        self.distribution = distribution
        self.priority = [1/len(entering)] * len(entering)

        self.redlights = redlights
    
    
    def divide_flux(self, active_redlights):
        '''
        When comparing the fluxes from different roads, use 
        gamma * f(rho) instead of f(rho)
        '''
        road_in = [self.roads[i] for i in self.entering]
        road_out = [self.roads[i] for i in self.leaving]

        rho_in = [road.rho[-road.pad] for road in road_in]
        gamma_in = [road.gamma for road in road_in]
        rho_out = [road.rho[road.pad-1] for road in road_out]
        gamma_out = [road.gamma for road in road_out]

        n = len(self.entering)
        m = len(self.leaving)

        active_idx = torch.zeros((n,m))
        for redlight in active_redlights:
            for i in range(n):
                for j in range(m):
                    if self.entering[i] in redlight.entering and self.leaving[j] in redlight.leaving:
                        active_idx[i,j] = 1

        fluxes = torch.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if not active_idx[i,j]:
                    fluxes[i,j] = min(self.distribution[j] * fv.D(rho_in[i].clone(), gamma_in[i]),
                                    self.priority[i]* fv.S(rho_out[j].clone(),  gamma_out[j]))
                
                
        fluxes_in = [0]*n
        fluxes_out = [0]*m

        for i in range(n):
            fluxes_in[i] = sum([fluxes[i,j] for j in range(m)])

        for j in range(m):
            fluxes_out[j] = sum([fluxes[i,j] for i in range(n)])

        # print(fluxes_in)
        # print(fluxes_out)
        
        return fluxes_in, fluxes_out


    def apply_bc(self, dt, t):
        '''
        Calculate how flux is divided among roads
        To this end gamma of each road is important
        Actual flux is gamma*f(rho) instad of just f(rho)
        .....
        '''
        # Needed to add clone when calculating flux - is this correct???


        road_in = [self.roads[i] for i in self.entering]
        road_out = [self.roads[i] for i in self.leaving]

        active_redlights = []
        for redlight in self.redlights:
            if redlight.active(t):
                active_redlights.append(redlight)
        
        #--------------------------------
        # Dividing flux need to somehow take into account gamma of each road
        # --------------------------------
        fluxes_in, fluxes_out = self.divide_flux(active_redlights)
        # outputed fluxes should now be gamma*f(rho)


        #---------------------------------------------
        # Note not exactly correct: using time tn+1 to do updating, but should actually use
        # time tn
        # Maybe not a very big problem ????
        # Solution: save out/in flux at previous step somewhere
        #           not very difficult so should maybe just do this
        #---------------------------------------------


        for i, flux in enumerate(fluxes_in):
            road = road_in[i]
            left, in_mid = road.rho[-road.pad-1], road.rho[-road.pad]
            s = torch.max(torch.abs(fv.d_flux(left, road.gamma)), torch.abs(fv.d_flux(in_mid, road.gamma)))
            left_f = fv.flux(left.clone(), road.gamma)
            mid_f = fv.flux(in_mid.clone(), road.gamma)
            left_flux = 0.5 * (left_f + mid_f) - 0.5 * s * (in_mid - left)

            # Don't multiply with gamma in denominator because flux is already multiplied with
            # gamma
            road.rho[-road.pad] = road.rho[-road.pad] - dt/road.dx * (flux - left_flux)
            if road.pad > 1:
                road.rho[-road.pad+1] = road.rho[-road.pad]
        
        for i, flux in enumerate(fluxes_out):
            road = road_out[i]

            right, out_mid = road.rho[road.pad], road.rho[road.pad-1]
            s = torch.max(torch.abs(fv.d_flux(out_mid, road.gamma)), torch.abs(fv.d_flux(right, road.gamma)))
            mid_f = fv.flux(out_mid.clone(), road.gamma)
            right_f = fv.flux(right.clone(), road.gamma)
            right_flux = 0.5 * (mid_f + right_f) - 0.5 * s * (right - out_mid)
            #print(right_flux)
            #print(right_flux - flux)

            road.rho[road.pad-1] = road.rho[road.pad-1] - dt / road.dx * (right_flux - flux)
            if road.pad > 1:
                road.rho[0] = road.rho[1]
            #road.rho = (0.05 - road.rho) * road.Vmax**2 + 0.01 * fluxes_out[0]


                
class RoadNetwork:

    # Roads have one or two ends in an junction
    # Intersection class should deal with boundary conditions on junctions
    # Ends not connected to junction need other boundary conditions
    # This should be dealt with here

    '''
    FV scheme:
        Keep track of which ends are connected to intersections and which are not
        Solve equation separately on each road, allowing for different flux functions
        and schemes
        Apply boundary conditions to all intersections
            Rankine Hugoniot condition of flux
            Maximise flux?
        Apply boundary conditions to all ends not connected to intersections
    '''


    roads = None # All roads in network
    junctions = None # All junctions in network
    debugging = None
    object_type = None
    iters = None
    T = None

    def __init__(self, roads, junctions, T, debugging = False, object_type = 0, iters = 1):
        # Check that unit length and dx are equal for all roads
        for i in range(1, len(roads)):
            assert roads[i].L == roads[i-1].L 
            assert roads[i].dx == roads[i-1].dx

        self.roads = roads
        self.junctions = junctions
        self.debugging = debugging
        self.object_type = object_type
        self.iters = iters
        self.T = T

        # Update gamma for each road
        for road in self.roads:
            road.calculate_gamma(self.T)

    def draw_network(self):
        # Function to draw a graph of the network to check that it is properly set up

        # Create an empty graph
        G = nx.Graph()

        # Create all necessary nodes
        for i in range(len(self.junctions)):
            G.add_node(f"J {i+1}")

        for i in range(len(self.roads)):
            if not self.roads[i].left:
                # Need to add left edge
                G.add_node(f"Left {i+1}")

            if not self.roads[i].right:
                # Need to add right edge
                G.add_node(f"Right {i+1}")
            
        # Create all necessary edges
        for i in range(len(self.roads)):
            # Either from left end to junction, 
            # junction to right end
            # or junction to junction
            # left to right only if only one road

            if not self.roads[i].left:
                # Left side not connected to anything
                for j in range(len(self.junctions)):
                    if self.roads[i] in self.junctions[j].roads:
                        G.add_edge(f"Left {i+1}", f"J {j+1}")
                        #break
                
                if not self.roads[i].right:
                    # This should only be the case when network only consists of single road
                    G.add_edge(f"Left {i+1}", f"Right {i+1}")
                
            elif not self.roads[i].right:
                # Left side not connected to anything
                for j in range(len(self.junctions)):
                    if self.roads[i] in self.junctions[j].roads:
                        G.add_edge(f"Right {i+1}", f"J {j+1}")
                        #break
            
            else:
                # Road connected to two junctions
                for j in range(len(self.junctions)):
                    if self.roads[i] in self.junctions[j].roads:
                        # One side connected to junction j
                        for k in range(j+1, len(self.junctions)):
                            if self.roads[i] in self.junctions[k].roads:
                                # Other side connected to junction k
                                G.add_edge(f"J {j+1}", f"J {k+1}")
                                #break

        nx.draw(G, with_labels=True)
        plt.margins(0.2)
        plt.show()


    def solve_cons_law(self):
        '''
        Takes in a road network consisting of roads and junctions.
        Each road defines has its own numerical scheme limiter if second order and speed limit.
        

        Later add possibility of having different flux functions
        Can also add different flux functions for each road

        Solves model of road untill time T
        '''
        
        t = 0
        rho_timesteps = {i : {0 : self.roads[i].rho} for i in range(len(self.roads))}

        i_count = 0
        while t < self.T:
            # Iterate untill time limit is reached

            #-------------------------------------
            # STEP 1: Find appropriate timestep
            #-------------------------------------
            dt = torch.tensor(self.T) - t
            for road in self.roads:
                dt = torch.min(dt, road.max_dt())
                #dt = road.max_dt()
            #dt = self.roads[0].dx / self.roads[0].Vmax
            
        
            t = t + dt

            #-------------------------------------
            # STEP 2: Solve internal system for each road
            #-------------------------------------
            for road in self.roads:
                # Solve internally on all roads in network
                #print(dt)
                road.solve_internally(dt)

            #-------------------------------------
            # STEP 3: Apply flux conditions for each Junction
            #-------------------------------------
            for J in self.junctions:
                # Apply boundary conditions to all junctions
                J.apply_bc(dt, t)
                
            # Vmax seems to be destroyed at this point
            # Why???
            # self.roads[0].rho = self.roads[0].rho * self.roads[0].Vmax**3


            #-------------------------------------
            # STEP 4: Apply BC to roads with one or more edges not connected to junction
            #-------------------------------------
            for road in self.roads:
                # Add boundary conditions to remaining roads
                road.apply_bc(t)

            #-------------------------------------
            # STEP 5: Store solution after time t
            #-------------------------------------
            for i in range(len(self.roads)):
                rho_timesteps[i][t] = self.roads[i].rho

            if self.debugging:
                i_count += 1
                if i_count >= self.iters:
                    # print(f"Last t: {t}")
                    t = self.T+1
            # if self.debugging:
            #     t = T+1 # Only do 1 iteration for debugging purposes
        #-------------------------------------
        # STEP 6: Calculate objective function and its derivatives
        #-------------------------------------
        history_of_network = rho_timesteps
        return history_of_network
    
    # def get_dummy_objective(self, type = 0):
    #     match type:
    #         case 0:
    #             torch.autograd.set_detect_anomaly(True)
    #             # Do one iteration of scheme
    #             self.debugging = True
    #             self.iters = 10
    #             history = self.solve_cons_law(10.)
    #             self.debugging = False

    #             tot_out = self.roads[0].rho[40]*self.roads[0].Vmax
    #             print(self.roads[0].rho[40])
    #             # for road in self.roads:
    #             #     tot_out = tot_out + road.rho[20]*road.Vmax
                
    #             tot_out.backward()
    #             for road in self.roads:
    #                 print(road.Vmax.grad)


    #             n = 50 + 2*2 # Add padding required from order of scheme to total number of nodes
    #             xL, xR = [-1,1]
    #             dx = (xR - xL) / (n + 1)
    #             j = torch.linspace(0, n, n+1)
    #             x = xL + (j + 1/2) * dx

    #             for i in range(len(history)):
    #                 full_array = []
    #                 for key in history[i].keys():
    #                     #print(key)
    #                     full_array.append(history[i][key].detach()[2:-2])

    #                 fig = plt.pcolor(x[2:-2], history[i].keys(), full_array, cmap="gray")
    #                 plt.colorbar(fig)
    #                 plt.title("Single lane")
    #                 plt.ylabel("Time")
    #                 plt.show()

    #         case 1:
    #             history = self.solve_cons_law(10.)
    #             tot_out = self.roads[0].rho[30]#*self.roads[0].Vmax
    #             print(self.roads[0].rho[30])
    #             for road in self.roads:
    #                 first_derivative = torch.autograd.grad(tot_out, road.Vmax, create_graph=True)[0]
    #                 # We now have dloss/dx
    #                 second_derivative = torch.autograd.grad(first_derivative, road.Vmax)[0]
    #                 print(first_derivative)
    #                 print(second_derivative)


    #                 # Performing Newton optimization
    #                 # Next guess=
    #                 print("Next speed limit")
    #                 print(road.Vmax - first_derivative/second_derivative)


if __name__ == "__main__":
    # import json
    # print("Running")
    # f = open("configs/single_lane.json")

    # data = json.load(f)
    # f.close()
    # print(data)
    # T = data["T"]
    # for road in data["roads"]:
    #     print(f"L = {road['L']}")
    #     print(road["Vmax"])
    #     print(road["N"])
    #     print(road["Scheme"])
    t = 1 # Will only do one iteration anyways so doesn't matter
    l = 1 # So that gamma = vmax
    vmax = 10 # Not very important
    
    init_func = lambda x : torch.linspace(0,0.5,len(x))
    road = Road(b=1, L = l, N = 3, Vmax=float(vmax), scheme =4,
                initial=init_func, inflow = 0.25)
    print(road.dx)
    network = RoadNetwork([road], [], t, debugging=True, iters=1)
    history = network.solve_cons_law()
   
    objective =  network.roads[0].rho[3]  * network.roads[0].Vmax
    print(f"Objective value : {objective}")
    first =  torch.autograd.grad(objective, network.roads[0].Vmax , create_graph=True, allow_unused=True)[0]
    second = torch.autograd.grad(first, network.roads[0].Vmax , create_graph=True, allow_unused=True)[0]
    times = list(history[0].keys())
    print(f"Initial density: {history[0][0]}")
    print(f"Internal nodes after one iteration {history[0][times[1]][2:5]}")
    print(f"Speed at middle node {objective}")
    print(f"First derivative of speed at middle node wrt speed limit {first}")
    print(f"Second derivative of speed at middle node wrt speed limit {second}")





    