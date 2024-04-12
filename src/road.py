import torch
# import numpy as np
import FV_schemes as fv

# 4 styles used:
# LIST OF TENSORS
# CLONING
# ADDING 0-TENSOR
# INPLACE 

class Road:
    '''
    Single laned road going from a to b

    ...
    Attributes
    ----------
    b : float
        length of road given as multiple of length L
    dx : float
        length of each cell in finite volume scheme
    rho : torch tensor
        density of road evaluated over each cell in grid
    L : float
        unit length of road given in meters
    Vmax : torch tensor
        speed limit of road, potentially 
        parameter to do optimization over
    gamma :
    scheme :
    pad :
    limiter :
    left :
    right :
    inflow :
    queue_length :
    control_points :
        fixed for each road - can potentially be made into a parameter to do optimization over

    
    scheme: different FV schemes
        - 0: Lax-Friedrich
        - 1: Rusanov
        - 2: Lax-Wendroff
        - 3: 2. order scheme with Rusanov flux and SSP RK for updating - this is the one being used
        - 4: 2. order in space scheme with Rusanov flux Euler updating

    Need to somehow specify boundary conditions
        - Calculate flux using distribution and priority parameters
        - 

    a = 0 for all roads,
    Length of road is a part of scaling parameter alpha
    To ensure that condition over junction makes sense, maybe make sure L is the same for
    all roads, and that dx is the same for roads.
    Let x go from 0 to b for some b not equal in general for all roads

    At the very least, make sure the above are same for roads in junction




    Potentially change to make user specify coordinates of start/end of road
    When roads meet in a junction, make sure the coordinates line up
    '''


    b = None
    dx = None
    rho = None
    L = None # Should be the same for all roads in system - instead let x go from 0 to a
    Vmax = None # Now a vector of speed limits at some control times
    gamma = None # Will be updated when simulation time is set
    scheme = None
    pad = None
    limiter = None
    left = False
    right = False
    queue_length = None
    control_points = None
    index = None # Index of control point being evaluated, needs to be evaluated for each road at each time
    max_dens = None
    left_pos = None
    right_pos = None
    periodic = False
    id = ""
    boundary_fnc = None

    def __init__(self, b, L, N, Vmax, control_points, scheme = 3, limiter = "minmod", 
                 initial = lambda x: torch.zeros_like(x), max_dens=1, left_pos = (-1,0), 
                 right_pos = (0,1), boundary_fnc = None, periodic = False, id = ""):
        
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
        self.dx = torch.tensor(1 / N) # Should be the same for all roads
        # Denominator = 1 since x goes from 0 to b (0 to 1 on a unit length)

        # Vmax is parameter that optimization should done wrt.
        # Therefore we set requires_grad to True
        if len(Vmax) != len(control_points) +1:
            print(len(Vmax))
            print(len(control_points))
        assert(len(Vmax) == len(control_points)+1)
        self.Vmax = [torch.tensor(float(v / 3.6), requires_grad=True) for v in Vmax]
        self.control_points = control_points
    
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
        self.N_internal = N*b
        N_full = self.N_internal +  2*self.pad
        # Maybe not 100% correct, but very close...
        j = torch.arange(-self.pad, self.N_internal + self.pad, 1)
        # If first order, x goes from -.dx to bL + dx
        # If higher order x goes from -dx*order * order to bL + dx*order
        # Defining the system in such a way means that all internal points in actual
        # road correspond to internal nodes in FV scheme
        x =  (j + 1/2) * self.dx 
        
        # Determine initial denisty according to some density function
        # Note that x < 0 at some points for higher order schemes
        self.rho = initial(x)

        # limiter is used if the scheme is of second order
        match limiter:
            case "minmod":
                self.limiter = torch.tensor(1.0)
            case "maxmod":
                self.limiter = torch.tensor(2.0)
            case "superbee":
                self.limiter = torch.tensor(3.0)
        # Inflow determines boundary condition in to road

        # For now it is assumed that all flux is allowed to exit road
        # Should maybe extend to limit the allowed flux that can exit road

        self.queue_length = torch.tensor(0.0)
        self.index = 0
        self.max_dens = max_dens

        self.left_pos = left_pos 
        self.right_pos = right_pos

        self.periodic = periodic
        self.id = id

        self.boundary_fnc = boundary_fnc
        

    def calculate_gamma(self, T):
        '''
        Calculate dimensionless parameter gamma using total 
        time of simulation
        This parameter does not depend on the length of the road, only the length of a 
        unit. 
        This is useful for determining the flux across a junction
        For now T is not in the variable
        Instead let t go from 0 to T to easier compare with traffic ligths
        '''
        # LIST OF TENSORS
        self.gamma = [v / self.L for v in self.Vmax]

    def demand(self):
        # clone needed?
        # CLONING
        return self.max_dens * fv.D(self.rho[-1].clone(), self.gamma[self.idx])

    def supply(self):
        # Clone needed?
        # CLONING
        return self.max_dens * fv.S(self.rho[-1].clone(), self.gamma[self.idx])
    
    def update_right_boundary(self, incoming_flux, dt):
        # left is internal cell, middle is first boundary cell
        left, middle = self.rho[-self.pad-1], self.rho[-self.pad]
        # Calculate rusanov flux:
        left_flux = fv.Rusanov_Flux_2(left.clone(), middle.clone(), self.gamma[self.idx])

        # Update density on boundary cell(s)
        # Inplace operation below!
        self.rho[-self.pad] = self.rho[-self.pad] - dt/self.dx * (incoming_flux - left_flux)
        if self.pad > 1:
            # More than one boundary cell -> put all other boundary cells equal to this one
            for i in range(self.pad - 1):
                self.rho[-self.pad+i+1] = self.rho[-self.pad]
        
    def update_left_boundary(self, outgoing_flux, dt):
        # right is internal cell, middle is first boundary cell
        right, middle = self.rho[self.pad], self.rho[self.pad-1]
        # Calculate Rusanov flux:
        right_flux = fv.Rusanov_Flux_2(middle.clone(), right.clone(), self.gamma[self.idx])

        # Update density on boundary cell(s)
        # Inplace operation!
        self.rho[self.pad-1] = self.rho[self.pad-1] - dt/self.dx * (right_flux - outgoing_flux)
        if self.pad > 1:
            for i in range(self.pad-1):
                self.rho[self.pad-2-i] = self.rho[self.pad-1]

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
        
        # Max density must appear also here
        
        # Should remove list comprehension here...
        
        #return CFL * self.dx / (self.max_dens * torch.max(torch.tensor([torch.abs(fv.d_flux(self.rho[j], self.gamma[self.idx])) for j in range(len(self.rho))])))
            
        # New attempt
        max_flux = torch.abs(fv.d_flux(self.rho, self.gamma[self.idx]))
        max_flux = torch.max(max_flux)
        # return CFL * self.dx / (max_flux)
        
        return CFL * self.dx / (self.max_dens * max_flux) # Should max_dens really be here?

    def solve_internally(self, dt):#, slowdown_factors):
        '''
        Solving conservation law for internal
        Slowdown_factors is a number for each interface that says how the flux should be
        reduced for each interface
        First iteration: Only add to the SSP_RK function
        '''
        # print("Solving internally")
        # print(self.scheme)
        # Solve conservation law for internal nodes on road
        #print(self.rho, self.dx, self.limiter, self.Vmax, dt)
        # Check if scheme is 1. or 2. order'
        # idx is the index currently being used
        # idx is determined in network method

        # Don't need to care about the maximum density here, since all terms are multiplied by it

        match self.scheme:
            # INPLACE 
            case 0:
                # Lax-Friedrich scheme
                F = fv.LxF_flux(self.rho, self.dx, dt, self.gamma[self.idx])
                self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
            case 1:
                # Rusanov scheme
                F = fv.Rusanov_Flux(self.rho, self.gamma[self.idx])
                self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
            case 2:
                F = fv.Lax_Wendroff_Flux(self.rho, self.dx, dt, self.gamma[self.idx])
                self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
            case 3:
                # self.rho = fv.SSP_RK_slowdown(self.rho, self.dx, self.limiter, dt, self.gamma[self.idx],
                #                               slowdown_factors)
                self.rho = fv.SSP_RK(self.rho, self.dx, self.limiter, dt, self.gamma[self.idx])
            case 4:
                # self.rho = fv.Euler_slowdown(self.rho, self.dx, self.limiter, dt, self.gamma[self.idx],
                #                              slowdown_factors)
                self.rho = fv.Euler(self.rho, self.dx, self.limiter, dt, self.gamma[self.idx])
    
    def solve_internally_slowdown(self, dt, slowdown_factors):
        '''
        Solving conservation law for internal
        Slowdown_factors is a number for each interface that says how the flux should be
        reduced for each interface
        First iteration: Only add to the SSP_RK function
        '''
        # print("Solving internally")
        # print(self.scheme)
        # Solve conservation law for internal nodes on road
        #print(self.rho, self.dx, self.limiter, self.Vmax, dt)
        # Check if scheme is 1. or 2. order'
        # idx is the index currently being used
        # idx is determined in network method

        # Don't need to care about the maximum density here, since all terms are multiplied by it

        match self.scheme:
            # INPLACE 
            case 0:
                # Lax-Friedrich scheme
                F = fv.LxF_flux(self.rho, self.dx, dt, self.gamma[self.idx])
                self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
            case 1:
                # Rusanov scheme
                F = fv.Rusanov_Flux(self.rho, self.gamma[self.idx])
                self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
            case 2:
                F = fv.Lax_Wendroff_Flux(self.rho, self.dx, dt, self.gamma[self.idx])
                self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
            case 3:
                self.rho = fv.SSP_RK_slowdown(self.rho, self.dx, self.limiter, dt, self.gamma[self.idx],
                                              slowdown_factors)
            case 4:
                self.rho = fv.Euler_slowdown(self.rho, self.dx, self.limiter, dt, self.gamma[self.idx],
                                             slowdown_factors)

    def apply_bc(self, dt, t):
        # Change structure of road to instead have a boundary condition function and use
        # this function to update the road.

        # Apply boundary conditions to edges not attached to any junction
        # If condition becomes on flux in then max dens might be necessary also here


        if self.periodic:
            # Set periodic boundary conditions
            match self.pad:
                # INPLACE 
                # CLONING
                case 1:
                    # In condition:
                    self.rho[0] = self.rho[-2].clone()

                    # Out condition
                    self.rho[-1] = self.rho[1].clone()   
                case _:
                    # In condition:
                    self.rho[0] = self.rho[-4].clone()
                    self.rho[1] = self.rho[-3].clone()

                    # Out condition
                    self.rho[-1] = self.rho[3].clone()
                    self.rho[-2] = self.rho[2].clone()

        else:
            # For right boundary, set boundary elements equal to closest interior
            # This means all flux is allowed to exit
            if not self.right:
                # INPLACR
                # Right boundary not attached to junction
                self.rho[-self.pad:] = self.rho[-self.pad-1]

            # For left boundary some inflow conditions are necessary       
            if not self.left:
                # Set some influx to left boundary
                if self.boundary_fnc is None:
                    raise ValueError(f"No boundary function specified for road {self.id}!")

                else:
                    f_in =  self.boundary_fnc(t)

                    if self.queue_length > 0:
                        # Set the influx to the maximum possible
                        D = fv.flux(torch.tensor(0.5), self.gamma[self.idx].clone())
                    else:
                        # Set the influx to the mimum of actual and maximum
                        D = torch.min(f_in, fv.flux(torch.tensor(0.5), self.gamma[self.idx].clone()))
                
                        # Set influx to the mimum of actual influx and maximum capacity
                        gamma_in = torch.min(D, fv.S(self.rho[self.pad-1].clone(), self.gamma[self.idx].clone())) # Actual flux in

                        # Update queue length using the difference between actual and desired flux in
                        # Potential problem: Queue length could become negative!!!
                        self.queue_length = self.queue_length + dt * (f_in - gamma_in)
                        
                        # Update density according to flux in
                        self.update_left_boundary(gamma_in, dt)

    def update_index(self, t):
        '''
        Find out which time interval we are in given time t. The time interval gives the
        index of which speed limit should be used
        Return time of next control point to get upper limit of how big dt can be
        '''
        if len(self.control_points) == 0:
            self.idx = 0
            return -1
        
        if t < self.control_points[0]:
            # First time interval
            self.idx = 0
            return self.control_points[0]
        
        if t >= self.control_points[-1]:
            # Last time interval
            self.idx = len(self.control_points)
            return -1 # -1 will correspond to time T
        
        for i in range(1,len(self.control_points)):
            if self.control_points[i-1] <= t and self.control_points[i] > t:
                # (i+1)th time interval
                self.idx = i
                return self.control_points[i]


    def get_gamma(self, t):
        if len(self.control_points) == 0:
            return self.gamma[-1]
        
        if t < self.control_points[0]:
            # First time interval
            return self.gamma[0]
        
        if t >= self.control_points[-1]:
            # Last time interval
            return self.gamma[len(self.control_points)] # -1 will correspond to time T
        
        for i in range(1,len(self.control_points)):
            if self.control_points[i-1] <= t and self.control_points[i] > t:
                # (i+1)th time interval
                return self.gamma[i]
        
    def get_node_at_length(self, length):
        n = self.rho.shape[0] - 2*self.pad
        a = n / self.b
        pos = a * length - 1/2

        if pos < 0:
            prev = 0
            next = 0
        elif pos > n-1:
            prev = n-1
            next = n-1
        else:
            prev = torch.floor(pos)
            next = torch.ceil(pos)

        return prev, next, pos
    
    def get_speed_at_node(self, prev, next, pos):
        '''
        Important: prev, next, pos refers to internal nodes
        When using actual densities need to shift by the number of boundary nodes
        '''
        # prev_speed = self.gamma[self.idx] * (1. - self.rho[int(prev)])
        # next_speed = self.gamma[self.idx] * (1. - self.rho[int(next)])
        prev_speed = self.gamma[self.idx] * (1. - self.rho[int(prev)+self.pad])
        next_speed = self.gamma[self.idx] * (1. - self.rho[int(next)+self.pad])
        
        if prev == next:
            avg_speed = prev_speed
        else:
            # avg_speed = ((1. + prev - pos) * prev_speed + (1. + pos - next) * next_speed) / 2
            avg_speed = (pos - prev) * prev_speed + (next - pos) * next_speed

        return avg_speed
    
    def get_speed(self, length, printing = False):
        '''
        Return the speed calculated at a given length
        For now, just use local speed

        What happens if the length is exactly on a cell midpoint?
        '''
        # Find which two nodes are closest to the length
        # Then interpolate between the two nodes
        
        # length is between 0 and b
        # 0 should map to node 1, b should map to node N
        # (nodes 0 and N+1 are outside of road)
        if length == 0:
            length = torch.tensor(0.0)
        # n = self.n_internal
        n = self.rho.shape[0] - 2*self.pad
        a = n / self.b
        
        pos = a * length - 1/2

        # Position now relates to the internal cells of the road
        # pos < 0 -> first cell
        # pos = 0 -> first cell
        # pos = 1 -> second cell
        # ...
        # pos = n - 1 -> last internal cell
        # pos > n - 1 -> last internal cell

        if pos < 0:
            # Speed given by first cell
            prev = 0
            next = 0
        elif pos > n - 1:
            # Speed given by last cell
            prev = n-1
            next = n-1
        else:
            # Speed given by combination of two cells
            prev = torch.floor(pos)
            next = torch.ceil(pos)

        if printing:
            print(f"Position: {pos}")
            print(f"Nodes: {prev} and {next}")
            print(f"Closeness to nodes: {1 + prev - pos} and {1 + pos - next}")
            print(f"Speed: {self.gamma[self.idx]}")
        
        prev_speed = self.gamma[self.idx] * (1. - self.rho[int(prev)])
        next_speed = self.gamma[self.idx] * (1. - self.rho[int(next)])
        
        if printing:
            try:
                print(f"Version of left density: {self.rho[int(prev)]._version}, {self.rho[int(prev)]}")
            except:
                pass

            try:
                print(f"Version of right density: {self.rho[int(next)]._version}, {self.rho[int(next)]}")
            except:
                pass

            print(type(self.rho[int(prev)]))
        
        if printing:
            print(f"Speeds: {prev_speed} and {next_speed}")
            print(f"Densities: {self.rho[int(prev)]} and {self.rho[int(next)]}")
            print(f"Speed limit: {self.Vmax[self.idx]}, version {self.Vmax[self.idx]._version}")
        
        if prev == next:
            avg_speed = prev_speed
        else:
            # avg_speed = ((1. + prev - pos) * prev_speed + (1. + pos - next) * next_speed) / 2
            avg_speed = (pos - prev) * prev_speed + (next - pos) * next_speed
        
        if printing:
            print(f"Average speed: {avg_speed}")
        
        return avg_speed
    
    def get_speed_updated(self, length, dt):
        '''
        Return the speed calculated at a given length
        First calculate the local speed, and then use this speed to calculate the next position
        Use this next position to calculate the speed also here
        Take weighted average over the two speeds and use that as actual speed
        '''
        # Find which two nodes are closest to the length
        # Then interpolate between the two nodes
        
        # length is between 0 and b
        # 0 should map to node 1, b should map to node N
        # (nodes 0 and N+1 are outside of road)
        if length == 0:
            length = torch.tensor(0.0)
        
        prev, next, pos = self.get_node_at_length(length)
        speed = self.get_speed_at_node(prev, next, pos)
        
        # Update speed using the next position of the bus:
        updated_length = length + dt*speed

        if updated_length >= self.b:
            updated_speed = self.gamma[self.idx] * (1. - self.rho[-1])
        else:
            # Next position is still on road
            updated_prev, updated_next, updated_pos = self.get_node_at_length(updated_length)
            updated_speed = self.get_speed_at_node(updated_prev, updated_next, updated_pos)
        
        return (1.2 * speed + 0.8 * updated_speed) / 2 