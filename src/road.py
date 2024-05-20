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
        # Depending on the order of scheme, the number of boundary cells might differ
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


        self.b = b
        self.L = L 
        self.dx = torch.tensor(1 / (N + 2*self.pad)) # Should be the same for all roads
        # Denominator = 1 since x goes from 0 to b (0 to 1 on a unit length)

        if len(Vmax) != len(control_points) +1:
            print(len(Vmax))
            print(len(control_points))
        assert(len(Vmax) == len(control_points)+1)
        
        # The speed must be given in m/s and whether or not it should track gradient must be specified in advance
        self.Vmax = Vmax #[v / 3.6 for v in Vmax]
        self.control_points = control_points
    

        # When order of scheme is determined, the grid can be defined
        # For each unit length we have N internal nodes
        # We also have 2*self.pad boundary nodes on each unit length
        # Finally the total length is divided into b different segments, each of length L

        # In total we have
        self.N_full = b * (N + 2*self.pad)


        # self.N_internal = N*b
        # N_full = self.N_internal +  2*self.pad


        j = torch.arange(0, self.N_full, 1)
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
        self.idx = 0
        self.max_dens = max_dens

        self.left_pos = left_pos 
        self.right_pos = right_pos

        self.periodic = periodic
        self.id = id

        self.boundary_fnc = boundary_fnc

        self.left_boundary = self.rho[:self.pad]
        self.right_boundary = self.rho[-self.pad:]
        self.left_flux = torch.tensor(0.0)
        self.right_flux = torch.tensor(0.0) 

        
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

    def activation_fnc(self, length):
        return 1. - torch.sigmoid(10/self.dx * (length - (self.b - 3/2 *self.dx)) - 5)

    def demand(self):
        # clone needed?
        return self.max_dens * fv.D(self.rho[-1].clone(), self.gamma[self.idx])

    def supply(self):
        # Clone needed?
        return self.max_dens * fv.S(self.rho[0].clone(), self.gamma[self.idx])

    def update_right_flux(self, incoming_flux):
        '''
        Update the flux leading to the right edge of the road
        Also update the time step if needed to ensure stability
        '''
        self.right_flux = incoming_flux
        if 1 - 4 *incoming_flux /  self.gamma[self.idx] <= 0:
            # This should actually never become negative, but keep as 
            # safety for now
            return self.max_dt()
        else:
            min_dt = self.dx * 1/ ( self.gamma[self.idx] * torch.sqrt(1 - 4 *incoming_flux /  self.gamma[self.idx]))
        return torch.max(min_dt, self.dx / (self.max_dens * self.gamma[self.idx]))
    
    def update_left_flux(self, outgoing_flux):
        '''
        Update the flux leading to the left edge of the road
        Also update the time step if needed to ensure stability
        '''
        self.left_flux = outgoing_flux
        if 1 - 4 *outgoing_flux /  self.gamma[self.idx] <= 0:
            return self.max_dt()
        else:
            min_dt = self.dx * 1/ ( self.gamma[self.idx] * torch.sqrt(1 - 4 * outgoing_flux /  self.gamma[self.idx]))
        return torch.max(min_dt, self.dx / (self.max_dens * self.gamma[self.idx]))
        
    def update_right_boundary(self, incoming_flux, dt, slowdown_factors):
        '''
        The boundary cells are updated using a first order Rusanov scheme
        Calculate the rusanov flux between the last internal cell and the first boundary
        cell. If there are more boundary cells, calculate the flux between the boundary
        cells
        For the scheme to be conservative, the flux from the interface near the internal nodes needs to 
        be calculated using the scheme that is used for the internal nodes
        '''
        # print(f"Right boundary is being updated!")
        # print(f"Flux leaving road: {incoming_flux}")
        # print(f"time step: {dt}")
        # print()
        # Calculate the left flux at the rightmost node
        left = self.rho[-2]
        right = self.rho[-1]

        F = torch.zeros(self.pad + 1)

        F[-2] = fv.Rusanov_Flux_2(left.clone(), right.clone(), self.gamma[self.idx]) * slowdown_factors[-1]
        F[-1] = incoming_flux.clone()
        
        if self.pad > 1:
            # Calculate this flux using the high resolution scheme
            F[0] = fv.get_right_boundary_flux(self.rho.clone(), self.dx, self.limiter, dt,self.gamma[self.idx]) * slowdown_factors[-2]

        self.right_boundary = self.right_boundary - dt / self.dx * (F[1:] - F[:-1])
        
    def update_left_boundary(self, outgoing_flux, dt, slowdown_factors):
        # Update boundary cells using a first order rusanov scheme
        # The leftmost flux is coming from either a regular junction or a roundabout junction

        left = self.rho[0]
        right = self.rho[1]

        F = torch.zeros(self.pad+1)

        F[1] = fv.Rusanov_Flux_2(left.clone(), right.clone(), self.gamma[self.idx]) * slowdown_factors[0]
        F[0] = outgoing_flux.clone()

        if self.pad > 1:
            F[2] = fv.get_left_boundary_flux(self.rho.clone(), self.dx, self.limiter, dt, self.gamma[self.idx]) * slowdown_factors[1]

        self.left_boundary = self.left_boundary - dt / self.dx * (F[1:] - F[:-1])

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
        
        max_flux = torch.abs(fv.d_flux(self.rho, self.gamma[self.idx]))
        max_flux = torch.max(max_flux)
        
        if self.gamma[self.idx] < 0:
            print(f"{self.id} has a negative gamma!")
            print(self.gamma[self.idx])

        if max_flux > self.gamma[self.idx]:
            # print(f"{self.id} has a wrong maximum flux")
            # print(self.rho)
            max_flux = self.gamma[self.idx]

        # Avoid 0 division:
        if torch.abs(max_flux) < 1e-5:
            max_flux = torch.tensor(0.000001)

        # return CFL * self.dx / (self.max_dens * self.gamma[self.idx])
        return CFL * self.dx / (self.max_dens * max_flux)

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
                # print(f"Densities: {self.rho}")
                new_rho = torch.zeros_like(self.rho)

                F = fv.Rusanov_Flux(self.rho.clone(), self.gamma[self.idx])
                new_rho[1:-1] = -dt/self.dx * (F[self.pad:] - F[:-self.pad])
                self.rho = self.rho + new_rho.clone()
                # self.rho[self.pad:-self.pad] -= dt/self.dx * (F[self.pad:] - F[:-self.pad])
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
            # Should probably update the boundary cells, and not directly the interior cells
            match self.pad:
                # INPLACE 
                # CLONING
                case 1:
                    # In condition:
                    self.left_boundary[0] = self.rho[-2].clone()
                    # self.rho[0] = self.rho[-2].clone()

                    # Out condition
                    self.right_boundary[-1] = self.rho[1].clone()   
                    # self.rho[-1] = self.rho[1].clone()   
                case _:
                    # In condition:
                    self.left_boundary[0] = self.rho[-4].clone()
                    self.left_boundary[1] = self.rho[-3].clone()

                    # self.rho[0] = self.rho[-4].clone()
                    # self.rho[1] = self.rho[-3].clone()

                    # Out condition
                    self.right_boundary[-1] = self.rho[3].clone()
                    self.right_boundary[-2] = self.rho[2].clone()

                    # self.rho[-1] = self.rho[3].clone()
                    # self.rho[-2] = self.rho[2].clone()

        else:
            # For right boundary, set boundary elements equal to closest interior
            # This means all flux is allowed to exit
            if not self.right:
                # INPLACE
                # Right boundary not attached to junction - set denity equal to closest interior point
                # print(self.id)
                # print(self.rho[-self.pad-1])
                # self.right_boundary[:] = self.rho[-self.pad-1]

                # self.right_boundary[:] = self.rho[-self.pad-1].clone()
                self.right_boundary = self.rho[-self.pad-1].clone()
            

            # For left boundary some inflow conditions are necessary       
            if not self.left:
                # Set some influx to left boundary
                if self.boundary_fnc is None:
                    raise ValueError(f"No boundary function specified for road {self.id}!")

                else:
                    f_in =  self.boundary_fnc(t)
                    
                    # Ensure that the queue never becomes negative
                    D = torch.min(fv.flux(torch.tensor(0.5), self.gamma[self.idx].clone()),
                                      f_in + self.queue_length / dt)
                
                    # Set influx to the mimum of actual influx and maximum capacity
                    gamma_in = torch.min(D, fv.S(self.rho[self.pad-1].clone(), self.gamma[self.idx].clone())) # Actual flux in

                    # Update queue length using the difference between actual and desired flux in
                    self.queue_length = self.queue_length + dt * (f_in - gamma_in)
                    
                    # Update density according to flux in
                    # self.update_left_boundary(gamma_in, dt)
                    new_dt = self.update_left_flux(gamma_in)
                    return new_dt
        return dt

    def update_boundary_cells(self, dt, slowdown_factors):
        # Update the boundary cells using the respective left and right fluxes
        if not self.periodic:
            self.update_left_boundary(self.left_flux, dt, slowdown_factors[:self.pad])

            if self.right:
                self.update_right_boundary(self.right_flux, dt, slowdown_factors[-self.pad:])
            
        self.left_flux = torch.tensor(0.0)
        self.right_flux = torch.tensor(0.0)

    def update_boundaries(self):
        # Set the boundary cells of rho equal to the artificial left boundary and right boundary
        if not self.periodic:
            self.rho[:self.pad] = self.left_boundary.clone()
            self.rho[-self.pad:] = self.right_boundary.clone() 

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
        
    # The next three functions need to be updated, taking into account that boundary cells are now
    # placed on the road

    def get_node_at_length(self, length):
        n = self.rho.shape[0] # Equal to self.N_full
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
        prev_speed = self.gamma[self.idx] * (1. - self.rho[int(prev)])
        next_speed = self.gamma[self.idx] * (1. - self.rho[int(next)])
        
        if prev == next:
            avg_speed = prev_speed
        else:
            # avg_speed = ((1. + prev - pos) * prev_speed + (1. + pos - next) * next_speed) / 2
            avg_speed = (pos - prev) * prev_speed + (next - pos) * next_speed

        return avg_speed
    
    def get_speed(self, length, dt):
        '''
        Return the speed calculated at a given length
        First calculate the local speed, and then use this speed to calculate the next position
        Use this next position to calculate the speed also here
        Take weighted average over the two speeds and use that as actual speed
        '''
        # Find which two nodes are closest to the length
        # Then interpolate between the two nodes
        
        if length == 0:
            length = torch.tensor(0.0)
        
        internal_activation = self.activation_fnc(length)

        prev, next, pos = self.get_node_at_length(length)
        speed = self.get_speed_at_node(prev, next, pos)
        
        # Update speed using the next position of the bus:
        updated_length = length + dt*speed

        if updated_length >= self.b:
            # The bus will potentially move outside the junction
            updated_speed = self.gamma[self.idx] * (1. - self.rho[-1])
        else:
            # Next position is still on road
            updated_prev, updated_next, updated_pos = self.get_node_at_length(updated_length)
            updated_speed = self.get_speed_at_node(updated_prev, updated_next, updated_pos)

        # Can also return the slowdown coming from the distance to the junction

        return (1.2 * speed + 0.8 * updated_speed) / 2, internal_activation