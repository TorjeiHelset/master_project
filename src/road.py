import torch
# import numpy as np
import FV_schemes as fv


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
    inflow = None
    queue_length = None
    control_points = None
    index = None # Index of control point being evaluated, needs to be evaluated for each road at each time
    max_dens = None
    left_pos = None
    right_pos = None
    periodic = False
    flux_in = None
    id = ""

    def __init__(self, b, L, N, Vmax, control_points, scheme = 3, limiter = "minmod", 
                 initial = lambda x: torch.zeros_like(x), inflow = -1, max_dens=1,
                 left_pos = (-1,0), right_pos = (0,1), periodic = False, flux_in = -1,
                 id = ""):
        
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
        self.dx = torch.tensor(1 / (N + 1)) # Should be the same for all roads
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
        self.N_internal = N*b + (b-1)
        N_full = self.N_internal +  2*self.pad
        # Maybe not 100% correct, but very close...
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
        match limiter:
            case "minmod":
                self.limiter = torch.tensor(1.0)
            case "maxmod":
                self.limiter = torch.tensor(2.0)
            case "superbee":
                self.limiter = torch.tensor(3.0)
        # Inflow determines boundary condition in to road
        self.inflow = torch.tensor(inflow)

        # For now it is assumed that all flux is allowed to exit road
        # Should maybe extend to limit the allowed flux that can exit road

        self.queue_length = torch.tensor(0.0)
        self.index = 0
        self.max_dens = max_dens

        self.left_pos = left_pos 
        self.right_pos = right_pos

        self.periodic = periodic
        self.flux_in = torch.tensor(flux_in)
        self.id = id
        

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
        self.gamma = [v / self.L for v in self.Vmax]

    def demand(self):
        # clone needed?
        return self.max_dens * fv.D(self.rho[-1].clone(), self.gamma[self.idx])

    def supply(self):
        # Clone needed?
        return self.max_dens * fv.S(self.rho[-1].clone(), self.gamma[self.idx])
    
    def update_right_boundary(self, incoming_flux, dt):
        # left is internal cell, middle is first boundary cell
        left, middle = self.rho[-self.pad-1], self.rho[-self.pad]
        # Calculate rusanov flux:
        left_flux = fv.Rusanov_Flux_2(left, middle, self.gamma[self.idx])

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
        right_flux = fv.Rusanov_Flux_2(middle, right, self.gamma[self.idx])

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
        
        return CFL * self.dx / (self.max_dens * max_flux)

    def solve_internally(self, dt):
        '''
        Solving conservation law for internal
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
                self.rho = fv.SSP_RK(self.rho, self.dx, self.limiter, dt, self.gamma[self.idx])
            case 4:
                self.rho = fv.Euler(self.rho, self.dx, self.limiter, dt, self.gamma[self.idx])

    def apply_bc(self, t, dt):
        # Change structure of road to instead have a boundary condition function and use
        # this function to update the road.

        # Apply boundary conditions to edges not attached to any junction
        # If condition becomes on flux in then max dens might be necessary also here


        if self.periodic:
            # Set periodic boundary conditions
            match self.pad:
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
            # For now set right boundary elements equal to closest interior
            # This means all flux is allowed to leave the road
            if not self.right:
                # Right boundary not attached to junction
                self.rho[-self.pad:] = self.rho[-self.pad-1]
                # Should maybe instead have some limit on the flux out
            
            if not self.left:
                # Set some influx to left boundary
                if self.inflow < 0:
                    # No inflow density specified, add artifical inflow
                    inflow = torch.max(torch.tensor(0), torch.sin(torch.tensor(2*torch.pi*t)))
                else:
                    # if self.id[0] == '1' or self.id[0] == '3':
                    #     if 50 < t < 100:
                    #         inflow = self.inflow * 4
                    #     elif t > 100:
                    #         inflow = self.inflow * 0.5
                    #     else:
                    #         inflow = self.inflow
                    # elif self.id[0] == '5' or self.id[0] == '7':
                    #     if t > 100:
                    #         inflow = self.inflow * 0.25
                    #     else:
                    #         inflow = self.inflow
                    # else:
                    #     inflow = self.inflow
                    if 200 < t < 400:
                        inflow = self.inflow * 0.3
                    elif 600 < t < 800:
                        inflow = self.inflow * 2
                    elif 800 <= t:
                        inflow = self.inflow * 0.5
                    else:
                        inflow = self.inflow

                if self.flux_in >= 0:
                    # Inflow is given as flux
                    # For now this is only allowed to be constant
                    # if 300 < t < 600:
                    #     f_in = self.flux_in * 2.0#0.5
                    # elif t > 800:
                    #     f_in = self.flux_in * 0.8
                    # else:
                    #     f_in = self.flux_in
                    # Manual update of the flux
                    # if self.id == "1_fw" or self.id[0] == "3_bw":
                    #     if 50 < t < 100:
                    #         f_in = self.flux_in * 4
                    #     elif t > 100:
                    #         f_in = self.flux_in * 0.5
                    #     else:
                    #         f_in = self.flux_in
                    # elif self.id[0] == "5" or self.id[0] == "7":
                    #     if 50 < t < 100:
                    #         f_in = self.flux_in * 0.25
                    #     else:
                    #         f_in = self.flux_in
                    # else:
                    #     f_in = self.flux_in
                    if 200 < t < 400:
                        f_in = self.flux_in * 0.3
                    elif 600 < t < 800:
                        f_in = self.flux_in * 2
                    elif 800 <= t:
                        f_in = self.flux_in * 0.5
                    else:
                        f_in = self.self.flux_in
                    # f_in = self.flux_in
                else:
                    # Inflow is given as density
                    if 300 < t < 600:
                        f_in = fv.D(inflow, self.gamma[self.idx]) * 0.5
                    else:
                        f_in = fv.D(inflow, self.gamma[self.idx]) # Caclulate inflow

                if self.queue_length > 0:
                    # Set the influx to the maximum possible
                    D = fv.flux(torch.tensor(0.5), self.gamma[self.idx].clone())
                else:
                    # Set the influx to the mimum of actual and maximum
                    D = torch.min(f_in, fv.flux(torch.tensor(0.5), self.gamma[self.idx].clone()))
                
                # Set influx to the mimum of actual influx and maximum capacity
                gamma_in = torch.min(D, fv.S(self.rho[self.pad-1].clone(), self.gamma[self.idx].clone())) # Actual flux in

                # Update queue length using the difference between actual and desired flux in
                self.queue_length = self.queue_length + dt * (f_in - gamma_in)

                # Update density according to flux in
                right, out_mid = self.rho[self.pad], self.rho[self.pad-1]
                s = torch.max(torch.abs(fv.d_flux(out_mid, self.gamma[self.idx])), torch.abs(fv.d_flux(right, self.gamma[self.idx])))
                mid_f = fv.flux(out_mid.clone(), self.gamma[self.idx])
                right_f = fv.flux(right.clone(), self.gamma[self.idx])
                right_flux = 0.5 * (mid_f + right_f) - 0.5 * s * (right - out_mid)
                
                self.rho[self.pad-1] = self.rho[self.pad-1] - dt / self.dx * (right_flux - gamma_in)
                if self.pad > 1:
                    self.rho[0] = self.rho[1]
                
                # Instead of directly setting the density at the boundary, find out how much
                # flux is sent into the road. If more flux than capacity is sent in, then 
                # the actual flux in should be the capacity, and the queue into the road should increase

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

        n = self.rho.shape[0] - 2
        a = (n-1) / self.b
        
        pos = a * length + 1

        prev = torch.floor(pos)
        next = prev + 1

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
        
        avg_speed = (1. + prev - pos) * prev_speed + (1. + pos - next) * next_speed
        
        if printing:
            print(f"Average speed: {avg_speed}")

        return avg_speed