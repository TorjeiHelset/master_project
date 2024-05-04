import torch
import FV_schemes as fv

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
    

class RoundaboutRoad:
    queue_length = torch.tensor(0.0)
    inflow_fnc = None
    max_inflow = None

    def __init__(self, inflow_fnc, max_inflow):
        self.inflow_fnc = inflow_fnc
        self.max_inflow = max_inflow
    
    def demand(self, t, dt):
        # The demand should be updated so that the queue does not become negative!
        flux = torch.tensor(0.0)
        f_in = self.inflow_fnc(t)
        if self.queue_length > 0:
            flux = torch.min(self.max_inflow,
                             f_in + self.queue_length / dt)
        else:
            flux = torch.min(self.inflow_fnc(t), self.max_inflow)
        return flux

    def update_queue(self, actual_flux, dt, t):
        # Can this queue length be negative?
        # print(actual_flux)
        # print(dt)
        # print(t)
        # print(self.inflow_fnc(t))
        # print()
        # The queue length could become negative!!!
        # For now add a maximum function call to avoid the queue being negative
        self.queue_length = torch.maximum(self.queue_length + dt * (self.inflow_fnc(t) - actual_flux),
                                      torch.tensor(0.0))

class RoundaboutJunction:
    '''
    The roundabouts of the junction are similar to the regular ones, with
    some differences. 
    This junctions must be a 2x2 junction
    There will always be mainline incoming and outgoing roads
    There will always be secondary incoming and outgoing roads
    The secondary roads may be actual roads of the system, that we are
    interested in. In this case the boundary nodes of these roads will be 
    updated.
    The secondary roads may also not be a part of the system, in which case
    all of the traffic is allowed enter the outgoing road, and the incoming
    traffic is represented by a queue. 
    Both secondary roads will either be a part of the simulation, or none of
    them will be.
    There are no traffic lights in the roundabout, and no crossing connections
    Traffic from incoming mainline can enter both outgoing mainline, and 
    the secondary outgoing road
    Traffic from secondary incoming, can only enter the mainline out
    The priority list matrix is known
    The zeros of the distribution is known.
    There are no crossing connections.
    The distribution of traffic is determined using a FIFO rule.
    
    If the secondary roads are not interesting, then a maximum flux value is also needed
    '''
    mainline_in = None
    mainline_out = None
    secondary_in = None
    secondary_out = None
    alpha = 0.5
    queue_junction = True

    def __init__(self, main_in, main_out, alpha, second_in, second_out = None,
                 queue_junction = True):
        self.mainline_in = main_in
        self.mainline_out = main_out
        self.alpha = alpha # Single parameter > 0
        self.secondary_in = second_in
        self.secondary_out = second_out
        # Maybe try to instead infer the queue_junction type based on the type of the secondary roads
        self.queue_junction = queue_junction

        # Setting boundary edges of mainline and secondary roads equal to true
        self.mainline_in.right = True
        self.mainline_out.left = True
        if not queue_junction:
            self.secondary_in.right = True
            self.secondary_out.left = True

    def divide_flux(self, dt, t):
        '''
        The junction of a roundabout are special, and the explicit solution is known
        Influx from secondary road eiter determined by density on road, or 
        by inflow function

        if secondary roads a part of simulation, the densities need to be
        updated.
        
        Check whether it should be alpha or (1-alpha)...
        '''
        # Calculate the single priority parameter based on density on mainline
        beta = priority_fnc(self.mainline_in.rho[-1])
        min_dt = dt + 1
        if not self.queue_junction:
            demands = [None, None]
            demands[0] = self.mainline_in.demand()
            demands[1] = self.secondary_in.demand()
            # Calculating the supplies
            supplies = [None, None]
            supplies[0] = self.mainline_out.supply()
            supplies[1] = self.secondary_out.supply()
        
            # Adjusting the first demand based on the supplies:
            demands[0] = torch.minimum(demands[0], supplies[1] / self.alpha)

            # Calculating the fluxes using a FIFO rule:
            in_fluxes = [None, None]
            out_fluxes = [None, None]
            max_main_in = torch.max(beta * supplies[0], supplies[0] - demands[1])
            max_second_in = torch.max((1-beta) * supplies[0], supplies[0] - (1-self.alpha)*demands[0])
            in_fluxes[0] = 1/(1-self.alpha) * torch.min((1-self.alpha)*demands[0], max_main_in)
            in_fluxes[1] = torch.min(demands[1], max_second_in)
            out_fluxes[0] = torch.min((1-self.alpha)*demands[0] + demands[1], supplies[0])
            out_fluxes[1] = self.alpha*in_fluxes[0]

            # Update the densities of roads:
            # self.mainline_in.update_right_boundary(in_fluxes[0] / self.mainline_in.max_dens, dt, t)
            # self.mainline_out.update_left_boundary(out_fluxes[0] / self.mainline_out.max_dens, dt, t)
            # self.secondary_in.update_right_boundary(in_fluxes[1] / self.secondary_in.max_dens, dt, t)
            # self.secondary_out.update_left_boundary(out_fluxes[1] / self.secondary_out.max_dens, dt, t)
            min_dt_ = self.mainline_in.update_right_flux(in_fluxes[0] / self.mainline_in.max_dens)
            min_dt = torch.min(min_dt, min_dt_)
            min_dt_ = self.mainline_out.update_left_flux(out_fluxes[0] / self.mainline_out.max_dens)
            min_dt = torch.min(min_dt, min_dt_)
            min_dt_ = self.secondary_in.update_right_flux(in_fluxes[1] / self.secondary_in.max_dens)
            min_dt = torch.min(min_dt, min_dt_)
            min_dt_ = self.secondary_out.update_left_flux(out_fluxes[1] / self.secondary_out.max_dens)
            min_dt = torch.min(min_dt, min_dt_)

        else:
            # Secondary roads not a part of the simulation
            main_rho_in = self.mainline_in.rho[-self.mainline_in.pad]
            main_rho_out = self.mainline_out.rho[self.mainline_out.pad]
            main_max_dens = self.mainline_in.max_dens
            main_gamma = self.mainline_in.gamma[self.mainline_in.idx]

            demands = [None, None]
            demands[0] = main_max_dens * fv.D(main_rho_in.clone(), main_gamma)
            demands[1] = self.secondary_in.demand(t, dt)

            # Calculate the supply:
            supply = main_max_dens * fv.S(main_rho_out.clone(), main_gamma)

            # Calculate the fluxes

            out_flux = torch.min((1-self.alpha)*demands[0] + demands[1], supply)
            in_fluxes = torch.zeros(2)
            max_main_in = torch.max(beta*supply, supply - demands[1])
            in_fluxes[0] = 1/(1-self.alpha) * torch.min((1-self.alpha)*demands[0],
                                                        max_main_in)
            max_second_in = torch.max((1-beta)*supply, supply - (1-self.alpha)*demands[0])
            in_fluxes[1] = torch.min(demands[1], max_second_in)

            # Update densities of roads:
            # self.mainline_in.update_right_boundary(in_fluxes[0] / self.mainline_in.max_dens, dt, t)
            # self.mainline_out.update_left_boundary(out_flux / self.mainline_out.max_dens, dt, t)
            min_dt_ = self.mainline_in.update_right_flux(in_fluxes[0] / self.mainline_in.max_dens)
            min_dt = torch.min(min_dt, min_dt_)
            min_dt_ = self.mainline_out.update_left_flux(out_flux / self.mainline_out.max_dens)
            min_dt = torch.min(min_dt, min_dt_)
            # Update queue length:
            self.secondary_in.update_queue(in_fluxes[1], dt, t)

        return min_dt

class Roundabout:
    junctions = []
    mainline_roads = []
    secondary_inroads = []
    secondary_outroads = []

    def __init__(self, mainline_roads, secondary_inroads, secondary_outroads,
                 roundabout_junctions):
        # Check that the mainline roads go in a loop
        # first_mainline = roundabout_junctions[0].mainline_in
        # last_mainline = roundabout_junctions[-1].mainline_out
        # assert first_mainline == last_mainline

        # # Check that all roads are a part of at least one roundabout junction
        # mainline_contained = False
        # for mainline in mainline_roads:
        #     for j in roundabout_junctions:
        #         if j.mainline_in == mainline:
        #             mainline_contained = True
        #             break
        # assert mainline_contained
        # second_in_cont = False
        # for second_in in secondary_inroads:
        #     for j in roundabout_junctions:
        #         if j.mainline_in == mainline:
        #             mainline_contained = True
        #             break
        # second_out_cont = False
        # for second_out in secondary_outroads:
        #     # Either None or a part of a junction
        #     if 

        self.junctions = roundabout_junctions
        self.mainline_roads = mainline_roads
        self.secondary_inroads = secondary_inroads
        self.secondary_outroads = secondary_outroads

    
    def apply_bc(self, dt, t):
        
        # Update flux on the junctions
        min_dt = dt + 1
        for j in self.junctions:
            # No check for right of way, because the right of way has been manually added
            # to the roundabout junctions as it is known beforehand
            min_dt_ = j.divide_flux(dt, t)
            min_dt = torch.min(min_dt, min_dt_)

        return min_dt