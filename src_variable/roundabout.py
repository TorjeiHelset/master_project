import torch
import FV_schemes as fv


class RoundaboutRoad:
    queue_length = None
    inflow_fnc = None
    max_inflow = None

    def __init__(self, inflow_fnc, max_inflow):
        self.inflow_fnc = inflow_fnc
        self.max_inflow = max_inflow
    
    def demand(self, t):
        return torch.max(self.inflow_fnc(t), self.max_inflow)

    def update_queue(self, actual_flux, dt, t):
        # Can this queue length be negative?
        self.queue_length = self.queue_length + dt * (self.inflow_fnc(t) - actual_flux)

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
                 queue_junction = True, max_flux = 1):
        self.mainline_in = main_in
        self.mainline_out = main_out
        self.alpha = alpha # Single parameter > 0
        self.secondary_in = second_in
        self.secondary_out = second_out
        self.queue_junction = queue_junction
        self.max_flux = max_flux

    def divide_flux(self, t, dt):
        # Explicit solution of this 2x2 junction is known
        # Influx from secondary road eiter determined by density on road, or 
        # by inflow function

        # if secondary roads a part of simulation, the densities need to be
        # updated.
        
        # Check whether it should be alpha or (1-alpha)...

        beta = ... # Calculate the single priority parameter based on density on mainline

        if not self.queue_junction:
            # Calculating the demands:
            demands = torch.zeros(2)
            demands[0] = self.mainline_in.demand()
            demands[1] = self.secondary_in.demand()

            # Calculating the supplies
            supplies = torch.zeros(2)
            supplies[0] = self.mainline_out.supply()
            supplies[1] = self.secondary_out.supply()
        
            # Adjusting the first demand based on the supplies:
            demands[0] = torch.min(demands[0], supplies[1] / self.distribution)

            # Calculating the fluxes using a FIFO rule:
            in_fluxes = torch.zeros(2)
            out_fluxes = torch.zeros(2)
            max_main_in = torch.max(beta * supplies[0], supplies[0] - demands[1])
            max_second_in = torch.max((1-beta) * supplies[0], supplies[0] - (1-self.alpha)*demands[0])
            in_fluxes[0] = 1/(1-self.alpha) * torch.min((1-self.alpha)*demands[0], max_main_in)
            in_fluxes[1] = torch.min(demands[1], max_second_in)
            out_fluxes[0] = torch.min((1-self.alpha)*demands[0] + demands[1], supplies[0])
            out_fluxes[1] = self.alpha*in_fluxes[0]

            # Update the densities of roads:
            self.mainline_in.update_right_boundary(in_fluxes[0], dt)
            self.mainline_out.update_left_boundary(out_fluxes[0], dt)
            self.secondary_in.update_right_boundary(in_fluxes[1], dt)
            self.secondary_out.update_left_boundary(out_fluxes[1], dt)


        else:
            # Secondary roads not a part of the simulation
            main_rho_in = self.mainline_in.rho[-self.mainline_in.pad]
            main_rho_out = self.mainline_out.rho[-self.mainline_out.pad]
            main_max_dens = self.mainline_in.max_dens
            main_gamma = self.mainline_in.gamma[self.mainline_in.idx]

            # Calculate the demands:
            demands = torch.zeros(2)
            demands[0] = main_max_dens * fv.D(main_rho_in, main_gamma)
            demands[1] = self.secondary_in.demand(t)

            # Calculate the supply:
            supply = main_max_dens * fv.S(main_rho_out, main_gamma)

            # Calculate the fluxes
            out_flux = torch.min((1-self.alpha)*demands[0] + demands[1])
            in_fluxes = torch.zeros(2)
            in_fluxes[0] = 1/(1-self.alpha) * torch.min((1-self.alpha)*demands[0],
                                                        supply - demands[1])
            max_second_in = torch.max((1-beta)*supply, supply  (1-self.alpha)*demands[0])
            in_fluxes[1] = torch.min(demands[1], max_second_in)

            # Update densities of roads:
            ...

            # Update queue length:
            self.secondary_in.update_queue(in_fluxes[1], dt, t)
