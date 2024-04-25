import torch
import FV_schemes as fv

def init_density(max_dens, type):
    '''
    Returns discrete initial density given the maximum density on road
    and the type

    The different initial distributions supported are:

    type = 0  ->  0
    type = 1  ->  Constant
    type = 2  ->  Linearly increasing
    type = 3  ->  Linearly decreasing
    '''

    match type:
        case 0:
            return lambda x : torch.zeros_like(x)
        case 1:
            # Constant density
            return lambda x : torch.ones_like(x) * max_dens
        case 2:
            # Linearly increasing
            # x is already increasing, so only need some scaling of x
            def lin_incr(x):
                return torch.linspace(0, max_dens, len(x))
            return lin_incr
        case 3:
            # Linearly decreasing
            def lin_decr(x):
                return torch.linspace(max_dens, 0, len(x))
            return lin_decr

def boundary_conditions(type, max_dens = 1, densities = [0], fluxes = [0], time_jumps = [], 
                        in_speed = 50, L = 50, amplitude = 0.3, period = 100, 
                        flux_amplitude = 0.01):
    '''
    Returns a boundary condition of a certain type given the maximum density
    The boundary condition retunred should be a potentially non-continuous function
    dending only on time.
    The boundary condition function are constructed by creating a dummy road of length
    L and speed limit in_speed. Then the density at the outgoing edge of the road
    may vary non-continuously in time. The density can then be used to calculate the
    flux.

    type = 0 -> No influx
    type = 1 -> Density on incoming road given by piecewise function
    type = 2 -> Density follows a sinus wave with amplitude and period centered at densities[0]
    type = 3 -> Fluxes directly given by fluxes changing at times given in
                flux_time_jumps
    type = 4 -> Fluxes given by sinus wave centered at fluxes[0]
    '''
    for rho in densities:
        assert 0 <= rho <= 1
    assert len(densities) == len(time_jumps)+1
    
    match type:
        case 0:
            return lambda t : torch.tensor(0.0)
        case 1:
            # Densities given as piecewise function
            if len(time_jumps) == 0:
                gamma = in_speed / L
                return lambda t : max_dens * fv.flux(densities[0], gamma)
            
            def piecewise(t):
                gamma = in_speed / L
                if t < time_jumps[0]:
                    return max_dens * fv.flux(densities[0], gamma)
                for i, time in enumerate(time_jumps):
                    if t >= time:
                        return max_dens * fv.flux(densities[i+1], gamma)
            return piecewise
        
        case 2:
            # Density given by sinus wave centered at densities[0]
            raise NotImplementedError("Sinus boundary condition for density on incoming road not implemented")
        case 3:
            # Fluxes given as piecewise function
            if len(time_jumps) == 0:
                return lambda t : max_dens * fluxes[0]
            
            def piecewise(t):
                if t < time_jumps[0]:
                    return max_dens * fluxes[0]
                for i, time in enumerate(time_jumps):
                    if t >= time:
                        return max_dens * fluxes[i+1]
            return piecewise
        
        case 4:
            # Fluxes given by sinus wave centered around fluxes[0]
            raise NotImplementedError("Sinus boundary condition for flux on incoming road not implemented")