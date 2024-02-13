import torch

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
            