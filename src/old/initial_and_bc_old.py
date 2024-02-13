import torch

def constant_density(x):
    # Constant 
    return torch.ones_like(x) * 0.5

def rarefaction_data(x):
    out = torch.ones_like(x)

    return out.where(x < 0, 0)

def shock_data(x):
    out = torch.ones_like(x)

    return out.where(x > 0, -1)

def circular_BC(U, order):
    # Apply circular BC to U
    # Let U[0] = U[-2] and U[-1] = U[1]
    # U[-2] last internal cell
    # U[1] first internal cell
    # U[0] left boundary
    # U[-1] right boundary

    match order:
        case 1:
            # In condition:
            U[0] = U[-2].clone()

            # Out condition
            U[-1] = U[1].clone()   
        case _:
            # In condition:
            U[0] = U[-4].clone()
            U[1] = U[-3].clone()

            # Out condition
            U[-1] = U[2].clone()
            U[-2] = U[3].clone()

    return U

def rarefaction_bc(U, order, t):

    U[:order] = 1.

    U[-order:] = 0.

    return U


def shock_bc(U, order, t):

    U[:order] = -1.
    U[-order:] = 1.
    return U
    

def variable_inflow(U, order, t):
    inflow = max(0, torch.sin(torch.tensor(2*torch.pi*t)))
    U[:order] = inflow

    U[-order:] = U[-order-1]
    return U



def constant

def init_density(max_dens, type):
    '''
    Returns discrete initial density given the maximum density on road
    and the type

    The different initial distributions supported are:

    type = 0  ->  Constant
    type = 1  ->  Linearly increasing
    type = 2  ->  Linearly decreasing
    ...
    '''

    match type:
        case 0:
            # Constant density
            return lambda x : torch.ones_like(x) * max_dens
        case 1:
            # Linearly increasing
            def lin_incr(x):
                out = torch.zeros_like(x)
                ...
                return out
            return lin_incr
        
        case 2: