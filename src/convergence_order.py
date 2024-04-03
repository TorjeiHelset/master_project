import torch

def perform_step(rho, dt):
    pass

def get_density(N, scheme, limiter, dt, initial = lambda x: torch.zeros_like(x)):
    '''
    Method for performing FV on a single lane to calculate convergence order
    Convergence to check is L1 convergence
    Domain is 1D in space with uniform grid size
    Time step will be fixed so as to guarantee that CFL condition is satisfied

    Vmax = 1,
    L = 1,
    b = 1,
    gamma = 1,

    N : Number of internal nodes
    scheme : Which scheme to use
    limiter :

    '''
    dx = 1 / (N + 1)
    match scheme:
        case 0 | 1 | 2:
            pad = 1
            scheme = scheme
        case 3 | 4:
            pad = 2
            scheme = scheme
        case _:
            pad = 2
            scheme = 3

    N_full = N + 2*pad
    j = torch.linspace(-(pad-1), (N+1) + (pad-1), N_full)
    x =  (j + 1/2) * dx 
    rho = initial(x)


    