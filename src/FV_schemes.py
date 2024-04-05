import torch

# Scaling parameters moved out of variables so that v_max no longer is input to flux functions
# When calculating approximate fluxes vmax is no longer relevant
# Scheme is now alpha * rho_t + (rho v)_x = 0, where alpha depends on flux
@torch.jit.script
def flux(rho, gamma):
    '''
    Return the flux of cars evaluated at density - no longer dependent on V_max, but rather gamma
    '''
    return gamma * rho * (1. - rho)

@torch.jit.script
def d_flux(rho, gamma):
    '''
    Return the derivative of flux given density rho

    Should work both for scalar and vector rho 
    '''
    return gamma * (1. - 2.*rho)


def LxF_flux(rho, dx, dt, gamma):
    '''
    Return the Lax-Friedrich approximation of the flux
    Maybe not correct? Should take left and right molecules, and not the current molecule in...
    '''
    left = rho[:-1]
    right = rho[1:]
    return 0.5*(flux(left, gamma) + flux(right, gamma)) - 0.5*dx/dt * (right - left)

def Rusanov_Flux(rho, gamma):
    # This is not being used...
    left = rho[:-1]
    right = rho[1:]

    # Can this be done quicker, i.e. without using for -- maybe, look at numlinalg numpy idx method....
    s = torch.tensor([max(abs(d_flux(rho[j], gamma)), abs(d_flux(rho[j+1], gamma))) for j in range(len(rho)-1)]) 

    # Potential fix that might save some time...
    # Needs to be checked
    #s = torch.max(torch.abs(d_flux(rho[:-1], gamma)), torch.abs(d_flux(rho[1:], gamma)), dim=0).values

    return 0.5*(flux(left, gamma) + flux(right, gamma)) - 0.5 * s * (right - left)


def Lax_Wendroff_Flux(rho, dx, dt, gamma):
    left = rho[:-1]
    right = rho[1:]
    a = d_flux(0.5*(left + right))
    return 0.5 * (flux(left, gamma) + flux(right, gamma)) - a*0.5*dt/dx * (flux(right, gamma) - flux(left, gamma))

@torch.jit.script
def minmod(a1, a2):
    '''
    Limiter that satisfies the TVD requirement
    '''
    return torch.tensor(.5) * (torch.sign(a1) + torch.sign(a2)) * torch.minimum(torch.abs(a1), torch.abs(a2))

@torch.jit.script 
def maxmod(a1, a2):
    '''
    Limiter that satisfies the TVD requirement
    '''
    return torch.tensor(.5) * (torch.sign(a1) + torch.sign(a2)) * torch.maximum(torch.abs(a1), torch.abs(a2))

@torch.jit.script
def superbee(a1, a2):
    '''
    Limiter that satisfies the TVD requirement
    '''
    return maxmod(minmod(2.*a2, a1), minmod(a2, 2.*a1))
    
# def MC():
#     pass

@torch.jit.script
def slope(rho, limiter):
    left = rho[:-1]
    right = rho[1:]
    a = right - left

    # match limiter:
    #     case "minmod":
    #         ##############################################
    #         # Divide by dx or not?
    #         ##############################################
    #         return minmod(a[1:], a[:-1]) / dx
    #     case "maxmod":
    #         return maxmod(a[1:], a[:-1]) / dx
    #     case "superbee":
    #         return superbee(a[1:], a[:-1]) / dx
    if limiter == torch.tensor(1.0):
        return minmod(a[1:], a[:-1])
    elif limiter == torch.tensor(2.0):
        return maxmod(a[1:], a[:-1])
    else:# limiter == torch.tensor(3.0):
        return superbee(a[1:], a[:-1])
    
        # case "mc":
        #     return 1/dx * mc(a[1:], a[:-1])

@torch.jit.script
def Rusanov_Flux_2(left, right, gamma):
    '''
    Slight modification to the Rusanov Flux function
    to fit the form needed for the 2. order scheme
    No longer take in full rho vector, but only left and right
    '''
    # Any more optimization?

    # Without for loop below??
    # Gamma moved out of torch.tensor and torch.max so that derivative depends on it
    
    s = gamma * torch.max(torch.abs(d_flux(left, torch.tensor(1.))), 
                          torch.abs(d_flux(right, torch.tensor(1.))))
    
    return 0.5*(flux(left, gamma) + flux(right, gamma)) - 0.5 * s * (right - left)

@torch.jit.script
def L_operator(rho, dx, limiter, gamma):
    sigma = slope(rho, limiter)
    left = torch.zeros(len(rho)-2)
    right = torch.zeros(len(rho)-2)

    left = rho[1:-1] + torch.tensor(.5)  * sigma 
    right = rho[1:-1] - torch.tensor(.5) * sigma
    F = Rusanov_Flux_2(left[:-1], right[1:], gamma)
    L_out = torch.zeros_like(rho)
    L_out[2:-2] = -1/dx * (F[1:] - F[:-1])
    return L_out

@torch.jit.script
def SSP_RK(rho, dx, limiter, dt, gamma):
    # Need also alpha parameter
    rho_ = rho + dt * L_operator(rho, dx, limiter, gamma)
    rho__ = rho_ + dt * L_operator(rho_, dx, limiter, gamma)
    rho_new = .5 * (rho + rho__)
    return rho_new
    
def Euler(rho, dx, limiter, dt, gamma):
    rho_new = rho + dt * L_operator(rho, dx, limiter, gamma)
    return rho_new

def total_flux_out(rho_dict, order):
    '''
    Caluclates integral of linear interpolation of flux out of system
    Consider flux as close to border as possible

    Does not actually consider total flux out of system, but sum of flux out of each road
    '''

    Integral = 0
    times = list(rho_dict.keys())
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        f1 = flux(rho_dict[times[i-1]][-order-1])
        f2 = flux(rho_dict[times[i]][-order-1])

        Integral = Integral + (f1 + f2) * dt/2 # Area of trapezoid

    return Integral

sigma = torch.tensor(0.5)

def fmax(gamma):
    return flux(sigma, gamma)

def D(rho, gamma):
    if rho <= sigma:
        return flux(rho, gamma)
    else:
        return fmax(gamma)

def S(rho, gamma):
    if rho <= sigma:
        return fmax(gamma)
    else:
        return flux(rho, gamma)

def D_non_torchscript(rho, gamma):
    if rho <= sigma:
        return flux_non_torchscript(rho, gamma)
    else:
        return fmax_non_torchscript(gamma)
    
def S_non_torchscript(rho, gamma):
    if rho <= sigma:
        return fmax_non_torchscript(gamma)
    else:
        return flux_non_torchscript(rho, gamma)

def fmax_non_torchscript(gamma):
    return flux_non_torchscript(sigma, gamma)

def flux_non_torchscript(rho, gamma):
    return gamma * rho * (1. - rho)
