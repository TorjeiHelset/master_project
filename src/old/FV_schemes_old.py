import torch

# Scaling parameters moved out of variables so that v_max no longer is input to flux functions
# When calculating approximate fluxes vmax is no longer relevant
# Scheme is now alpha * rho_t + (rho v)_x = 0, where alpha depends on flux

def flux(rho):
    '''
    Return the flux of cars evaluated at density - no longer dependent on V_max
    '''
    return rho * (1. - rho)


def d_flux(rho):
    '''
    Return the derivative of flux given density rho
    '''
    return 1. - 2.*rho


def LxF_flux(rho, dx, dt):
    '''
    Return the Lax-Friedrich approximation of the flux
    '''
    left = rho[:-1]
    right = rho[1:]
    return 0.5*(flux(left) + flux(right)) - 0.5*dx/dt * (right - left)

def Rusanov_Flux(rho):
    left = rho[:-1]
    right = rho[1:]
    # Can this be done quicker, i.e. without using for -- maybe, look at numlinalg numpy idx method....
    s = torch.tensor([max(abs(d_flux(rho[j])), abs(d_flux(rho[j+1]))) for j in range(len(rho)-1)]) 

    return 0.5*(flux(left) + flux(right)) - 0.5 * s * (right - left)


def Lax_Wendroff_Flux(rho, dx, dt):
    left = rho[:-1]
    right = rho[1:]
    a = d_flux(0.5*(left + right))

    return 0.5 * (flux(left) + flux(right)) - a*0.5*dt/dx * (flux(right) - flux(left))

def slope(rho, dx, limiter):
    left = rho[:-1]
    right = rho[1:]
    a = right - left

    match limiter:
        case "minmod":
            return minmod(a[1:], a[:-1]) #/ dx
        case "maxmod":
            return maxmod(a[1:], a[:-1]) / dx
        case "superbee":
            return superbee(a[1:], a[:-1]) / dx
        # case "mc":
        #     return 1/dx * mc(a[1:], a[:-1])
        


def minmod(a1, a2):
    '''
    Limiter that satisfies the TVD requirement
    '''

    return .5 * (torch.sign(a1) + torch.sign(a2)) * torch.minimum(torch.abs(a1), torch.abs(a2))

            
def maxmod(a1, a2):
    '''
    Limiter that satisfies the TVD requirement
    '''

    return .5 * (torch.sign(a1) + torch.sign(a2)) * torch.maximum(torch.abs(a1), torch.abs(a2))

def superbee(a1, a2):
    '''
    Limiter that satisfies the TVD requirement
    '''
    return maxmod(minmod(2.*a2, a1), minmod(a2, 2.*a1))
    

# def MC():
#     pass

def Rusanov_Flux_2(left, right):
    '''
    Slight modification to the Rusanov Flux function
    to fit the form needed for the 2. order scheme
    No longer take in full rho vector, but only left and right
    '''

    # Without for loop below??
    s = torch.tensor([max(abs(d_flux(left[j])), abs(d_flux(right[j]))) for j in range(len(left))])

    return 0.5*(flux(left) + flux(right)) - 0.5 * s * (right - left)

def L_operator(rho, dx, limiter, gamma):
    sigma = slope(rho, dx, limiter)
    left = torch.zeros(len(rho)-2)
    right = torch.zeros(len(rho)-2)

    left = rho[1:-1] - .5 * dx * sigma 
    right = rho[1:-1] + .5 * dx * sigma

    F = Rusanov_Flux_2(left[:-1], right[1:])
    #print(F[1:] - F[:-1])
    L_out = torch.zeros_like(rho)
    L_out[2:-2] = -gamma/dx * (F[1:] - F[:-1]) # Check dividing by dx here, in sigma or both - at this point alpha variable need to be introduced
    return L_out

def SSP_RK(rho, dx, limiter, dt, gamma):
    # Need also alpha parameter
    rho_ = rho + dt * L_operator(rho, dx, limiter, gamma)
    rho__ = rho_ + dt * L_operator(rho_, dx, limiter, gamma)
    rho_new = .5 * (rho + rho__)

    return rho_new
    
def Euler(rho, dx, limiter, dt, gamma):
    # Alpha parameter
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

        Integral += (f1 + f2) * dt/2 # Area of trapezoid

    return Integral



sigma = 0.5
fmax =  flux(sigma)

def D(rho):
    if rho <= sigma:
        return flux(rho)
    else:
        return fmax
    
def S(rho):
    if rho <= sigma:
        return fmax
    else:
        return flux(rho)