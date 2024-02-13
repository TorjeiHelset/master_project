import torch
import gurobipy as gp


def flux(rho, gamma):
    '''
    Return the flux of cars evaluated at density u and 
    given maximum velocity Vmax
    '''
    return gamma * rho * (1. - rho)

sigma = 0.5
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

def find_parameters(rho_in, rho_out, alpha, gamma_in, gamma_out, active,
                    max_dens_in, max_dens_out):
    '''
    Finding beta that maximizes flux could give more realistic results, with the potential cost
    of increasing running time

    -> optimmization procedure have to be solved at each step - costly
    '''

    n = len(rho_in) # number of cars coming in to junction
    m = len(rho_out) # number of cars leaving system

    with gp.Env(empty=True) as env:
        env.setParam('outputFlag',0)
        env.start()

        with gp.Model(env=env) as model:
            # Alpha parameters already specified
            beta = model.addVars(n,m, vtype=gp.GRB.CONTINUOUS, name="beta")
            fluxes = model.addVars(n,m, vtype=gp.GRB.CONTINUOUS, name="f")

            # Priority of all roads sum to 1
            # Maybe add constraint on minimum value of priority parameters
            model.addConstrs(gp.quicksum(beta[i,j] for i in range(n)) == 1 for j in range(m))

            # Need auxiliary variables for alpha*D and beta*S
            alpha_D = model.addVars(n,m, vtype=gp.GRB.CONTINUOUS, name="alpha_D")
            beta_S = model.addVars(n,m, vtype=gp.GRB.CONTINUOUS, name="beta_S")

            # Do multiplication
            model.addConstrs(alpha_D[i,j] == active[i][j]*alpha[i][j]*max_dens_in[i]*D(rho_in[i], gamma_in[i]) for i in range(n) for j in range(m))
            model.addConstrs(beta_S[i,j] == beta[i,j]*max_dens_out[j]*S(rho_out[j], gamma_out[j]) for i in range(n) for j in range(m))

            # Calculate fluxes
            model.addConstrs(fluxes[i,j] == gp.min_(alpha_D[i,j], beta_S[i,j]) for i in range(n) for j in range(m))

            obj = model.addVar(vtype = gp.GRB.CONTINUOUS, name = "obj")
            model.addConstr(obj == gp.quicksum(fluxes[i,j] for i in range(n) for j in range(m)))
            # Setting objective value
            model.setObjective(obj, gp.GRB.MAXIMIZE)

            model.optimize()

            beta_out = [[beta[i,j].X for j in range(m)] for i in range(n)]
            fluxes_out = [[fluxes[i,j].X for j in range(m)] for i in range(n)]

            return beta_out, fluxes_out

