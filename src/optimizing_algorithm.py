###################################################
# Problem: Running out of memory if many iterations are performed
# - Need to release memory between iterations of optimization algorithm
# - Easiest way to do this is to put each iteration of the optimization algorithm inside
#   a dedicated function. When this function is exited, the memory will be released.

# No guarantee that global optimum found so multiple searches might be necessary
# - How to pick starting points?
# - Can parts of the search space be eliminated before? I.e. if the current 
#   iterate has been visited before, it will lead to the same solution, so 
#   algorithm should stop

#################################################################

# Importing necessary libraries:
import torch
...

################################
# Objective functions:
################################

def average_delay_time(bus_delays):
    avg_delay = torch.tensor(0.0)
    n_stops_reached = 0

    for delays in bus_delays:
        for delay in delays:
            if delay > 0.0:
                avg_delay = avg_delay + delay
                n_stops_reached += 1

    avg_delay = avg_delay / n_stops_reached
    return avg_delay

################################
# Gradient descent functions
################################

def jacobi(objective, params):
    '''
    Calculates the gradient of the objective function wrt the paramers.
    Is this the most efficient way to do this?
    Could maybe simply use .backward() to populate the gradients
    '''
    j = torch.zeros(len(params))

    for i in range(len(params)):
        derivative =  torch.autograd.grad(objective, params[i], create_graph=True, allow_unused=True)[0]
        if derivative:
            j[i] = derivative
    return j

def project(params, Lower, Upper):
    '''
    Sometimes the direction giving the most decrease in the object function sends 
    some parameters outside the feasible region. Whenever this happens
    these parameters should instead be chosen as the upper/lower limits
    '''
    out = params.clone()
    for i in range(len(Lower)):
        out[i] = max(Lower[i], min(Upper[i], params[i]))
    return out

def project_int(params, Lower, Upper):
    '''
    Similar function to project() with the added constraint that parameters should be integers
    '''
    out = params.clone()
    for i in range(len(Lower)):
        out[i] = int(max(Lower[i], min(Upper[i], params[i])))
    return out

def gradient_descent_first_step( ):
    '''
    First step of optimization approach. In this step there is no previous results to 
    compare with.
    '''
    # Create first network
    _, _, bus_network = ...

    # Solve conservation law
    densities, queues, bus_lengths, bus_delays = bus_network.solve_cons_law()

    # Calculate objective function to minimize
    objective = average_delay_time
    
    # Storing parameters in a list
    params = bus_network.get_tensors()

    # Calculating gradient wrt parameters
    grad = jacobi(objective, params)
    print(f"Gradient after the first step: {grad}")

    return grad.detach(), objective.detach()


def gradient_descent_step(prev_params, prev_gradient, prev_objective):
    '''
    Performs gradient step with armijo line search.
    Assumes that the first step is always so big that a necessary decrease is achieved
    If armijo condition is not satisfied, reduce step length until it is
    The first step length is chosen in such a way that the parameter being changed the 
    most is changed by some fixed number. This could i.e. be 10 or 20, both of which are 
    relevant for speed limits and cycle times. When scaling the gradient, elements too close
    to the boundary should not be used for the scaling (unless no other are non-zero).
    Needs to take in previous objective and previous gradient to be able to do line-search
    Should also take in upper/lower bounds of parameters
    Only do updating until a certain number of line search fails reached. Can i.e. stop iterating
    when the parameter being updated the most is updated by less than 1
    The type of objective function should be be specified as an argument
    '''
    armijo_fails = 0
    step_taken = False
    max_update = 20

    while not step_taken:
        # 1. Scale gradient so that parameter being updated most is updated by max_update

        # 2. Use previous gradient and previous parameters to update parameters
        new_params = ... 

        # 3. Create network using the new parameters
        bus_network = ...

        # 4. Calculate objective and gradient - this should be done in separate function so that 
        #    memory is released
        new_gradient, new_objective = ...

        # 5. Check armijo condition:
        armijo_satisfied = ...

        if armijo_satisfied:
            # If armijo condition is satisfied, the new iterate can be returned
            return new_params, new_gradient, new_objective
        
        else:
            # If armijo condition not satisfied, increment the number of fails and 
            # decrease the maximum change in parameters
            armijo_fails += 1
            max_update = max_update * 0.5

        if armijo_fails > ... or max_update < ...:
            # If either the number of fails are too many or the maximum update becomes too small
            # stop iterating and return the old iterate
            return prev_params, prev_gradient, prev_objective
        
def gradient_descent( ):
    '''
    Full method for the gradient descent algorithm. The funcion should take in a filename for 
    the initial configuration of the network. In addition the objetive type needs to be specified.

    Nothing inside this function should require gradient!
    '''

    # 1. Load configuration from filename
    prev_params, upper_bounds, lower_bounds = ...

    # 2. Do first simulation using the initial configuration
    prev_grad, prev_objective = gradient_descent_first_step( )

    # 2.1 Store results from first iteration
    params_history = [prev_params]
    grad_history = [prev_grad]
    objective_history = [prev_objective]

    # 2.2 Print results of first simulation:
    #       Give option for detailed print, and less detailed print

    # 3. Perform gradient descent steps until some criteria is reached
    curr_iter, max_iter, error, tol = ...
    while curr_iter < max_iter and error > tol:

        # 3.1 Do the gradient descnent iteration
        new_params, new_grad, new_objective = gradient_descent_step(prev_params, prev_grad, prev_objective)

        # 3.2 Print results from iteration
        #       Give option for detailed print, and less detailed print

        # 3.3 Calculate error and increment number of iterations
        curr_iter += 1
        error = ...

        # 3.4 Store the new results before doing the next iteration
        params_history.append(new_params)
        grad_history.append(new_grad)
        objective_history.append(new_objective)
        prev_params, prev_grad, prev_objective = new_params, new_grad, new_objective

    # 4. Stopping criteria reached. Return results from simulation
    return params_history, grad_history, objective_history