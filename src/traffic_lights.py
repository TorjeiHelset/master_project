import torch

def period(x, c1, c2):
    '''
    Used for activation function for one period
    '''
    return torch.sigmoid(x - (c1 + 5)) - torch.sigmoid(x - (c1 + c2 + 5))

def jump(x, start):
    return torch.sigmoid(x - (start + 5)) # Plus 5 to shift start of sigmoid to start

def full_jump(x, start, t):
    return jump(x, start) - jump(x, start + t)
    
class TrafficLightContinous:
    '''
    Simple traffic light connecting some incoming roads with some outgoing roads.
    Should be a member of the Junction class.
    If the light is red, no cars will cross the junction, if the light is green 
    cars will cross the junction.

    The state of the traffic light is given by a cycle. This cycle can be as simple as 
    having a fixed time of green and then a fixed time of red and repeating these times.
    It can also be a more complex cycle


    State False means green so cars can cross junction
    State True means red so no cars can move

    Change from implementation above to let traffic light be partially on and 
    partially off using linear combination of sigmoid functions

    No delay here?
    '''

    starting_state = None # The current state of the light - maybe not needed
    entering = []
    leaving = []
    cycle = [] 
    cycle_time = None
    activation_func = None # Function to use for calculating flux through junction

    def __init__(self, starting_state, entering, leaving, cycle):
        self.starting_state = starting_state
        self.entering = entering
        self.leaving = leaving
        assert(len(cycle)%2 == 0) # Need as many times for on as for off
        assert(len(cycle) >= 2)
        # First time in cycle will be time spent in first state
        self.cycle_time = sum(cycle)
        # self.cycle = [torch.tensor(c, requires_grad=True) for c in cycle]  
        self.cycle = cycle
            


    def init_activation_function(self, T):
        '''
        Function for initializing the activation function
        '''

        def full_function(x):
            '''
            Repeating a period until time T is reached
            '''
            c1 = self.cycle[0]
            c2 = self.cycle[1]

            period_time = c1.detach() + c2.detach() # Time of a single period

            n_periods = int(torch.ceil(T / period_time)) # Number of periods the time interval spans

            out = period(x, c1, c2)

            for i in range(1, n_periods):
                out += period(x, i*period_time + c1, c2)

            if self.starting_state:
                # Starting as green
                return out
            else:
                # Starting as red
                return 1. - out
        
        self.activation_func = full_function
    
    
class CoupledTrafficLightContinuous:
    '''
    Class for coupled redlights
    For now only support one to one coupling between trafficlights.

    This class implicitely defines two traffic lights

    Can extend for more complex relations between traffic lights

    For now only use one activation function for both a and b
    Should probably extend to two function. By extending to two functions
    delay between switching of lights can be incorporated.
    '''

    starting_state = None # If true, a light active from beginning, 
                          # if false, b light active
    a_entering = []
    a_leaving = []
    b_entering = []
    b_leaving = []
    cycle = []
    # activation_func = None # Function to use for calculating flux through junction
    a_activation = None
    b_activation = None

    delaytime = None

    def __init__(self, starting_state, a_in, a_out, b_in, b_out, cycle, delaytime=5):
        self.starting_state = starting_state
        self.a_entering = a_in
        self.b_entering = b_in
        self.a_leaving = a_out
        self.b_leaving = b_out
        assert(len(cycle)%2 == 0) # Need as many times for on as for off
        assert(len(cycle) >= 2)
        # First time in cycle will be time spent in first state
        self.cycle_time = sum(cycle)
        # self.cycle = [torch.tensor(c, requires_grad=True) for c in cycle]
        self.cycle = cycle
        self.delaytime = delaytime # Number of seconds where both lights are red

    
    def init_activation_function(self, T):
        '''
        Function for initializing the activation function
        Should initialize two activation functions so that both are not acitve at the same time
        '''

        def full_function(x):
            '''
            Repeating a period until time T is reached

            Instead of defining a linear combination of sigmoids,
            can maybe just do one period, and then project values into starting interval
            '''
            c1 = self.cycle[0]
            c2 = self.cycle[1]

            period_time = c1.detach() + c2.detach() # Time of a single period

            n_periods = int(torch.ceil(T / period_time)) # Number of periods the time interval spans

            out = period(x, c1, c2)

            for i in range(1, n_periods):
                out += period(x, i*period_time + c1, c2)

            if self.starting_state:
                # Starting as red
                return out
            else:
                # Starting as green
                return 1. - out
        
        self.activation_func = full_function

    
    def init_activation_function2(self, T):
        '''
        Function for initializing the activation function
        Should initialize two activation functions so that both are not acitve at the same time
        '''
        
        #########################################################
        # STEP 2: Initialize the activation functions
        #########################################################


        if self.starting_state:
            # This is wrong

            # A state is first state
            def a_activation_func(x):
                t1 = self.cycle[0]
                t2 = self.cycle[1]
                # dt : time between one light switching to yellow before the other light 
                # turns green - fix to 15 for now
                dt = 10 + self.delaytime

                #########################################################
                # STEP 1: Calculate the number of full periods to be used
                #########################################################

                period_time = t1.detach() + dt + t2.detach() + dt
                # Rounding the number of periods up to not add activation function
                # over a too short period
                n_periods = int(torch.ceil( T / period_time))
                

                # Add first period no matter what
                out = full_jump(x, torch.tensor([0]), t1)

                for i in range(1, n_periods):
                    # Start from 1 since first period already added
                    # If a new period is started on, add the whole period
                    out += full_jump(x, i*period_time, t1)
                return out

            def b_activation_func(x):
                t1 = self.cycle[0]
                t2 = self.cycle[1]
                # dt : time between one light switching to yellow before the other light 
                # turns green - fix to 15 for now
                dt = 10 + self.delaytime

                #########################################################
                # STEP 1: Calculate the number of full periods to be used
                #########################################################

                period_time = t1.detach() + dt + t2.detach() + dt
                # Rounding the number of periods up to not add activation function
                # over a too short period
                n_periods = int(torch.ceil( T / period_time))

                # Add first jump no matter what
                out = full_jump(x, t1+dt, t2)

                for i in range(1, n_periods):
                    # Start from 1 since first period already added
                    # If a new period is started on, add the whole period
                    out += full_jump(x, i*period_time + t1+dt, t2)
                return out
            
            self.a_activation = a_activation_func
            self.b_activation = b_activation_func
            
        else:
            # B state is first state
            # Swap functions above
            def a_activation_func(x):
                t1 = self.cycle[0]
                t2 = self.cycle[1]
                # dt : time between one light switching to yellow before the other light 
                # turns green - fix to 15 for now
                dt = 10 + self.delaytime

                #########################################################
                # STEP 1: Calculate the number of full periods to be used
                #########################################################

                period_time = t1.detach() + dt + t2.detach() + dt
                # Rounding the number of periods up to not add activation function
                # over a too short period
                n_periods = int(torch.ceil( T / period_time))

                # Add first period no matter what
                out = full_jump(x, t2+dt, t1)

                for i in range(1, n_periods):
                    # Start from 1 since first period already added
                    # If a new period is started on, add the whole period
                    out += full_jump(x, i*period_time+t2+dt, t1)
                return out
            
            def b_activation_func(x):
                t1 = self.cycle[0]
                t2 = self.cycle[1]
                # dt : time between one light switching to yellow before the other light 
                # turns green - fix to 15 for now
                dt = 10 + self.delaytime

                #########################################################
                # STEP 1: Calculate the number of full periods to be used
                #########################################################

                period_time = t1.detach() + dt + t2.detach() + dt
                # Rounding the number of periods up to not add activation function
                # over a too short period
                n_periods = int(torch.ceil( T / period_time))
                # Add first jump no matter what
                out = full_jump(x, torch.tensor([0]), t2)

                for i in range(1, n_periods):
                    # Start from 1 since first period already added
                    # If a new period is started on, add the whole period
                    out += full_jump(x, i*period_time, t2)
                return out
            
            self.a_activation = a_activation_func
            self.b_activation = b_activation_func

