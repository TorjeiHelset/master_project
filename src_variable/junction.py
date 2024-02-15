import torch
import FV_schemes as fv
import src_old_code.optimize_flux as opt

# Maybe want to use torch.jit.script to speed up code, but not possible for member functions
# In that case, it would be necessary to define a function outside of the class, call it 
# from the member function and send more of the member variables as arguments to the 
# function

class Junction:
    # Allow for roads to have different flux function
    roads = None # All roads that pass through Junction
    entering = None # Index of roads going in
    leaving = None # Index of roads going out
    priority = None # Assume priority equal for all roads
    distribution = None # Assume all roads have same distribution

    trafficlights = None # Define all traffic lights in junction
    coupled_trafficlights = None # Define all coupled traffic lights in junction


    def __init__(self, roads, entering, leaving, distribution, trafficlights, coupled_trafficlights):
        # Make sure roads are either entering or leaving
        assert set(entering).isdisjoint(leaving)
        # Make sure all roads actually cross junction
        assert len(roads) == len(entering) + len(leaving)

        # Check that all roads have correct position of junction
        # junction_pos = roads[entering[0]].right_pos
        # for i in entering:
        #     assert roads[i].right_pos == junction_pos
            
        # for j in leaving:
        #     assert roads[j].left_pos == junction_pos


        # Make sure distribution is of correct dimension

        assert len(distribution[0]) == len(leaving)
        # Make sure distribution sums to 1
        # assert abs(sum(distribution) -1 ) <= 1e-4
        for i in range(len(distribution)):
            # For every incoming road all of the flux should be distributed
            # Distribution changed to be 2d array
            assert abs(sum(distribution[i]) -1) <= 1e-4

        for trafficlight in trafficlights:
            # Check that traffic light only contains roads in junction
            assert set(trafficlight.entering).issubset(set(entering))
            assert set(trafficlight.leaving).issubset(set(leaving))

        for coupled in coupled_trafficlights:
            # Check that the coupled traffic light only contains roads in junction
            assert set(coupled.a_entering).issubset(set(entering))
            assert set(coupled.b_entering).issubset(set(entering))
            assert set(coupled.a_leaving).issubset(set(leaving))
            assert set(coupled.b_leaving).issubset(set(leaving))

        self.roads = roads
        self.entering = entering
        for i in self.entering:
            self.roads[i].right = True

        self.leaving = leaving
        for j in self.leaving:
            self.roads[j].left = True

        self.distribution = distribution
        self.priority = [1/len(entering)] * len(entering)

        self.trafficlights = trafficlights
        self.coupled_trafficlights = coupled_trafficlights
        self.road_in = [self.roads[i] for i in self.entering]
        self.road_out = [self.roads[i] for i in self.leaving]

    def divide_flux(self, t):
        '''
        When comparing the fluxes from different roads, use 
        gamma * f(rho) instead of f(rho)
        idx is index of speed limit to be used


        Optimization problems:
        
        The function uses a lot of for loops, which could potentially be avoided
        Should probably split into seperate functions to find out where exactly the time is 
        being spent

        The optimization part - find_parameters uses the most of the time
        This is difficult to optimize
        '''

        # Some of these probably also don't need to be created every time...
        rho_in = [road.rho[-road.pad] for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        max_dens_in = [road.max_dens for road in self.road_in]
        rho_out = [road.rho[road.pad-1] for road in self.road_out]
        gamma_out = [road.gamma[road.idx] for road in self.road_out]
        max_dens_out = [road.max_dens for road in self.road_out]

        # Can below code be optimized in any meaningfull way(?)
        # Would maybe make it very unreadable...

        n = len(self.entering)
        m = len(self.leaving)
        active = torch.ones((n,m))
        for light in self.trafficlights:
            for i in range(n):
                for j in range(m):
                    if self.entering[i] in light.entering and self.leaving[j] in light.leaving:
                        active[i,j] = light.activation_func(t)
        
        for light in self.coupled_trafficlights:
            for i in range(n):
                for j in range(m):
                    if self.entering[i] in light.a_entering and self.leaving[j] in light.a_leaving:
                        active[i,j] = light.a_activation(t)

                    if self.entering[i] in light.b_entering and self.leaving[j] in light.b_leaving:
                        active[i,j] = light.b_activation(t)

        ####################################################
        # Instead of returning fluxes, return beta parameters
        # Use beta parameters to calculate fluxes
        #####################################################
        beta, _ = opt.find_parameters(rho_in, rho_out, self.distribution, gamma_in, gamma_out, active,
                                        max_dens_in, max_dens_out)
        fluxes = torch.zeros((n, m))
        for i in range(n):
            for j in range(m):
                fluxes[i,j] = min(active[i,j]*self.distribution[i][j]*max_dens_in[i] * fv.D(rho_in[i].clone(), gamma_in[i]),
                                  beta[i][j]*max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j]))

        fluxes_in = [0]*n
        fluxes_out = [0]*m

        for i in range(n):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_in[i] = sum([fluxes[i][j] for j in range(m)]) / max_dens_in[i]

        for j in range(m):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_out[j] = sum([fluxes[i][j] for i in range(n)]) / max_dens_out[j]
        
        return fluxes_in, fluxes_out

    # @torch.jit.script
    def divide_flux_wo_opt(self, t):
        rho_in = [road.rho[-road.pad] for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        max_dens_in = torch.tensor([road.max_dens for road in self.road_in])
        rho_out = [road.rho[road.pad-1] for road in self.road_out]
        gamma_out = [road.gamma[road.idx] for road in self.road_out]
        max_dens_out = torch.tensor([road.max_dens for road in self.road_out])


        # If sentences can be moved outside of j for loop and split into two if-sentences
        # Would reducde the number of times the if-sentence is evaluated
        n = len(self.entering)
        m = len(self.leaving)
        active = torch.ones((n,m))

        # Probably quicker - need to check that the functionality is the same!
        for light in self.trafficlights:
            for i in range(n):
                if self.entering[i] in light.entering:
                    for j in range(m):
                        if self.leaving[j] in light.leaving:
                            active[i,j] = light.activation_func(t)

        
        for light in self.coupled_trafficlights:
            for i in range(n):
                if self.entering[i] in light.a_entering:
                    for j in range(m):
                        if self.leaving[j] in light.a_leaving:
                            active[i,j] = light.a_activation(t)

                if self.entering[i] in light.b_entering:
                    for j in range(m):
                        if self.leaving[j] in light.b_leaving:
                            active[i,j] = light.b_activation(t)
        
        # fluxes[i,j] is the flux from road i to road j
        fluxes = torch.zeros((n, m))

        # Calculate the desired flux from each road i to road j
        for i in range(n):
            # move D to be here to reduce the number of calls
            i_flux = fv.D(rho_in[i].clone(), gamma_in[i])            
            for j in range(m):
                fluxes[i,j] = active[i,j]*self.distribution[i][j]*max_dens_in[i] * i_flux
        
        
        cloned_fluxes = fluxes.clone()
    
        for j in range(m):
            capacity = max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j])

            # Update the flux from all roads into road j
            sum_influx = torch.sum(fluxes[:,j])
            if sum_influx > capacity:
                # If the sum of the fluxes is larger than the capacity, scale down the fluxes
                cloned_fluxes[:,j] = fluxes[:,j] * capacity / sum_influx
                # for i in range(n):
                #     cloned_fluxes[i,j] = fluxes[i,j] * capacity / sum_influx


        fluxes_in = [0]*n
        fluxes_out = [0]*m


        for i in range(n):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_in[i] = torch.sum(cloned_fluxes[i]) / max_dens_in[i]


        for j in range(m):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_out[j] = torch.sum(cloned_fluxes[:,j]) / max_dens_out[j]
        
        return fluxes_in, fluxes_out
    

    def apply_bc(self, dt, t):
        '''
        Calculate how flux is divided among roads
        To this end gamma of each road is important
        Actual flux is gamma*f(rho) instad of just f(rho)
        ...
        '''
        #--------------------------------
        # Dividing flux need to somehow take into account gamma of each road
        # --------------------------------

        # Need to change divide_flux3 so that the fluxes depend on parameter, i.e. are differentiable
        fluxes_in, fluxes_out = self.divide_flux(t)
        # outputed fluxes should now be gamma*f(rho)


        #---------------------------------------------
        # Note not exactly correct: using time tn+1 to do updating, but should actually use
        # time tn
        # Maybe not a very big problem ????
        # Solution: save out/in flux at previous step somewhere
        #           not very difficult so should maybe just do this
        #---------------------------------------------

        # Can this be done without for loop?
        # Most of the time save is inside divide_flux anyways

        # These two codeblocks take up over half of the running time of the entire code
        # It is very worthwhile to try to vectorize these if possible bc this is where most of the time save is
        # Don't spend time on this rn...

        for i, flux in enumerate(fluxes_in):
            road = self.road_in[i]
            left, in_mid = road.rho[-road.pad-1], road.rho[-road.pad]
            s = torch.max(torch.abs(fv.d_flux(left, road.gamma[road.idx])), torch.abs(fv.d_flux(in_mid, road.gamma[road.idx])))
            left_f = fv.flux(left.clone(), road.gamma[road.idx])
            mid_f = fv.flux(in_mid.clone(), road.gamma[road.idx])
            left_flux = 0.5 * (left_f + mid_f) - 0.5 * s * (in_mid - left)

            # Don't multiply with gamma in denominator because flux is already multiplied with
            # gamma
            road.rho[-road.pad] = road.rho[-road.pad] - dt/road.dx * (flux - left_flux)
            if road.pad > 1:
                road.rho[-road.pad+1] = road.rho[-road.pad]
        
        for i, flux in enumerate(fluxes_out):
            road = self.road_out[i]

            right, out_mid = road.rho[road.pad], road.rho[road.pad-1]
            s = torch.max(torch.abs(fv.d_flux(out_mid, road.gamma[road.idx])), torch.abs(fv.d_flux(right, road.gamma[road.idx])))
            mid_f = fv.flux(out_mid.clone(), road.gamma[road.idx])
            right_f = fv.flux(right.clone(), road.gamma[road.idx])
            right_flux = 0.5 * (mid_f + right_f) - 0.5 * s * (right - out_mid)

            road.rho[road.pad-1] = road.rho[road.pad-1] - dt / road.dx * (right_flux - flux)
            if road.pad > 1:
                road.rho[0] = road.rho[1]

    # @torch.jit.script
    def apply_bc_wo_opt(self, dt, t):
        '''
        To save time, assume that the flux is distributed equally among roads
        Could alternatively make the user specify the distribution
        '''
        # Also not necessary to create every time, instead store as member
        # variable!

        fluxes_in, fluxes_out = self.divide_flux_wo_opt(t)

        # Ideally want to reduce the number of calls to d_flux and flux
        # The code below is rather slow, but difficult to optimize
        
        for i, flux in enumerate(fluxes_in):
            road = self.road_in[i]
            left, in_mid = road.rho[-road.pad-1], road.rho[-road.pad]
            
            s = torch.max(torch.abs(fv.d_flux(left, road.gamma[road.idx])), torch.abs(fv.d_flux(in_mid, road.gamma[road.idx])))
            left_f = fv.flux(left.clone(), road.gamma[road.idx])
            mid_f = fv.flux(in_mid.clone(), road.gamma[road.idx])
            left_flux = 0.5 * (left_f + mid_f) - 0.5 * s * (in_mid - left)

            # Don't multiply with gamma in denominator because flux is already multiplied with
            # gamma
            road.rho[-road.pad] = road.rho[-road.pad] - dt/road.dx * (flux - left_flux)
            if road.pad > 1:
                road.rho[-road.pad+1] = road.rho[-road.pad]
        
        for i, flux in enumerate(fluxes_out):
            road = self.road_out[i]

            right, out_mid = road.rho[road.pad], road.rho[road.pad-1]
            s = torch.max(torch.abs(fv.d_flux(out_mid, road.gamma[road.idx])), torch.abs(fv.d_flux(right, road.gamma[road.idx])))
            mid_f = fv.flux(out_mid.clone(), road.gamma[road.idx])
            right_f = fv.flux(right.clone(), road.gamma[road.idx])
            right_flux = 0.5 * (mid_f + right_f) - 0.5 * s * (right - out_mid)

            road.rho[road.pad-1] = road.rho[road.pad-1] - dt / road.dx * (right_flux - flux)
            if road.pad > 1:
                road.rho[0] = road.rho[1]
        

        

    def get_next_control_point(self, t):
        '''
        Given a time t, this function returns the next time where a jump occurs
        It should also maybe return some of the points in the jump itself to capture
        the full change in state


        For some reason gets stuck whenever t1 or t2 are float ... Why????
        '''
        if torch.is_tensor(t):
            t = t.detach()

        control_point = t + 100 # Just set to some value so that actual control points will be considered

        for light in self.trafficlights:
            t1 =  torch.round(light.cycle[0].detach(), decimals=0)
            t2 =  torch.round(light.cycle[1].detach(), decimals=0)
            jump_time = 10
            period_time = t % (t1 + t2)
            if period_time < jump_time/2:
                control_point = min(control_point, t-period_time+jump_time/2)
            elif period_time < jump_time:
                control_point = min(control_point, t-period_time+jump_time)
            elif period_time < t1:
                control_point = min(control_point, t-period_time+t1)
            elif period_time < t1 + jump_time/2:
                control_point = min(control_point, t-period_time+t1+jump_time/2)
            elif period_time < t1 + jump_time:
                control_point = min(control_point, t-period_time+t1+jump_time)
            else:
                control_point = min(control_point, t-period_time+t1+t2)

        for light in self.coupled_trafficlights:
            # Maybe add some extra point here if two activation functions are used
            t1 =  torch.round(light.cycle[0].detach(), decimals=0)
            t2 =  torch.round(light.cycle[1].detach(), decimals=0)
            jump_time = 10
            period_time = t % (t1 + t2)
            if period_time < jump_time/2:
                control_point = min(control_point, t-period_time+jump_time/2)
            elif period_time < jump_time:
                control_point = min(control_point, t-period_time+jump_time)
            elif period_time < t1:
                control_point = min(control_point, t-period_time+t1)
            elif period_time < t1 + jump_time/2:
                control_point = min(control_point, t-period_time+t1+jump_time/2)
            elif period_time < t1 + jump_time:
                control_point = min(control_point, t-period_time+t1+jump_time)
            else:
                control_point = min(control_point, t-period_time+t1+t2)
            
        return control_point

    def get_activation(self, t, id_1, id_2):
        '''
        Given a time t, this function returns the activation of roads with id_1 and id_2
        '''
        # First check if both id_1 and id_2 are a part of the junction
        id_1_in = False
        in_idx = -1
        for i in self.entering:
            if self.roads[i].id == id_1:
                id_1_in = True
                in_idx = i
                break
        
        id_2_in = False
        out_idx = -1
        for i in self.leaving:
            if self.roads[i].id == id_2:
                id_2_in = True
                out_idx = i
                break
        
        if not id_1_in or not id_2_in:
            return False, 0.0
        
        else:
            # id_1 is entering into the junction, id_2 is leaving
            # Assume that this specific crossing is either combined with a traffic light or a 
            # coupled traffic light, but not both
            activation = 1.0
            for light in self.trafficlights:
                if in_idx in light.entering and out_idx in light.leaving:
                    activation = light.activation_func(t)
                    # print("Regular traffic light")
                    return True, activation

            for light in self.coupled_trafficlights:
                if in_idx in light.a_entering and out_idx in light.a_leaving:
                    activation = light.a_activation(t)
                    # print("Coupled traffic light, state a")
                    return True, activation

                if in_idx in light.b_entering and out_idx in light.b_leaving:
                    activation = light.b_activation(t)
                    return True, activation
        return True, activation
    
    def get_speed(self, t, id_1, id_2):
        '''
        Need to reevaluate how the flux is divided among the roads
        -> Change this to avoid reevaluations...
        '''
        idx_1 = 0
        idx_2 = 0
        for i, road in enumerate(self.road_in):
            if road.id == id_1:
                idx_1 = i
                break
        for i, road in enumerate(self.road_out):
            if road.id == id_2:
                idx_2 = i
                break
        

        rho_in = [road.rho[-road.pad] for road in self.road_in]
        gamma_in = [road.gamma[road.idx] for road in self.road_in]
        max_dens_in = torch.tensor([road.max_dens for road in self.road_in])
        rho_out = [road.rho[road.pad-1] for road in self.road_out]
        gamma_out = [road.gamma[road.idx] for road in self.road_out]
        max_dens_out = torch.tensor([road.max_dens for road in self.road_out])

        n = len(self.entering)
        m = len(self.leaving)
        active = torch.ones((n,m))

        for light in self.trafficlights:
            for i in range(n):
                if self.entering[i] in light.entering:
                    for j in range(m):
                        if self.leaving[j] in light.leaving:
                            active[i,j] = light.activation_func(t)

        
        for light in self.coupled_trafficlights:
            for i in range(n):
                if self.entering[i] in light.a_entering:
                    for j in range(m):
                        if self.leaving[j] in light.a_leaving:
                            active[i,j] = light.a_activation(t)

                if self.entering[i] in light.b_entering:
                    for j in range(m):
                        if self.leaving[j] in light.b_leaving:
                            active[i,j] = light.b_activation(t)

        fluxes = torch.zeros((n, m))

        # Calculate the desired flux from each road i to road j
        for i in range(n):
            # move D to be here to reduce the number of calls
            i_flux = fv.D(rho_in[i].clone(), gamma_in[i])            
            for j in range(m):
                # print(f"Desired flux from road {i} to road {j}: {self.distribution[i][j]* i_flux * max_dens_in[i]}")
                # print(f"Flux scaled with traffic light from road {i} to road {j}: {active[i,j]*self.distribution[i][j] * i_flux * max_dens_in[i]}")
                fluxes[i,j] = active[i,j]*self.distribution[i][j]*max_dens_in[i] * i_flux
        
        cloned_fluxes = fluxes.clone()
        # calculate the capacity of each road j
        for j in range(m):
            # print(f"Capacity of road {j}: {max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j])}")
            capacity = max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j])

            # Update the flux from all roads into road j
            sum_influx = torch.sum(fluxes[:,j])
            # print(f"Total sum of fluxes into road {j}: {sum_influx}")
            if sum_influx > capacity:
                # If the sum of the fluxes is larger than the capacity, scale down the fluxes
                cloned_fluxes[:,j] = fluxes[:,j] * capacity / sum_influx # is this the inplace in question?
                
        flux = cloned_fluxes[idx_1, idx_2] / max_dens_in[idx_1]
        speed = flux / rho_in[idx_1] * self.road_in[idx_1].L
        return speed