def divide_flux2(self, t):
        '''
        When comparing the fluxes from different roads, use 
        gamma * f(rho) instead of f(rho)
        idx is index of speed limit to be used
        '''
        road_in = [self.roads[i] for i in self.entering]
        road_out = [self.roads[i] for i in self.leaving]

        rho_in = [road.rho[-road.pad] for road in road_in]
        gamma_in = [road.gamma[road.idx] for road in road_in]
        max_dens_in = [road.max_dens for road in road_in]
        rho_out = [road.rho[road.pad-1] for road in road_out]
        gamma_out = [road.gamma[road.idx] for road in road_out]
        max_dens_out = [road.max_dens for road in road_out]

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

        #####################################################################
        # The distribution should be replaced by a 2d array
        #
        # The flux into the junction should be scaled by the relative maximum 
        # capacity. 
        # This should be done for flux into junction as f = f*rho_max
        # and it should be scaled back when updating the densities of the roads
        # I.e. the output from this function should not be f but f/rho_max since
        # each lane is scaled to have density between 0 and 1.
        #
        # Priority should also probably be a 2d matrix that should be found by
        # solving small optimization step 
        ######################################################################
        fluxes = torch.zeros((n, m))
        for i in range(n):
            for j in range(m):
                fluxes[i,j] = min(active[i,j]*self.distribution[i][j] * max_dens_in[i] * fv.D(rho_in[i].clone(), gamma_in[i]),
                                self.priority[i]* max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j]))
                # fluxes[i,j] = min(active[i,j]*self.distribution[i][j] * fv.D(rho_in[i].clone(), gamma_in[i]),
                #                 self.priority[i]* fv.S(rho_out[j].clone(),  gamma_out[j]))
                
        fluxes_in = [0]*n
        fluxes_out = [0]*m

        for i in range(n):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_in[i] = sum([fluxes[i,j] for j in range(m)]) / max_dens_in[i]

        for j in range(m):
            # Scaling flux back to correspond to maximum density equal to 1
            fluxes_out[j] = sum([fluxes[i,j] for i in range(n)]) / max_dens_out[j]
        
        return fluxes_in, fluxes_out


 def divide_flux_wo_opt_list(self, t):
        # Is all of the below necessary? 

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

        active = [[torch.tensor(0.0) for _ in range(m)] for _ in range(n)]

        # Probably quicker - need to check that the functionality is the same
        for light in self.trafficlights:
            for i in range(n):
                if self.entering[i] in light.entering:
                    for j in range(m):
                        if self.leaving[j] in light.leaving:
                            active[i][j] = light.activation_func(t)

        
        for light in self.coupled_trafficlights:
            for i in range(n):
                if self.entering[i] in light.a_entering:
                    for j in range(m):
                        if self.leaving[j] in light.a_leaving:
                            active[i][j] = light.a_activation(t)

                if self.entering[i] in light.b_entering:
                    for j in range(m):
                        if self.leaving[j] in light.b_leaving:
                            active[i][j] = light.b_activation(t)
        
        # fluxes[i,j] is the flux from road i to road j
        fluxes = [[torch.tensor(0.0) for _ in range(m)] for _ in range(n)]

        # Calculate the desired flux from each road i to road j
        for i in range(n):
            # move D to be here to reduce the number of calls
            i_flux = fv.D(rho_in[i].clone(), gamma_in[i])            
            for j in range(m):
                fluxes[i][j] = active[i][j]*self.distribution[i][j]*max_dens_in[i] * i_flux
        
        
        # Skip this part to check if error is here...

        for j in range(m):
            capacity = max_dens_out[j] * fv.S(rho_out[j].clone(),  gamma_out[j])

            # Update the flux from all roads into road j
            sum_influx = torch.tensor(0.0)
            for i in range(n):
                sum_influx = sum_influx +  fluxes[i][j]

            if sum_influx > capacity:
                # If the sum of the fluxes is larger than the capacity, scale down the fluxes
                # fluxes[:,j] = fluxes[:,j] * capacity / sum_influx
                for i in range(n):
                    fluxes[i][j] = fluxes[i][j] * capacity / sum_influx


        fluxes_in = [0]*n
        fluxes_out = [0]*m

        for i in range(n):
            # Scaling flux back to correspond to maximum density equal to 1
            influx = torch.tensor(0.0)
            for j in range(m):
                influx = influx + fluxes[i][j]
            fluxes_in[i] = influx / max_dens_in[i]


        for j in range(m):
            # Scaling flux back to correspond to maximum density equal to 1
            outflux = torch.tensor(0.0)
            for i in range(n):
                outflux = outflux + fluxes[i][j]
            fluxes_out[j] = outflux / max_dens_out[j]
            # fluxes_out[j] = sum([fluxes[i][j] for i in range(n)]) / max_dens_out[j]
        
        return fluxes_in, fluxes_out