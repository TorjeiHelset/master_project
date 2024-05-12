import networkx as nx
import matplotlib.pyplot as plt
import torch


class RoadNetwork:
    
    # Roads have one or two ends in an junction
    # Intersection class should deal with boundary conditions on junctions
    # Ends not connected to junction need other boundary conditions
    # This should be dealt with here

    '''
    FV scheme:
        Keep track of which ends are connected to intersections and which are not
        Solve equation separately on each road, allowing for different flux functions
        and schemes
        Apply boundary conditions to all intersections
            Rankine Hugoniot condition of flux
            Maximise flux?
        Apply boundary conditions to all ends not connected to intersections
    '''


    roads = None # All roads in network
    junctions = None # All junctions in network
    busses = None # All buses in network
    roundabouts = []
    debugging = None
    object_type = None
    iters = None
    T = None
    debug_dt = None

    def __init__(self, roads, junctions, T, roundabouts = [], busses = [], debugging = False, object_type = 0, iters = 1, dt = 0.001,
                store_densities = True, print_control_points = False):
        # Check that unit length and dx are equal for all roads
        for i in range(1, len(roads)):
            assert roads[i].L == roads[i-1].L 
            assert roads[i].dx == roads[i-1].dx

        self.roads = roads
        self.junctions = junctions
        self.busses = busses
        self.roundabouts = roundabouts
        self.debugging = debugging
        self.object_type = object_type
        self.iters = iters
        self.T = T
        self.debug_dt = dt
        self.store_densities = store_densities
        self.print_control_points = print_control_points


        # Update gamma for each road
        for road in self.roads:
            road.calculate_gamma(self.T)

        # Initialize activation function for traffic lights
        for junction in self.junctions:
            for light in junction.trafficlights:
                light.init_activation_function(T)

        # Initialize activation function for coupled traffic lights
        for junction in self.junctions:
            for light in junction.coupled_trafficlights:
                light.init_activation_function2(T)

    def get_road(self, id):
        for i, road in enumerate(self.roads):
            if road.id == id:
                return i, road
        return None
    
    def update_position_of_bus(self, bus, dt, t, slowdown_factors):
        # Also update member function of thebus that tells the bus how much it 
        # should slow down
        if t < bus.start_time:
            # Bus has not started its route yet
            return slowdown_factors, None
        
        # 1. Find the road the bus is on
        road_id, length, next_id = bus.get_road_id()

        if road_id == "":
            # Bus has reached the end of the route
            # Stop updating the bus
            return slowdown_factors, None
        
        i, road = self.get_road(road_id)
        slowing_idx = None
        # i used to find out what slowdown_factors to modify

        #---------------------------------------------------
        # SLOWDOWN FACTORS:
        #---------------------------------------------------

        # 2. Use position of bus to calculate the slowdown factors
        slowdown_factors[i], bus_started, stop_factor = bus.get_slowdown_factor(slowdown_factors[i].clone(), road_id,
                                                      length, road)
        if bus_started:
            slowing_idx = i
        
        # Now use either maximum density or road in other direction
        # to update the slowdown factor
        if road.max_dens == 1:
            # Look for lane going in other direction, and 
            # use density/flux on that lane to update the
            # slowdown factors
            # Check roads close to index i
            opposite_road = None
            if road.id[-2:] == "fw" or road.id[-2:] == "bw":
                if i > 0: 
                    if self.roads[i-1].id[:-2] == road.id[:-2]:
                        opposite_road = self.roads[i-1]
                if i < len(self.roads)-1:
                    if self.roads[i+1].id[:-2] == road.id[:-2]:
                        opposite_road = self.roads[i+1]

            if opposite_road is not None:
                # Opposite road found - use densities on this road
                # to update the slowdown factor
                # If fully congested, then let slowdown factor be as 
                # before
                # If empty, reduce slowdown factor by 50(?)%
                # Linear interpolation between the two
                # Check that the number of nodes is equal for the two roads:
                if len(opposite_road.rho) != len(road.rho):
                    print(f"Roads {road.id} and {opposite_road.id} are of different sizes!")


                # internal = torch.flip(opposite_road.rho[opposite_road.pad-1:-opposite_road.pad+1],dims=[0])
                internal = torch.flip(opposite_road.rho, dims=[0])
                avg = (internal[1:] + internal[:-1]) / 2
                slowdown_factors[i] = torch.ones(road.N_full-1) - (1. - slowdown_factors[i]) * (1. + avg) / 2

        else:
            # Current road has more lanes
            # use only density/flux on this road to update the slowdown
            # factors
            
            n_extra_lanes = road.max_dens - 1
            # internal = torch.flip(road.rho[road.pad-1:-road.pad+1],dims=[0])
            avg = (road.rho[1:] + road.rho[:-1]) / 2
            slowdown_factors[i] = torch.ones(road.N_full-1) - (1. - slowdown_factors[i]) * (1. + avg) / (2**n_extra_lanes)
        
        #------------------------------------------------------------
        # SLOWDOWN FACTORS FINISHED
        #------------------------------------------------------------

        new_length = length / road.L
        # speed = road.get_speed(new_length) * road.L
        # Use the distance to the bus stop to calculate the slowdown
        # Also multiply by 1. - stop_factor to get the slowdown of the bus
        speed_, road_activation = road.get_speed(new_length, dt)

        speed = speed_ * road.L * (1.0 - stop_factor)


        activation = torch.tensor(1.0)
        # 3. Find the the junction (and traffic light) that connects the two roads
        # This is actually not necessary unless the bus has almost reached the end of the road
        # Add some check on the length here
        if next_id == "" or length + speed * dt < road.L*road.b - road.dx*road.L: # and length < ...
            # Road_id is at the last road, don't need to find the next junction
            # or bus is not close to the junction
            pass
        else:
            # Find the junction and the activation function of the two roads
            # If the activation is above i.e. 0.5, the bus can cross the junction
            # If the activation is below, the bus cannot go further than the junction
            for j in self.junctions:
                check, activ_ = j.get_activation(t, road_id, next_id)
                if check:
                    activation = activ_
                    break

            # At this point both the activation from the distance to the junction the activation from 
            # the traffic light at the junction is calculated
            combined_activation = torch.max(road_activation, activation)

            speed = speed * combined_activation

            # Check if the density of a the next road is needed
            if length + speed * dt >= road.L * road.b:
                for j in self.junctions:
                    if j.check_roads_contained(road_id, next_id):
                        jnc_speed = torch.maximum((road.L*road.b - length) / dt - 0.0001,
                                                  j.get_speed(t, road_id, next_id))
                        speed = torch.min(speed, jnc_speed)

        relative_length = road.L*road.b - length # Remaining length
        # bus.update_position(t.clone(), dt.clone(), speed, activation, relative_length, printing=False)
        bus.update_position(dt, t, speed, activation, relative_length, printing=False)
        return slowdown_factors, slowing_idx
    
    def update_position_of_bus_restarting(self, bus, dt, t, slowdown_factors):
        # Also update member function of thebus that tells the bus how much it 
        # should slow down
        if t < bus.start_time:
            # Bus has not started its route yet
            return slowdown_factors, None, False, None
        
        # 1. Find the road the bus is on
        road_id, length, next_id = bus.get_road_id()

        if road_id == "":
            # Bus has reached the end of the route
            # Stop updating the bus
            return slowdown_factors, None, False, None
        
        i, road = self.get_road(road_id)
        slowing_idx = None
        # i used to find out what slowdown_factors to modify

        #---------------------------------------------------
        # SLOWDOWN FACTORS:
        #---------------------------------------------------

        # 2. Use position of bus to calculate the slowdown factors
        slowdown_factors[i], bus_started, stop_factor = bus.get_slowdown_factor(slowdown_factors[i].clone(), road_id,
                                                      length, road)
        if bus_started:
            slowing_idx = i
        
        # Now use either maximum density or road in other direction
        # to update the slowdown factor
        if road.max_dens == 1:
            # Look for lane going in other direction, and 
            # use density/flux on that lane to update the
            # slowdown factors
            # Check roads close to index i
            opposite_road = None
            if road.id[-2:] == "fw" or road.id[-2:] == "bw":
                if i > 0: 
                    if self.roads[i-1].id[:-2] == road.id[:-2]:
                        opposite_road = self.roads[i-1]
                if i < len(self.roads)-1:
                    if self.roads[i+1].id[:-2] == road.id[:-2]:
                        opposite_road = self.roads[i+1]

            if opposite_road is not None:
                # Opposite road found - use densities on this road
                # to update the slowdown factor
                # If fully congested, then let slowdown factor be as 
                # before
                # If empty, reduce slowdown factor by 50(?)%
                # Linear interpolation between the two
                # Check that the number of nodes is equal for the two roads:
                if len(opposite_road.rho) != len(road.rho):
                    print(f"Roads {road.id} and {opposite_road.id} are of different sizes!")

                # internal = torch.flip(opposite_road.rho[opposite_road.pad-1:-opposite_road.pad+1],dims=[0])
                internal = torch.flip(opposite_road.rho,dims=[0])
                avg = (internal[1:] + internal[:-1]) / 2
                slowdown_factors[i] = torch.ones(road.N_full-1) - (1. - slowdown_factors[i]) * (1. + avg) / 2

        else:
            # Current road has more lanes
            # use only density/flux on this road to update the slowdown
            # factors
            n_extra_lanes = road.max_dens - 1
            avg = (road.rho[1:] + road.rho[:-1]) / 2
            slowdown_factors[i] = torch.ones(road.N_full-1) - (1. - slowdown_factors[i]) * (1. + avg) / (2**n_extra_lanes)
        
        #------------------------------------------------------------
        # SLOWDOWN FACTORS FINISHED
        #------------------------------------------------------------

        new_length = length / road.L
        # speed = road.get_speed(new_length) * road.L
        # Use the distance to the bus stop to calculate the slowdown
        # Also multiply by 1. - stop_factor to get the slowdown of the bus
        speed_, road_activation = road.get_speed(new_length, dt)

        speed = speed_ * road.L * (1.0 - stop_factor)


        activation = torch.tensor(1.0)
        # 3. Find the the junction (and traffic light) that connects the two roads
        # This is actually not necessary unless the bus has almost reached the end of the road
        # Add some check on the length here
        if next_id == "" or length + speed * dt < road.L*road.b - road.L*road.dx: # and length < ...
            # Road_id is at the last road, don't need to find the next junction
            # or bus is not close to the junction
            pass
        else:
            # Find the junction and the activation function of the two roads
            # If the activation is above i.e. 0.5, the bus can cross the junction
            # If the activation is below, the bus cannot go further than the junction
            for j in self.junctions:
                check, activ_ = j.get_activation(t, road_id, next_id)
                if check:
                    activation = activ_
                    break

            combined_activation = torch.max(road_activation, activation)

            speed = speed * combined_activation

            if length + speed * dt >= road.L * road.b:
                for j in self.junctions:
                    if j.check_roads_contained(road_id, next_id):
                        jnc_speed = torch.maximum((road.L*road.b - length) / dt - 0.0001,
                                                  j.get_speed(t, road_id, next_id))
                        speed = torch.min(speed, jnc_speed)

        relative_length = road.L*road.b - length # Remaining length
        # bus.update_position(t.clone(), dt.clone(), speed, activation, relative_length, printing=False)
        stopping, delay = bus.update_position_restarting(dt, t, speed, activation, relative_length, printing=False)
        return slowdown_factors, slowing_idx, stopping, delay
    
    def update_position_of_bus_no_slowdown(self, bus, dt, t):
        if t < bus.start_time:
            # Bus has not started its route yet
            return #slowdown_factors
        
        # 1. Find the road the bus is on
        road_id, length, next_id = bus.get_road_id()
        if road_id == "":
            # Bus has reached the end of the route
            # Stop updating the bus
            return #slowdown_factors
        
        i, road = self.get_road(road_id)
        # i used to find out what slowdown_factors to modify
        # 2. Use position of bus to calculate the slowdown factors

        activation = torch.tensor(1.0)
        # 3. Find the the junction (and traffic light) that connects the two roads
        # This is actually not necessary unless the bus has almost reached the end of the road
        # Add some check on the length here
        if next_id == "" or length < road.L*road.b: # and length < ...
            # Road_id is at the last road, don't need to find the next junction
            # or bus is not close to the junction
            pass
        else:
            # Find the junction and the activation function of the two roads
            # If the activation is above i.e. 0.5, the bus can cross the junction
            # If the activation is below, the bus cannot go further than the junction
            for j in self.junctions:
                check, activ_ = j.get_activation(t, road_id, next_id)
                if check:
                    activation = activ_
                    break

        # 4. Using the road id and length, find the speed of the bus
        # Okay to use local speed here?
        if length >= road.L*road.b:
            if activation >= 0.5:
                for j in self.junctions:
                    if j.check_roads_contained(road_id, next_id):
                        speed = torch.maximum((road.L*road.b - length) / dt - 0.0001,
                                                  j.get_speed(t, road_id, next_id))
            else:
                # Setting speed equal to 0 means that the bus stops before the junction...
                # Should maybe set the speed as the minimum from the get_speed
                # and the speed that ensures the bus reaches the junction...
                speed = torch.tensor(0.0) # Differentiable...?
        else:
            new_length = length / road.L
            speed_, road_activation = road.get_speed(new_length, dt)

            speed = speed_ * road.L
        # At this point the position to the bus stop on this road can be used to update the
        # speed
        # If the bus stop is close, then the speed can be reduced

        relative_length = road.L*road.b - length # Remaining length
        bus.update_position(dt, t, speed, activation, relative_length, printing=False)
        return #slowdown_factors
    
    def solve_cons_law(self):
        '''
        Takes in a road network consisting of roads and junctions.
        Each road defines has its own numerical scheme limiter if second order and speed limit.
        

        Later add possibility of having different flux functions
        Can also add different flux functions for each road

        Solves model of road untill time T
        '''
        printing = False
        
        t = torch.tensor(0.0)
        if self.store_densities:
            rho_timesteps = {i : {0 : self.roads[i].rho.clone()} for i in range(len(self.roads))}
            # queue_timesteps = {i : {0 : self.roads[i].queue_length.clone()} for i in range(len(self.roads))}
            queue_timesteps = {i : {0 : self.roads[i].queue_length} for i in range(len(self.roads))}

        else:
            rho_timesteps = {i : {} for i in range(len(self.roads))}
            queue_timesteps = {i : {} for i in range(len(self.roads))}

        bus_timesteps = {i : {0 : self.busses[i].length_travelled} for i in range(len(self.busses))}

        i_count = 0
        while t < self.T:
            # Iterate untill time limit is reached
            controlpoint = self.T
            for road in self.roads:
                # Update index of controlpoint in use for each road
                # Get the first controlpoint reached and use as upper limit of 
                # how large time step dt can be
                new_time = road.update_index(t)
                if new_time == -1:
                    new_time = self.T

                ######################################################################
                # The time of jumps should also be added as control points, to ensure
                # That they are reached when simulating
                ######################################################################

                controlpoint = min(controlpoint, new_time)

            for j in self.junctions:
                controlpoint= min(controlpoint, j.get_next_control_point(t))
                
            if self.print_control_points:
                print("\n-----------------------------------------")
                print(f"Controlpoint: {controlpoint}")
                print("-----------------------------------------\n")
                print(f"road index {road.idx}")
                print(f"Speed limit on road {road.Vmax[road.idx]}")
                print(f"Gamma parameter {road.gamma[road.idx]}")
                
            while t < controlpoint:
                #-------------------------------------
                # STEP 1: Find appropriate timestep
                #-------------------------------------                
                dt = controlpoint - t

                for road in self.roads:
                    dt = torch.min(dt, road.max_dt())
                
            
                t = t + dt
                old_dt = dt.clone()

                #-------------------------------------
                # STEP 2: Calculate fluxes across junction
                #         Potentially update dt
                #-------------------------------------
                for J in self.junctions:
                    # Apply boundary conditions to all junctions
                    #J.apply_bc_wo_opt(dt, t)
                    min_dt = J.apply_bc(dt,t)
                    dt = torch.min(min_dt, dt)
                    # J.apply_bc(dt,t)

                #-------------------------------------
                # STEP 3: Calculate fluxes across roundabout junctions
                #         Potentially update dt
                #-------------------------------------
                for roundabout in self.roundabouts:
                    # Apply boundary conditions to all roundabouts
                    min_dt = roundabout.apply_bc(dt, t)
                    dt = torch.min(min_dt, dt)
                    # roundabout.apply_bc(dt, t)


                #-------------------------------------
                # STEP 4: Calculate fluxes to roads with one or more edges 
                #         not connected to junction
                #-------------------------------------
                for road in self.roads:
                    # Add boundary conditions to remaining roads
                    min_dt = road.apply_bc(dt, t)
                    dt = torch.min(min_dt, dt)

                if old_dt > dt:
                    t = t - old_dt + dt

                #-------------------------------------
                # STEP 5: Update positions of busses
                #-------------------------------------
                # Sowdown_factors is a list of how much to reduce the flux on each
                # cell interface for each road determined by the bus
                slowdown_factors = [torch.ones(road.N_full-1) for road in self.roads]
                slowdown_indexes = []
                for bus in self.busses:
                    # slowdown_factors, slowing_idx = self.update_position_of_bus(bus, dt.clone(), t.clone(), slowdown_factors)
                    slowdown_factors, slowing_idx = self.update_position_of_bus(bus, dt, t, slowdown_factors)
                    if slowing_idx is not None:
                        slowdown_indexes.append(slowing_idx)


                #-------------------------------------
                # STEP 6: Solve internal system for each road
                #-------------------------------------
                for i, road in enumerate(self.roads):
                    road.update_boundary_cells(dt, slowdown_factors[i])

                    # Solve internally on all roads in network
                    # Before updating internal values, values near boundary should maybe be saved to 
                    # update boundary properly
                    # Right now values near boundary at next time step is used to update boundary.
                    # Solution: Make road.solve_internally(dt) return a copy of the densities
                    # for each road instead of overwriting the values
                    # Update internal values later
                    if i in slowdown_indexes:
                        # Slowdown factors is the slowdown for all interfaces, but only
                        # send in slowdowns related to internal interfaces here
                        ######################################
                        ######################################
                        if road.pad == 1:
                            road.solve_internally_slowdown(dt, slowdown_factors[i])
                        else:
                            road.solve_internally_slowdown(dt, slowdown_factors[i][1:-1])
                    else:
                        road.solve_internally(dt)

                    #-------------------------------------
                    # STEP 7: Update boundaries on road
                    #-------------------------------------
                    # road.update_boundaries(dt)
                    road.update_boundaries()

                #-------------------------------------
                # STEP 8: Store solution after time t
                # Maybe a bit too much to store the solution at all times?
                #-------------------------------------
                if self.store_densities:
                    for i in range(len(self.roads)):
                        rho_timesteps[i][t] = self.roads[i].rho
                        # queue_timesteps[i][t] = self.roads[i].queue_length.clone()
                        queue_timesteps[i][t] = self.roads[i].queue_length

                for i in range(len(self.busses)):
                    bus_timesteps[i][t] = self.busses[i].length_travelled.clone()

                if self.debugging:
                    i_count += 1
                    if i_count >= self.iters:
                        t = self.T+1
                
                if printing:
                    print("-----------------------------------------\n")

        history_of_network = rho_timesteps
        queues = queue_timesteps
        bus_times = bus_timesteps
        bus_delays = {i : self.busses[i].delays for i in range(len(self.busses))}
        return history_of_network, queues, bus_times, bus_delays
    
    def solve_cons_law_counting(self):
        '''
        Takes in a road network consisting of roads and junctions.
        Each road defines has its own numerical scheme limiter if second order and speed limit.
        

        Later add possibility of having different flux functions
        Can also add different flux functions for each road

        Solves model of road untill time T
        '''
        printing = False
        n_stops_reached = 0
        
        t = torch.tensor(0.0)
        if self.store_densities:
            rho_timesteps = {i : {0 : self.roads[i].rho} for i in range(len(self.roads))}
            # queue_timesteps = {i : {0 : self.roads[i].queue_length.clone()} for i in range(len(self.roads))}
            queue_timesteps = {i : {0 : self.roads[i].queue_length} for i in range(len(self.roads))}

        else:
            rho_timesteps = {i : {} for i in range(len(self.roads))}
            queue_timesteps = {i : {} for i in range(len(self.roads))}

        bus_timesteps = {i : {0 : self.busses[i].length_travelled} for i in range(len(self.busses))}

        i_count = 0
        while t < self.T:
            # Iterate untill time limit is reached
            controlpoint = self.T
            for road in self.roads:
                # Update index of controlpoint in use for each road
                # Get the first controlpoint reached and use as upper limit of 
                # how large time step dt can be
                new_time = road.update_index(t)
                if new_time == -1:
                    new_time = self.T

                ######################################################################
                # The time of jumps should also be added as control points, to ensure
                # That they are reached when simulating
                ######################################################################

                controlpoint = min(controlpoint, new_time)

            for j in self.junctions:
                controlpoint= min(controlpoint, j.get_next_control_point(t))
                
            if self.print_control_points:
                print("\n-----------------------------------------")
                print(f"Controlpoint: {controlpoint}")
                print("-----------------------------------------\n")
                print(f"road index {road.idx}")
                print(f"Speed limit on road {road.Vmax[road.idx]}")
                print(f"Gamma parameter {road.gamma[road.idx]}")
                
            while t < controlpoint:
                #-------------------------------------
                # STEP 1: Find appropriate timestep
                #-------------------------------------                
                dt = controlpoint - t
                
                for road in self.roads:
                    dt = torch.min(dt, road.max_dt())
            
                t = t + dt
                old_dt = dt.clone()

                #-------------------------------------
                # STEP 2: Apply flux conditions for each Junction
                #-------------------------------------
                # if t < 1000:
                # print(t)
                # print()
                for J in self.junctions:
                    # Apply boundary conditions to all junctions
                    #J.apply_bc_wo_opt(dt, t)
                    # J.apply_bc(dt,t)
                    min_dt = J.apply_bc(dt,t)
                    dt = torch.min(min_dt, dt)


                #-------------------------------------
                # STEP 3: Apply flux conditions for each Roundabout
                #-------------------------------------
                # if t < 1000:
                for roundabout in self.roundabouts:
                    # Apply boundary conditions to all roundabouts
                    # roundabout.apply_bc(dt, t)
                    min_dt = roundabout.apply_bc(dt,t)
                    dt = torch.min(min_dt, dt)
                
                #-------------------------------------
                # STEP 4: Apply BC to roads with one or more edges not connected to junction
                #-------------------------------------
                for road in self.roads:
                    # Add boundary conditions to remaining roads
                    min_dt = road.apply_bc(dt, t)
                    dt = torch.min(min_dt, dt)

                if old_dt > dt:
                    t = t - old_dt + dt

                #-------------------------------------
                # STEP 5: Update positions of busses
                # This should be a separate function...
                #-------------------------------------
                # Sowdown_factors is a list of how much to reduce the flux on each
                # cell interface for each road determined by the bus
                slowdown_factors = [torch.ones(road.N_full-1) for road in self.roads]
                slowdown_indexes = []
                for bus in self.busses:
                    # slowdown_factors, slowing_idx = self.update_position_of_bus(bus, dt.clone(), t.clone(), slowdown_factors)
                    slowdown_factors, slowing_idx, stopping, _ = self.update_position_of_bus_restarting(bus, dt, t, slowdown_factors)
                    if stopping:
                        n_stops_reached += 1

                    if slowing_idx is not None:
                        slowdown_indexes.append(slowing_idx)
                
                #-------------------------------------
                # STEP 6: Solve internal system for each road
                #-------------------------------------
                for i, road in enumerate(self.roads):
                    road.update_boundary_cells(dt, slowdown_factors[i])
                    # Solve internally on all roads in network
                    # Before updating internal values, values near boundary should maybe be saved to 
                    # update boundary properly
                    # Right now values near boundary at next time step is used to update boundary.
                    # Solution: Make road.solve_internally(dt) return a copy of the densities
                    # for each road instead of overwriting the values
                    # Update internal values later
                    if i in slowdown_indexes:
                        if road.pad == 1:
                            road.solve_internally_slowdown(dt, slowdown_factors[i])
                        else:
                            road.solve_internally_slowdown(dt, slowdown_factors[i][1:-1])
                    else:
                        road.solve_internally(dt)

                    road.update_boundaries()

                # At this point, the old internal values are not used anymore, so they can safely be 
                # overwritten

                #-------------------------------------
                # STEP 6: Store solution after time t
                # Maybe a bit too much to store the solution at all times
                # If the objective function could be calculated here instead, ¨
                # it would probably save a lot of memory...
                #-------------------------------------
                if self.store_densities:
                    for i in range(len(self.roads)):
                        rho_timesteps[i][t] = self.roads[i].rho.clone()
                        # queue_timesteps[i][t] = self.roads[i].queue_length.clone()
                        queue_timesteps[i][t] = self.roads[i].queue_length

                for i in range(len(self.busses)):
                    bus_timesteps[i][t] = self.busses[i].length_travelled.clone()

                if self.debugging:
                    i_count += 1
                    if i_count >= self.iters:
                        t = self.T+1
                
                if printing:
                    print("-----------------------------------------\n")

        history_of_network = rho_timesteps
        queues = queue_timesteps
        bus_times = bus_timesteps
        bus_delays = {i : self.busses[i].delays for i in range(len(self.busses))}
        return history_of_network, queues, bus_times, bus_delays, n_stops_reached
    
    def solve_until_stop_reached(self, t = torch.tensor(0)):
        '''
        Takes in a road network consisting of roads and junctions.
        Each road defines has its own numerical scheme limiter if second order and speed limit.
        

        Later add possibility of having different flux functions
        Can also add different flux functions for each road

        Solves model of road untill time T

        To reduce the memory usage of the method, every time a bus stops is reached the gradient is 
        calculated before the computational graph is reset.
        The hope is that this is still a decent approximation of the actual gradient
        '''
        printing = False
        
        # t = torch.tensor(0)
        if self.store_densities:
            rho_timesteps = {i : {t : self.roads[i].rho.clone()} for i in range(len(self.roads))}
            # queue_timesteps = {i : {0 : self.roads[i].queue_length.clone()} for i in range(len(self.roads))}
            queue_timesteps = {i : {t : self.roads[i].queue_length.clone()} for i in range(len(self.roads))}

        else:
            rho_timesteps = {i : {} for i in range(len(self.roads))}
            queue_timesteps = {i : {} for i in range(len(self.roads))}

        bus_timesteps = {i : {t : self.busses[i].length_travelled.clone()} for i in range(len(self.busses))}

        while t < self.T:
            # Iterate untill time limit is reached
            controlpoint = self.T
            for road in self.roads:
                # Update index of controlpoint in use for each road
                # Get the first controlpoint reached and use as upper limit of 
                # how large time step dt can be
                new_time = road.update_index(t)
                if new_time == -1:
                    new_time = self.T

                ######################################################################
                # The time of jumps should also be added as control points, to ensure
                # That they are reached when simulating
                ######################################################################

                controlpoint = min(controlpoint, new_time)

            for j in self.junctions:
                controlpoint= min(controlpoint, j.get_next_control_point(t))
                
            if self.print_control_points:
                print("\n-----------------------------------------")
                print(f"Controlpoint: {controlpoint}")
                print("-----------------------------------------\n")
                print(f"road index {road.idx}")
                print(f"Speed limit on road {road.Vmax[road.idx]}")
                print(f"Gamma parameter {road.gamma[road.idx]}")
                
            while t < controlpoint:
                #-------------------------------------
                # STEP 1: Find appropriate timestep
                #-------------------------------------                
                dt = controlpoint - t
                
                for road in self.roads:
                    dt = torch.min(dt, road.max_dt())

                t = t + dt
                old_dt = dt.clone()

                #-------------------------------------
                # STEP 2: Apply flux conditions for each Junction
                #-------------------------------------
                # if t < 1000:
                for J in self.junctions:
                    # Apply boundary conditions to all junctions
                    #J.apply_bc_wo_opt(dt, t)
                    # J.apply_bc(dt,t)
                    min_dt = J.apply_bc(dt,t)
                    dt = torch.min(min_dt, dt)
                
                #-------------------------------------
                # STEP 3: Apply flux conditions for each Roundabout
                #-------------------------------------
                # if t < 1000:
                for roundabout in self.roundabouts:
                    # Apply boundary conditions to all roundabouts
                    # roundabout.apply_bc(dt, t)
                    min_dt = roundabout.apply_bc(dt, t)
                    dt = torch.min(min_dt, dt)

                #-------------------------------------
                # STEP 4: Apply BC to roads with one or more edges not connected to junction
                #-------------------------------------
                for road in self.roads:
                    # Add boundary conditions to remaining roads
                    min_dt = road.apply_bc(dt, t)
                    dt = torch.min(min_dt, dt)

                if old_dt > dt:
                    t = t - old_dt + dt

                #-------------------------------------
                # STEP 5: Update positions of busses
                # This should be a separate function...
                #-------------------------------------
                # Sowdown_factors is a list of how much to reduce the flux on each
                # cell interface for each road determined by the bus
                slowdown_factors = [torch.ones(road.N_full-1) for road in self.roads]
                slowdown_indexes = []
                stop_reached = False
                next_delay = []
                for bus in self.busses:
                    # slowdown_factors, slowing_idx = self.update_position_of_bus(bus, dt.clone(), t.clone(), slowdown_factors)
                    slowdown_factors, slowing_idx, stopping, delay = self.update_position_of_bus_restarting(bus, dt, t, slowdown_factors)
                    if stopping == True:
                        stop_reached = True
                        next_delay.append(delay)

                    if slowing_idx is not None:
                        slowdown_indexes.append(slowing_idx)
                
                #-------------------------------------
                # STEP 6: Solve internal system for each road
                #-------------------------------------
                for i, road in enumerate(self.roads):
                    road.update_boundary_cells(dt, slowdown_factors[i])
                    # Solve internally on all roads in network
                    # Before updating internal values, values near boundary should maybe be saved to 
                    # update boundary properly
                    # Right now values near boundary at next time step is used to update boundary.
                    # Solution: Make road.solve_internally(dt) return a copy of the densities
                    # for each road instead of overwriting the values
                    # Update internal values later
                    if i in slowdown_indexes:
                        if road.pad == 1:
                            road.solve_internally_slowdown(dt, slowdown_factors[i])
                        else:
                            road.solve_internally_slowdown(dt, slowdown_factors[i][1:-1])
                    else:
                        road.solve_internally(dt)

                    road.update_boundaries()

                # At this point, the old internal values are not used anymore, so they can safely be 
                # overwritten

                #-------------------------------------
                # STEP 7: Store solution after time t
                # Maybe a bit too much to store the solution at all times
                # If the objective function could be calculated here instead, ¨
                # it would probably save a lot of memory...
                #-------------------------------------
                if self.store_densities:
                    for i in range(len(self.roads)):
                        rho_timesteps[i][t] = self.roads[i].rho.detach()
                        # queue_timesteps[i][t] = self.roads[i].queue_length.clone()
                        queue_timesteps[i][t] = self.roads[i].queue_length.detach()

                for i in range(len(self.busses)):
                    bus_timesteps[i][t] = self.busses[i].length_travelled.detach()

                # If any stops have been reached during this iteration, the 
                # function should be exited and relevant member variables should be 
                # detached

                if stop_reached:
                    # Calculating the gradient:
                    tot_delay = torch.tensor(0.0)
                    for delay in next_delay:
                        # Calling .backward() adds the contribution to the 
                        # existing gradient
                        # delay.backward()
                        tot_delay += delay
                    print("Calculating the gradient of the delay...")
                    if tot_delay > 0:
                        tot_delay.backward()

                    # Return the current densities, queues and bus_times
                    bus_delays = {i : [d.detach().clone() for d in self.busses[i].delays] for i in range(len(self.busses))}

                    return rho_timesteps, queue_timesteps, bus_timesteps, bus_delays, t.detach().clone(), len(next_delay)

        bus_delays = {i : [d.detach().clone() for d in self.busses[i].delays] for i in range(len(self.busses))}

        return rho_timesteps, queue_timesteps, bus_timesteps, bus_delays, self.T, 0

    def get_speed_limit_grads(self):
        '''
        Return a list of the derivative wrt the different speed limits of the roads
        '''
        tensors = []
        ids_added = []
        for road in self.roads:
            # print(f"Looking at road {road.id}")
            if road.id[-2:] in ["fw", "bw"]:
                if road.id[:-3] not in ids_added:
                    ids_added.append(road.id[:-3])
                    for v in road.Vmax:
                        # print(f"Adding {road.id}")
                        # tensors.append(v.grad)
                        if v.grad is not None:
                            tensors.append(v.grad.item())
                        else:
                            tensors.append(0.0)


            else:
                if road.id[:-1] not in ids_added:
                    ids_added.append(road.id[:-1])
                    for v in road.Vmax:
                        # print(f"Adding {road.id}")
                        # tensors.append(v.grad)
                        if v.grad is not None:
                            tensors.append(v.grad.item())
                        else:
                            tensors.append(0.0)
        return tensors

    def get_traffic_light_grads(self):
        '''
        Return a list of the derivative wrt the different speed limits of the roads
        '''
        tensors = []
        for junction in self.junctions:
            for light in junction.trafficlights:
                for c in light.cycle:
                    # print(f"Adding regular light connecting {[road.id for road in junction.roads]}")
                    if c.grad is not None:
                            tensors.append(c.grad.item())
                    else:
                        tensors.append(0.0)

            for light in junction.coupled_trafficlights:
                for c in light.cycle:
                    # print(f"Adding coupled light connecting {[road.id for road in junction.roads]}")

                    if c.grad is not None:
                        tensors.append(c.grad.item())
                    else:
                        tensors.append(0.0)
        return tensors
    
    def draw_network(self):
        # Function to draw a graph of the network to check that it is properly set up

        # Create an empty graph
        G = nx.Graph()

        # Create all necessary nodes
        for i in range(len(self.junctions)):
            G.add_node(f"J {i+1}")

        for i in range(len(self.roads)):
            if not self.roads[i].left:
                # Need to add left edge
                G.add_node(f"Left {i+1}")

            if not self.roads[i].right:
                # Need to add right edge
                G.add_node(f"Right {i+1}")
            
        # Create all necessary edges
        for i in range(len(self.roads)):
            # Either from left end to junction, 
            # junction to right end
            # or junction to junction
            # left to right only if only one road

            if not self.roads[i].left:
                # Left side not connected to anything
                for j in range(len(self.junctions)):
                    if self.roads[i] in self.junctions[j].roads:
                        G.add_edge(f"Left {i+1}", f"J {j+1}")
                        #break
                
                if not self.roads[i].right:
                    # This should only be the case when network only consists of single road
                    G.add_edge(f"Left {i+1}", f"Right {i+1}")
                
            elif not self.roads[i].right:
                # Left side not connected to anything
                for j in range(len(self.junctions)):
                    if self.roads[i] in self.junctions[j].roads:
                        G.add_edge(f"Right {i+1}", f"J {j+1}")
                        #break
            
            else:
                # Road connected to two junctions
                for j in range(len(self.junctions)):
                    if self.roads[i] in self.junctions[j].roads:
                        # One side connected to junction j
                        for k in range(j+1, len(self.junctions)):
                            if self.roads[i] in self.junctions[k].roads:
                                # Other side connected to junction k
                                G.add_edge(f"J {j+1}", f"J {k+1}")
                                #break

        pos = nx.spring_layout(G)
        print(pos)
        nx.draw(G, pos, with_labels=True, node_color="skyblue", font_size=10)
        node_positions = nx.get_node_attributes(G, 'pos')
        print(node_positions)
        plt.margins(0.2)
        plt.show()

    def get_node_pos(self):
        # Function to get position of roads

        # Create an empty graph

        road_pos = {i : {} for i in range(len(self.roads))}

        G = nx.Graph()
        # Create all necessary nodes
        for i in range(len(self.junctions)):
            G.add_node(f"J {i+1}")

        for i in range(len(self.roads)):
            if not self.roads[i].left:
                # Need to add left edge
                G.add_node(f"Left {i+1}")
                road_pos[i][f"Left {i+1}"] = None # Add placeholder for position of left node

            if not self.roads[i].right:
                # Need to add right edge
                G.add_node(f"Right {i+1}")
                road_pos[i][f"Right {i+1}"] = None
            
        # Create all necessary edges
        for i in range(len(self.roads)):
            # Either from left end to junction, 
            # junction to right end
            # or junction to junction
            # left to right only if only one road

            if not self.roads[i].left:
                # Left side not connected to anything
                for j in range(len(self.junctions)):
                    if self.roads[i] in self.junctions[j].roads:
                        G.add_edge(f"Left {i+1}", f"J {j+1}")
                        #break
                        road_pos[i][f"J {j+1}"] = None
                
                if not self.roads[i].right:
                    # This should only be the case when network only consists of single road
                    G.add_edge(f"Left {i+1}", f"Right {i+1}")
                    road_pos[i][f"Right {i+1}"] = None
                
            elif not self.roads[i].right:
                # Left side not connected to anything
                for j in range(len(self.junctions)):
                    if self.roads[i] in self.junctions[j].roads:
                        G.add_edge(f"Right {i+1}", f"J {j+1}")
                        #break
                        road_pos[i][f"J {j+1}"] = None
            
            else:
                # Road connected to two junctions
                for j in range(len(self.junctions)):
                    if self.roads[i] in self.junctions[j].roads:
                        # One side connected to junction j
                        for k in range(j+1, len(self.junctions)):
                            if self.roads[i] in self.junctions[k].roads:
                                # Should find out which edge is connected to which junction
                                # Need to check if road is entering or leaving

                                # Other side connected to junction k
                                G.add_edge(f"J {j+1}", f"J {k+1}")
                                #break
                                road_pos[i][f"J {j+1}"] = None
                                road_pos[i][f"J {j+1}"] = None

        pos = nx.spring_layout(G)

        for i in range(len(self.roads)):
            for key in road_pos[i].keys():

                road_pos[i][key] = pos[key]

        return road_pos
    

if __name__ == "__main__":
    import json
    import generate_kvadraturen as gk
    network_file = "kvadraturen_networks/with_e18/network_1_1.json"
    config_file = "kvadraturen_networks/with_e18/config_1_1.json"

    f = open(network_file)
    data = json.load(f)
    f.close()
    T = data["T"]
    N = data["N"]
    speed_limits = data["speed_limits"] # Nested list
    control_points = data["control_points"] # Nested list
    cycle_times = data["cycle_times"] # Nested list

    f = open(config_file)
    data = json.load(f)
    f.close()
    config = data


    network = gk.generate_kvadraturen_from_config_e18(50, N, speed_limits, control_points,
                                                      cycle_times, config, track_grad=False)
    
    densities, _, _, _ =  network.solve_cons_law()
    print(len(densities[0]))
    times = list(densities[0].keys())
    print(times)
    min_dt = torch.tensor(1.0)
    for road in network.roads:
        # print(f"Road id: {road.id}")
        # print(road.max_dt())
        # print(road.dx)
        # print(road.gamma[0])
        # print(road.dx / road.gamma[0])
        min_dt = torch.min(min_dt, road.dx / road.gamma[0])
        # print()

    print(min_dt)