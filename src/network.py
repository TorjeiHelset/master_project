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
        slowdown_factors[i], bus_started = bus.get_slowdown_factor(slowdown_factors[i].clone(), road_id,
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

                # OLD:
                # for l in range(opposite_road.N_internal-1):
                #     avg_density = (opposite_road.rho[-opposite_road.pad+1-l] + opposite_road.rho[-opposite_road.pad-l]) / 2
                #     slowdown_factors[i][l] = torch.tensor(1.0) - (1-slowdown_factors[i][l])/4 + avg_density * (slowdown_factors[i][l]-1)/4
                internal = torch.flip(opposite_road.rho[opposite_road.pad-1:-opposite_road.pad+1],dims=[0])
                avg = (internal[1:] + internal[:-1]) / 2
                slowdown_factors[i] = torch.ones(road.N_internal+1) - (1. - slowdown_factors[i]) * (1. + avg) / 2

        else:
            # Current road has more lanes
            # use only density/flux on this road to update the slowdown
            # factors
            # OLD:
            # avg = (road.rho[road.pad:-road.pad+1] + road.rho[road.pad-1:-road.pad]) / 4
            # for l in range(len(avg)-1):
            #     slowdown_factors[i][l] = torch.tensor(1.0) - (1-slowdown_factors[i][l]) / (4**n_extra_lanes) + avg[l] * (slowdown_factors[i][l] - 1 + (1-slowdown_factors[i][l])/(2**n_extra_lanes))

            n_extra_lanes = road.max_dens - 1
            internal = torch.flip(road.rho[road.pad-1:-road.pad+1],dims=[0])
            avg = (internal[1:] + internal[:-1]) / 2
            slowdown_factors[i] = torch.ones(road.N_internal+1) - (1. - slowdown_factors[i]) * (1. + avg) / (2**n_extra_lanes)
        
        #------------------------------------------------------------
        # SLOWDOWN FACTORS FINISHED
        #------------------------------------------------------------

        new_length = length / road.L
        # speed = road.get_speed(new_length) * road.L
        speed = road.get_speed_updated(new_length, dt) * road.L


        activation = torch.tensor(1.0)
        # 3. Find the the junction (and traffic light) that connects the two roads
        # This is actually not necessary unless the bus has almost reached the end of the road
        # Add some check on the length here
        if next_id == "" or length + speed * dt < road.L*road.b: # and length < ...
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
            if activation >= 0.5:
                # The bus can enter the junction, but the flux on the outgoing road should also be considered
                speed = torch.minimum(speed, j.get_speed(t, road_id, next_id))
            else:
                # Setting speed equal to 0 means that the bus stops before the junction...
                # Should maybe set the speed as the minimum from the get_speed
                # and the speed that ensures the bus reaches the junction...
                # speed = torch.tensor(0.0) # Diff erentiable...?
                # put speed so that the bus almost reaches the junction
                speed = torch.minimum(speed, (road.L*road.b - length) / dt - 0.0001) 


        # At this point the position to the bus stop on this road can be used to update the
        # speed
        # If the bus stop is close, then the speed can be reduced

        relative_length = road.L*road.b - length # Remaining length
        # bus.update_position(t.clone(), dt.clone(), speed, activation, relative_length, printing=False)
        bus.update_position(dt, t, speed, activation, relative_length, printing=False)
        return slowdown_factors, slowing_idx
    
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
                speed = j.get_speed(t, road_id, next_id) 
            else:
                # Setting speed equal to 0 means that the bus stops before the junction...
                # Should maybe set the speed as the minimum from the get_speed
                # and the speed that ensures the bus reaches the junction...
                speed = torch.tensor(0.0) # Differentiable...?
        else:
            new_length = length / road.L
            # speed = road.get_speed(new_length) * road.L # Need to multiply with L to get actual speed in m/s
            speed = road.get_speed_updated(new_length, dt) * road.L
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
        
        t = torch.tensor(0)
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

                #-------------------------------------
                # STEP 2: Update positions of busses
                # This should be a separate function...
                #-------------------------------------
                # Sowdown_factors is a list of how much to reduce the flux on each
                # cell interface for each road determined by the bus
                slowdown_factors = [torch.ones(road.N_internal+1) for road in self.roads]
                slowdown_indexes = []
                for bus in self.busses:
                    # slowdown_factors, slowing_idx = self.update_position_of_bus(bus, dt.clone(), t.clone(), slowdown_factors)
                    slowdown_factors, slowing_idx = self.update_position_of_bus(bus, dt, t, slowdown_factors)
                    if slowing_idx is not None:
                        slowdown_indexes.append(slowing_idx)

                #-------------------------------------
                # STEP 3: Apply flux conditions for each Junction
                #-------------------------------------
                # if t < 1000:
                for J in self.junctions:
                    # Apply boundary conditions to all junctions
                    #J.apply_bc_wo_opt(dt, t)
                    J.apply_bc(dt,t)

                #-------------------------------------
                # STEP 4: Apply flux conditions for each Roundabout
                #-------------------------------------
                # if t < 1000:
                for roundabout in self.roundabouts:
                    # Apply boundary conditions to all roundabouts
                    roundabout.apply_bc(dt, t)

                #-------------------------------------
                # STEP 5: Apply BC to roads with one or more edges not connected to junction
                #-------------------------------------
                for road in self.roads:
                    # Add boundary conditions to remaining roads
                    road.apply_bc(dt, t)
                
                #-------------------------------------
                # STEP 6: Solve internal system for each road
                #-------------------------------------
                for i, road in enumerate(self.roads):
                    # Solve internally on all roads in network
                    # Before updating internal values, values near boundary should maybe be saved to 
                    # update boundary properly
                    # Right now values near boundary at next time step is used to update boundary.
                    # Solution: Make road.solve_internally(dt) return a copy of the densities
                    # for each road instead of overwriting the values
                    # Update internal values later
                    if i in slowdown_indexes:
                        road.solve_internally_slowdown(dt, slowdown_factors[i])
                    else:
                        road.solve_internally(dt)

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
    option = 1

    match option:
        case 0:
            import road as r
            import junction as j
            import traffic_lights as tf
            import plotting as plot
            import initial_and_bc as ib

            
            road1 = r.Road(1, 1000, 50, [80, 80, 50], [10, 40], initial=ib.init_density(0.8, 1), inflow=1.)

            road2 = r.Road(1, 1000, 50, [50, 50, 50], [10, 40], initial=ib.init_density(0.3, 1))
            # road3 = r.Road(1, 1000, 50, [50, 50, 50], [10, 40])
            # road4 = r.Road(1, 1000, 50, [50, 30, 40], [10, 40])
            print(road1.rho)

            trafficlight = tf.TrafficLightContinous(False, [0], [1], [25., 10.])
            #coupledlight = tf.CoupledTrafficLight(False, [0], [2], [1], [3], [1.2,1.4,1.1,0.4])
            junction = j.Junction([road1, road2], entering=[0], leaving=[1],
                                    distribution=[1.], trafficlights=[trafficlight], coupled_trafficlights=[])

            network = RoadNetwork([road1, road2], [junction], 120)

            # print(road1.gamma)
            # print(road2.gamma)
            # print(road3.gamma)
            # print(road4.gamma)


            densities, queues = network.solve_cons_law()
            # print(queues[0])

            objective = torch.tensor(0.0)

            # for t in densities[1].keys():
            #     objective += torch.sum(densities[1][t])

            for t in queues[0].keys():
                objective += torch.sum(queues[0][t])

            print(objective)
            print(junction.trafficlights[0].cycle)
            torch.autograd.set_detect_anomaly(True)
            second = torch.autograd.grad(objective,junction.trafficlights[0].cycle[1], create_graph=True, allow_unused=True)[0]
            third = torch.autograd.grad(objective, network.roads[0].Vmax[0], create_graph=True, allow_unused=True)[0]
            fourth = torch.autograd.grad(objective, network.roads[1].Vmax[0], create_graph=True, allow_unused=True)[0]
            first = torch.autograd.grad(objective,junction.trafficlights[0].cycle[0], create_graph=True, allow_unused=True)[0]

            # first = torch.autograd.grad(objective,junction.trafficlights[0].cycle[0], create_graph=True, allow_unused=True)[0]
            # second = torch.autograd.grad(objective,junction.trafficlights[0].cycle[1], create_graph=True, allow_unused=True)[0]
            print(first, second, third, fourth)



            


        
        case 1:
            import loading_json as load
            import plotting as plot


            loaded_roads, loaded_junctions, network = load.initialize_road_network("networks/1-1.json")

            # densities, queues = network.solve_cons_law()
            # # print(densities)
            # fig, axes = plot.plot_results(densities, queues, network)
            # plt.show()
            network.get_node_pos()

         
        case 2:
            import loading_json as load
            import plotting as plot


            # loaded_roads, loaded_junctions, network = load.initialize_road_network("networks/1-1trafficLight.json")

            loaded_roads, loaded_junctions, network = load.initialize_road_network("networks/2-2_coupledlights.json")

            densities, queues = network.solve_cons_law()
            # print(densities)
            fig, axes = plot.plot_results(densities, queues, network)
            plt.show()
