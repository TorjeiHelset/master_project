import torch


def calculate_slowdown_factor(d, alpha = 0.5, beta = 2):
    '''
    The two factors alpha and beta are used to control the rate of slowing down both for the bus around the stop
    and for the flow of traffic near the bus.
    Alpha controls the slowing down in front of a stop/in front of the bus
    Beta controls the slowing down after a stop/in front of the bus
    Increasing alpha or beta makes the period of slowing down shorter
    '''
    return torch.max(torch.tensor(0.0), torch.sigmoid(alpha * d + 5) - torch.sigmoid(beta * d - 5))

class Bus:
    '''
    Class for keeping track of the route with its stops and
    the times it should be at each stop, as well as handling the position of a bus
    The route has a certain length and the stops ar at certain lengths
    along the route.
    '''

    def __init__(self, ids, stops, times, network, start_time = 0.0, id = ""):
        self.ids = ids # List of id's of the roads
        # Check that the number of stops and stopping times are equal
        assert len(stops) == len(times)
        self.times = times # List of times the bus should be at each stop
        self.stops = stops # List of lengths of the route at which the stops are located
        # stops could alternatively be a list of tuples with the id of the road and the length on that road
        self.lengths = self.get_lengths(network)
        self.length_travelled = torch.tensor(0.0)
        self.at_stop = False # True if bus is at a stop
        self.remaining_stop_time = torch.tensor(0.0)
        self.stop_lengths = self.get_stop_lengs(stops)
        self.next_stop = 0
        # LIST OF TENSORS - this is okay here
        self.delays = [torch.tensor(0.0) for _ in range(len(stops))]
        self.start_time = start_time # Possible for bus route to enter at a later time
        self.active = False
        self.id = id
        self.stop_factor = torch.tensor(0.0)

    def get_stop_lengs(self, stops):
        '''
        Returns the lengths from the beginning of the route to the stops
        The output stop_lengths is a list of floats/integers
        '''
        stop_lengths = [0 for _ in range(len(stops)+1)]
        for i, stop in enumerate(stops):
            stop_length = 0
            for length, id in zip(self.lengths, self.ids):
                if stop[0] == id:
                    stop_lengths[i] = stop_length + stop[1]
                else:
                    stop_length += length
        stop_lengths[-1] = sum(self.lengths) + 100
        return stop_lengths
    
    def get_lengths(self, network):
        '''
        Returns the lengths of each of the roads of the route
        The route consists of a list of id's. Each of these id's are looked for in the network
        and then the length of said road is calculated.
        Lengths will be a list of floats/integers
        '''
        lengths = [0 for _ in range(len(self.ids))]
        for i, id in enumerate(self.ids):
            _, road = network.get_road(id)
            try:
                # The boundary cells should probably not be displayed
                lengths[i] = road.L * road.b # Not taking into account the boundary cells...
            except:
                # Raise error?
                print(f"Road with id {id} not found in network...")
        return lengths
    
    def get_remaining_length(self):
        '''
        Returns the remaining length of the road the bus is on
        '''
        raise NotImplementedError("Function not defined yet")
    
    def get_road_id(self):
        '''
        Returns the id of the road the bus is currently on as well as the length travelled on this road and 
        the id of the next road. If the current road is the last road of the simulation, the id of the next
        road is set to an empty string.

        Not very pretty code, but is functional
        '''
        tot_length = 0
        for i, length in enumerate(self.lengths):
            tot_length += length
            # Greater or equal to, because the bus might be at the end of the road
            if tot_length >= self.length_travelled: 
                if i < len(self.ids)-1:
                    # At least one more road left
                    return self.ids[i], length - (tot_length - self.length_travelled), self.ids[i+1]
                else:
                    return self.ids[i], length - (tot_length - self.length_travelled), ""
        return "", 0, "" # End of route reached, stop updating...
    
    def get_road_id_at_length(self, length_travelled):
        '''
        Returns the id of the road at the given length of the route, as well as the length
        travelled on this road

        Very similar to get_road_id, but instead of using the length of the bus, an external length is used
        Also does not return the id of the next road
        '''
        tot_length = 0
        for i, length in enumerate(self.lengths):
            tot_length += length
            if tot_length >= length_travelled:
                return self.ids[i], length - (tot_length - length_travelled)
        return "", 0 # End of route reached, stop updating...
    
    def get_road_position(self, network):
        '''
        Returns the index of the road at the given length, and the length travelled on
        this road
        If the road is at the position of a junction, assume that it is at the end of the 
        previous road, instead of at the beginning of the next road.

        Index -1 is used when road is not found. This should maybe be changed to prevent any unexpted bugs.
        Could use None instead.
        '''
        # Find the current road
        road_id, length, next_id = self.get_road_id()
        next_idx = -1
        if next_id != "":
            # Next road is also a part of the simulation
            # for i, road in enumerate(self.ids):
            for i, road in enumerate(network.roads):
                if road.id == next_id:
                    next_idx = i
                    break

        if road_id == "":
            # End of route reached
            return "", 0, ""
        
        for i, road in enumerate(network.roads):
            # Find the road with correct id and calculate the length travelled
            if road.id == road_id:
                relative_length = length / road.L # Mapping to x-coord in the road
                return i, relative_length, next_idx
            
        return -1, 0, -1 # Road not found

    def get_slowdown_factor(self, slowdown_factor, road_id, length, road):
        '''
        This function calculates the rate of slowing down of the bus. First, if the bus has not started its route yet, then 
        nothing is done. If the bus has started the route, then given the id of the current road
        (the id could also be calculated here, but take in as argument to save time), the method looks for any stops on this 
        road. If no stops are found, then again nothing is done.
        If there is at least one stop on the current road, then the rate of slowing down is calculated. For each stop on the 
        road, the distance to the stop is calculated. The factor specifying the speed of the bus is calculated using the 
        calculate_slowdown_factor method with the distance as argument. Then, for each interface between cells on the road, 
        the distance from the interface to the bus is calculated. The flow of traffic across each interface will be scaled using
        the product of slowing rate of the bus and the output from calculate_slowdown_factor with the distance from the 
        interface to the bus.
        If there are more stops on the road, the calculation is done for each stop, and for each interface the maximum of the 
        slowing rates on this interface.

        The input parameter length, is the length the bus has travelled on the current road
        '''
        if not self.active:
            # Bus not actually a part of the simulation yet
            return slowdown_factor, False, torch.tensor(0.0)
        
        # ADDING new tensor factors at each iteration - could maybe use slowdown_factor directly
        factors = torch.ones(road.N_full-1) # One less interface than the number of cells
        stop_factor = torch.tensor(0.0)
        for stop in self.stops:
            if stop[0] == road_id:
                # This stop is on the road - need to calculate the distance from the bus:
                stop_pos = stop[1]
                distance = length - stop_pos
                # Calculate the slow down factor based on this distance:
                # Multiply by 0.8 to always allow for some cars to cross and to avoid bus having zero speed
                stop_factor = calculate_slowdown_factor(distance) * 0.8
                self.stop_factor = torch.max(self.stop_factor, stop_factor)
                # Update factors on the interface:
                interface_positions = torch.arange(road.dx, road.b, road.dx)
                # Faster by removing for loop...
                interface_factors = calculate_slowdown_factor(length - interface_positions*road.L)
                factors = torch.minimum(factors, 1.0 - interface_factors*stop_factor)     
        
        return factors, True, stop_factor

    def update_position(self, dt, t, speed, activation, length, printing = False):
        '''
        Calculates the next position given the current position and the speed and the
        time step
        Need the time t to compare to the time the bus should be at a stop
        Assume the bus should always stop at bus stops
        If the bus reaches the bus stop before the scheduled time, it should wait

        The speed is calculated at the current node at the current time. This is not
        100% accurate, but try as an initial guess

        The activation is a number specifying how much of the flux reaching the next junction
        will pass through. This is used to simulate the traffic lights. If the activation is
        lower than 0.5, the bus should wait at the junction, and if it is higher, the bus should
        pass through.

        Should maybe take in the slowdown factor as well so that the bus actually slows down 
        before and after stops

        At this point, the member function specifying the slowdown factor 
        of the bus is calculated, and can be used when calculating the
        speed

        speed = (1-self.stop_factor) * speed
        '''
        # Modify the speed using the slowing factor
        speed = (1-self.stop_factor)*speed
        self.stop_factor = torch.tensor(0.0)
        
        if not self.active:
            # Bus has not started its route yet
            if t > self.start_time:
                # This should only be reached once for each bus
                # Bus should start its route
                dt = t - self.start_time
                self.active = True
            else:
                # Bus should not start yet
                return

        if self.at_stop:
            if self.remaining_stop_time > dt:
                # The bus is still waiting...
                self.remaining_stop_time = torch.max(self.remaining_stop_time - dt, torch.tensor(0.0))
            else:
                # The bus can start moving again
                moving_dt = dt - self.remaining_stop_time
                self.remaining_stop_time = torch.tensor(0.0)
                self.at_stop = False

                self.length_travelled = self.length_travelled + speed * moving_dt
                
        else:
            # Check if the bus should stop at the next stop
            # Assume that two stops are not too close
            length_of_next_stop = self.stop_lengths[self.next_stop]

            if self.length_travelled + speed * dt >= length_of_next_stop:
                # The bus should stop at a bus stop
                actual_dt = (length_of_next_stop - self.length_travelled)/speed
                self.at_stop = True
                self.remaining_stop_time = torch.maximum(torch.tensor(30.0), self.times[self.next_stop] - t)  - (dt - actual_dt)
                print(f"Bus {self.id} reached bus stop {self.next_stop} at time {t}, should wait for {self.remaining_stop_time} seconds")


                if self.next_stop < len(self.times):
                    # At least one stop left
                    self.delays[self.next_stop] = self.delays[self.next_stop] + torch.maximum(torch.tensor(0.0), t + actual_dt - self.times[self.next_stop])
                    # self.delays[self.next_stop] = self.delays[self.next_stop].clone() + torch.max(torch.tensor(0.0), t.clone() + actual_dt.clone() - self.times[self.next_stop])
                    # inplace is fine
                    self.next_stop += 1

                self.length_travelled = self.length_travelled + speed * actual_dt
            else:
                self.length_travelled = self.length_travelled + speed * dt
                



    def update_position_restarting(self, dt, t, speed, activation, length, printing = False):
        '''
        Calculates the next position given the current position and the speed and the
        time step
        Need the time t to compare to the time the bus should be at a stop
        Assume the bus should always stop at bus stops
        If the bus reaches the bus stop before the scheduled time, it should wait

        The speed is calculated at the current node at the current time. This is not
        100% accurate, but try as an initial guess

        The activation is a number specifying how much of the flux reaching the next junction
        will pass through. This is used to simulate the traffic lights. If the activation is
        lower than 0.5, the bus should wait at the junction, and if it is higher, the bus should
        pass through.

        Should maybe take in the slowdown factor as well so that the bus actually slows down 
        before and after stops

        At this point, the member function specifying the slowdown factor 
        of the bus is calculated, and can be used when calculating the
        speed

        speed = (1-self.stop_factor) * speed
        '''
        stopping = False
        delay = torch.tensor(0.0)
        # Update the speed using the slowing factor
        speed = (1-self.stop_factor)*speed
        self.stop_factor = torch.tensor(0.0)

        if not self.active:
            # Bus has not started its route yet
            if t > self.start_time:
                # This should only be reached once for each bus
                # Bus should start its route
                dt = t - self.start_time
                self.active = True
            else:
                # Bus should not start yet
                return False, None

        if self.at_stop:

            if self.remaining_stop_time > dt:
                # The bus is still waiting...
                self.remaining_stop_time = torch.max(self.remaining_stop_time - dt, torch.tensor(0.0))
            else:
                # The bus can start moving again
                moving_dt = dt - self.remaining_stop_time
                self.remaining_stop_time = torch.tensor(0.0)
                self.at_stop = False
                self.length_travelled = self.length_travelled + speed * moving_dt
                
                
        else:
            # Check if the bus should stop at the next stop
            # Assume that two stops are not too close
            length_of_next_stop = self.stop_lengths[self.next_stop]

            if self.length_travelled + speed * dt >= length_of_next_stop:
                
                actual_dt = (length_of_next_stop - self.length_travelled)/speed
                stopping = True
                self.at_stop = True
                # This should maybe be a torch tensor that requires tracking the gradient...
                self.remaining_stop_time = torch.maximum(torch.tensor(30.0), self.times[self.next_stop] - t)  - (dt - actual_dt)
                print(f"Bus {self.id} reached bus stop {self.next_stop} at time {t}, should wait for {self.remaining_stop_time} seconds")

                # Calculate delay time
                if self.next_stop < len(self.times):
                    # At least one stop left
                    self.delays[self.next_stop] = self.delays[self.next_stop] + torch.maximum(torch.tensor(0.0), t + actual_dt - self.times[self.next_stop])
                    # self.delays[self.next_stop] = self.delays[self.next_stop].clone() + torch.max(torch.tensor(0.0), t.clone() + actual_dt.clone() - self.times[self.next_stop])
                    # inplace is fine
                    delay = self.delays[self.next_stop]
                    self.next_stop += 1
                
                # self.length_travelled = self.length_travelled.clone() + speed * actual_dt # Could set equal to length_of_next_stop, but
                # then it would not be possible to differentiate
                self.length_travelled = self.length_travelled + speed * actual_dt
            else:
                self.length_travelled = self.length_travelled + speed * dt
            
        return stopping, delay
        
