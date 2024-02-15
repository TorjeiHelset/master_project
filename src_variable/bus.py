import torch


'''
So far roads/junctions have not had global positions, but only relative positions
The fact that a bus follows a specific route, means that this approach might not be
good enough.
Two options:

1. Introduce a global coordinate system, and a mapping between coordinates and roads/junctions
 - This will entail at least some rewriting of the existing code

2. Introduce a route object, that needs to be able to map a length of the route to a 
position in the road system.

- Each road/junction could have an id, and the route could then know at what lengths of the 
bus route corresponds to what id-s.
- This requires minimal rewriting of the existing code.
- The route could still have a global position, as long as it is able to map from id to 
global position. Might be more work than it is worth.


The simplest might be for the bus to only have a length as position, and then the 
route object can map from length to road/junction. The route object will also know at 
what position on the specified road the bus is at.


Potential list of id's for bus 19

["v_strand_1", "v_strand_2", "v_strand_3", "v_strand_4", "v_strand_5", "v_strand_6", "v_strand_7",
    "tollbod_1", "tollbod_2", "tollbod_3", "tollbod_4", "tollbod_5", "tollbod_6",
    "elvegata", "lundsbro"]

    Potential list of id's for bus M1
["h_w_1", "h_w_2", "h_w_3", "h_w_4", "festning_4", "festning_5", "festning_6", "festning_7",
    "tollbod_4", "tollbod_5", "tollbod_6", "elvegata", "lundsbro"]



All id's in system: (for roads with two directions, also the direction is needed in the id)

v_strand_1 - v_strand_8 (2-way)
tollbod_1 - tollbod_3 (1-way)
tollbod_4 - tollbod_6 (2-way)
elvegata  (2-way)
lundsbro (2-way)
dronning_1 - dronning_6 (2-way)
festning_1 - festning_9 (2-way)
h_w_1 - h_w_4 (1-way)


All junctions in system:
v_strand_1 - v_strand_2 : no traffic light - does there need to be a junction?
v_strand_2 - v_stand_3 : no traffic light - does there need to be a junction?
v_strand_3 - v_strand_4 - h_w_1 - h_w_2: traffic light
v_strand_4 - v_strand_5 : traffic light
v_strand_5 - v_strand_6 : traffic light
v_strand_6 - v_strand_7 : traffic light
v_strand_7 - v_strand_8 - tollbod_1 : traffic light
v_strand_8 - dronning_1 : traffic light


bus 19 stops:
when should the times be added into the simulation?
Assume that 15:27 means 15:27:00 ?


times kirkegården 15:23
h_w_3 - 30m - 15:27
festning_5 - 40m - 15:28
tollbod_4 -25m - 15:29
tollbod_6 - 60m - 15:30
lund torv A - 15:32


bus 19 other direction:
lundsbro
elvegata
tollbod6 - tollbod1
v_strand_7 - v_strand_1
stops:
tollbod_6 - 50m - 13.00
tollbod_3 - 90m - 13.01
tollbod_1 - 30m - 13.05
v_strand_3 - 25m - 13.06

stops = [(tollbod_6, 50), (tollbod_3, 90), (tollbod_1, 30), (v_strand_3, 25)]

# Map to stops being absolute length of the route
'''

class Bus:
    '''
    Small class for keeping track of the route with its stops and
    the times it should be at each stop

    The route has a certain length and the stops ar at certain lengths
    along the route.

    The route object also needs to be able to map from length to road/junction
    This is done by storing the indexes of the roads/junctions in the route, and at
    what lengths of the route they start

    Needs to contain a list of id's of the roads/junctions (in order) that the route consists
    of

    Need a list of the lengths at which the stops are located

    Can also take in a list of times the bus should be at each stop - sometimes the bus will
    stop for a longer period of time at a stop

    '''

    

    def __init__(self, ids, stops, times, network, start_time = 0.0):
        '''
        Needs to have a check to ensure that all id's in route are in the network and not empty strings
        '''
        self.ids = ids # List of id's of the roads
        # Check that the number of stops and stopping times are equal
        assert len(stops) == len(times)
        self.times = times # List of times the bus should be at each stop
        self.stops = stops # List of lengths of the route at which the stops are located
        # stops could alternatively be a list of tuples with the id of the road and the length on that road
        self.times = times # List of times the bus should be at each stop
        # self.network = network
        self.lengths = self.get_lengths(network)
        self.length_travelled = 0
        # When the bus is at a stop, it should stop for a certain amount of time (not always?)
        # It also has to wait for the time in the schedule
        self.at_stop = False
        self.remaining_stop_time = 0
        self.stop_lengths = self.get_stop_lengs(stops)
        self.next_stop = 0
        self.delays = [torch.tensor(0.0) for _ in range(len(stops))]
        self.start_time = start_time


    def get_stop_lengs(self, stops):
        '''
        Returns the lengths of the stops
        '''
        # stop_lengths is not a torch variable so inplace operation is fine

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
        Returns the lengths of the route
        '''
        lengths = [0 for _ in range(len(self.ids))]
        for i, id in enumerate(self.ids):
            road = network.get_road(id)
            try:
                lengths[i] = road.L * road.b
            except:
                print(f"Road with id {id} not found in network...")
        return lengths
    
    def get_remaining_length(self):
        '''
        Returns the remaining length of the road the bus is on
        '''
        return
    
    def get_road_id(self):
        '''
        Returns the id of the road at the current length of the route
        as well as the remaining length of the road and the id of the next road
        '''
        # tot_length is not a torch tensor, so inplace operations is fine
        tot_length = 0
        for i, length in enumerate(self.lengths):
            tot_length += length
            # Greater or equal to, because the bus might be at the end of the road
            if tot_length >= self.length_travelled: 
                if i < len(self.ids)-1:
                    return self.ids[i], length - (tot_length - self.length_travelled), self.ids[i+1]
                else:
                    return self.ids[i], length - (tot_length - self.length_travelled), ""
        return "", 0, "" # End of route reached, stop updating...
    
    def get_road_id_at_length(self, length_travelled):
        '''
        Returns the id of the road at the given length of the route
        '''
        # inplace operation is fine
        tot_length = 0
        for i, length in enumerate(self.lengths):
            tot_length += length
            if tot_length >= length_travelled:
                return self.ids[i], length - (tot_length - length_travelled)
        return "", 0 # End of route reached, stop updating...
    

    def get_road_position(self, network):
        '''
        Returns the index of the road/junction at the given length, and the length of the road
        If the road is at the position of a junction, assume that it is at the end of the 
        previous road
        '''
        road_id, length, next_id = self.get_road_id()
        next_idx = -1
        if next_id != "":
            for i, road in enumerate(self.ids):
                if road == next_id:
                    next_idx = i
                    break

        if road_id == "":
            return "", 0, "" # End of route reached, stop updating...
        
        for i, road in enumerate(network.roads):
            if road.id == road_id:
                relative_length = length / road.L # Mapping to x-coord in the road
                return i, relative_length, next_idx
        return -1, 0, -1 # Road not found, stop updating...

    def update_position(self, t, dt, speed, activation, length, printing = False):
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
        '''


        if self.at_stop:
            if printing:
                print(f"Bus is at stop!")

            if self.remaining_stop_time > dt:
                if printing:
                    print(f"Bus should wait for {self.remaining_stop_time} seconds, more than the next time step")
                self.remaining_stop_time = self.remaining_stop_time - dt
                # The bus does not move
            else:
                if printing:
                    print("The bus has waited long enough!")
                moving_dt = dt - self.remaining_stop_time
                self.remaining_stop_time = 0
                self.at_stop = False
                if activation >= 0.5:
                    self.length_travelled = self.length_travelled + speed * moving_dt
                else:
                    try:
                        self.length_travelled = self.length_travelled + torch.min(speed * moving_dt, length)
                    except:
                        self.length_travelled = self.length_travelled + min(speed * moving_dt, length)
                
        else:
            # Check if the bus should stop at the next stop
            length_of_next_stop = self.stop_lengths[self.next_stop]

            if activation >= 0.5:
                #print(f"t = {t}, bus is allowed to cross the junction")
                # Bus can pass through the junction

                if self.length_travelled + speed * dt >= length_of_next_stop:
                    if printing:
                        print("Bus should stop at the busstop in this time step")
                        try:
                            print(f"Length travelled verison: {self.length_travelled._version}")
                        except:
                            pass

                        try:
                            print(f"Version of timestep: {dt._version}")
                        except:
                            pass

                        try:
                            print(f"Version of speed: {speed._version}")
                        except:
                            pass

                        try:
                            print(f"Version of length: {length._version}")
                        except:
                            pass

                        try:
                            print(f"Version of length_of_next_stop: {length_of_next_stop._version}")
                        except:
                            pass

                        try:
                            print(f"Version of dt: {dt._version}")
                        except:
                            pass
                        
                    # print(f"t = {t}, bus should stop at the busstop")
                    actual_dt = (length_of_next_stop - self.length_travelled)/speed
                    # The bus should stop at the next stop
                    self.at_stop = True
                    self.remaining_stop_time = min(30, self.times[self.next_stop])  - (dt - actual_dt) # This might be requiring gradient
                    
                    # Calculate delay time
                    if self.next_stop < len(self.times):
                        # At least one stop left
                        self.delays[self.next_stop] = self.delays[self.next_stop] + torch.max(torch.tensor(0.0), t + actual_dt - self.times[self.next_stop])
                        # inplace is fine
                        self.next_stop += 1

                    self.length_travelled = self.length_travelled + speed * actual_dt # Could set equal to length_of_next_stop, but
                    # then it would not be possible to differentiate
                else:
                    if printing:
                        print(f"Bus should travel full distance of {speed*dt} meters")
                    # print(f"t = {t}, bus should travel full distance of {speed*dt} meters")
                    self.length_travelled = self.length_travelled + speed * dt
            else:
                # Bus should stop at the junction
                if length + self.length_travelled >= length_of_next_stop:
                    # Bus could be stopped at the next stop
                    if self.length_travelled + speed * dt >= length_of_next_stop:
                        actual_dt = (length_of_next_stop - self.length_travelled)/speed
                        # The bus should stop at the next stop
                        self.at_stop = True
                        self.remaining_stop_time = 30 - (dt - actual_dt)
                        # Calculate delay time
                        if self.next_stop < len(self.times):
                            self.delays[self.next_stop] = self.delays[self.next_stop] + torch.max(torch.tensor(0.0), t + actual_dt - self.times[self.next_stop])
                            self.next_stop += 1
                        self.length_travelled = self.length_travelled + speed * actual_dt
                    else:
                        self.length_travelled = self.length_travelled + speed * dt

                else:
                    if printing:
                        print(f"Busshould stop at the junction!")
                    # Bus could be stopped at the junction
                    try:
                        self.length_travelled = self.length_travelled + torch.min(speed * dt, length)
                    except:
                        self.length_travelled = self.length_travelled + min(speed * dt, length)
        


if __name__ == "__main__":
    import network as nw
    import generate_kvadraturen as gk
    option = 2

    match option:
        case 0:
            network = gk.generate_kvadraturen_small(10.0)

            ids = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
                "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
                "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]
            
            stops = []
            times = []
            bus = Bus(ids, stops, times, network)
            print("Bus created!")
            print(f"Lengths of roads in route: {bus.lengths}")
            print(f"Total length = {sum(bus.lengths)}")

            roads = network.roads
            junctions = network.junctions
            T = network.T

            bus_network = nw.RoadNetwork(roads, junctions, T, [bus])
            print("Network created!")

            densities, queues, bus_lengths = bus_network.solve_cons_law()

            print("Solved conservation law!")
            print(f"Times: {densities[0].keys()}")
            print(f"Bus lengths travelled: {bus_lengths[0]}")
        case 1:
            T = 400
            network = gk.generate_kvadraturen_small(T)

            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_6bw", "tollbod_5bw", "tollbod_4bw", 
                   "tollbod_3bw", "tollbod_2bw", "tollbod_1bw", "v_strand_7bw", "v_strand_6bw",
                     "v_strand_5bw", "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"]            
            
            stops_bw = [("tollbod_6bw", 50), ("tollbod_3bw", 90), ("tollbod_1bw", 30), ("v_strand_3bw", 25)]
            times_bw = [40, 130, 190, 250]
            bus_bw = Bus(ids_bw, stops_bw, times_bw, network)

            ids_fw = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
                "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
                "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]
            
            stops_fw = [("h_w_3", 30), ("festning_5fw", 40), ("tollbod_4fw", 25), 
                        ("tollbod_6fw", 60)]
            times_fw = [50, 110, 130, 230]
            bus_fw = Bus(ids_fw, stops_fw, times_fw, network)


            roads = network.roads
            junctions = network.junctions
            T = network.T

            bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw])

            _, _, bus_lengths, bus_delays = bus_network.solve_cons_law()

            print("Solved conservation law!")

            print("\nDelays:")
            print(bus_delays)
        case 2:
            T = 400
            network = gk.generate_kvadraturen_small(T)


            ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_6bw", "tollbod_5bw", "tollbod_4bw", 
                   "tollbod_3bw", "tollbod_2bw", "tollbod_1bw", "v_strand_7bw", "v_strand_6bw",
                     "v_strand_5bw", "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"]            
            

            stops_bw = [("tollbod_6bw", 50), ("tollbod_3bw", 90), ("tollbod_1bw", 30), ("v_strand_3bw", 25)]
            times_bw = [40, 130, 190, 250]
            bus_bw = Bus(ids_bw, stops_bw, times_bw, network)

            ids_fw = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
                "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
                "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]
            
            stops_fw = [("h_w_3", 30), ("festning_5fw", 40), ("tollbod_4fw", 25), 
                        ("tollbod_6fw", 60)]
            times_fw = [50, 110, 130, 230]
            bus_fw = Bus(ids_fw, stops_fw, times_fw, network)


            roads = network.roads
            junctions = network.junctions
            T = network.T

            bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw])

            _, _, bus_lengths, bus_delays = bus_network.solve_cons_law()

            print("Solved conservation law!")

            # print(bus_lengths)

            print("\nDelays:")
            print(bus_delays)
            
            params = [v for road in bus_network.roads for v in road.Vmax] +\
                     [t for junction in bus_network.junctions for traffic_light in junction.trafficlights for t in traffic_light.cycle] +\
                     [t for junction in bus_network.junctions for traffic_light in junction.coupled_trafficlights for t in traffic_light.cycle]
            objective = 0
            for i in range(len(bus_delays)):
                for delay in bus_delays[i]:
                    objective += delay
        

            # Check gradient:
            j = torch.zeros(len(params))
            for i in range(len(params)):
                derivative =  torch.autograd.grad(objective*1, params[i], create_graph=True, allow_unused=True)[0]
                if derivative:
                    j[i] = derivative
            print("Gradient:")
            print(j)