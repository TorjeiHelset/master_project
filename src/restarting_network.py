import torch
import generate_kvadraturen as gk
import network as nw


class NetworkState:
    '''
    Small class for collecting the state of a network object
    '''
    def __init__(self, network):
        # Read from roads:
        self.road_state = []
        for road in network.roads:
            road_dict = {}
            road_dict["rho"] = road.rho.detach().clone()
            road_dict["queue_length"] = road.queue_length.detach().clone()
            road_dict["index"] = road.index
            self.road_state.append(road_dict)

        # Read from busses:
        self.bus_state = []
        for bus in network.busses:
            bus_dict = {}
            bus_dict["length_travelled"] = bus.length_travelled.detach().clone()
            bus_dict["remaining_stop_time"]  = bus.remaining_stop_time.detach().clone()
            bus_dict["stop_factor"] = torch.tensor(0.0)
            bus_dict["at_stop"] = bus.at_stop
            bus_dict["next_stop"] = bus.next_stop
            bus_dict["active"] = bus.active
            self.bus_state.append(bus_dict)

        # Read from roundabouts:
        self.roundabout_state = []
        for roundabout in network.roundabouts:
            junction_state = []
            for junction in roundabout.junctions:
                roundabout_dict = {}
                if junction.queue_junction:
                    roundabout_dict["queue_length"] = junction.secondary_in.queue_length.detach().clone()

                junction_state.append(roundabout_dict)
            self.roundabout_state.append(junction_state)

        # Read gradient:
        self.speed_limit_grads = network.get_speed_limit_grads()
        self.traffic_light_grads = network.get_traffic_light_grads()


    def copy_to_network(self, network):
        for road, road_state in zip(network.roads, self.road_state):
            road.rho = road_state["rho"]
            road.queue_length = road_state["queue_length"]
            road.index = road_state["index"]

        for bus, bus_state in zip(network.busses, self.bus_state):
            bus.length_travelled = bus_state["length_travelled"]
            bus.remaining_stop_time = bus_state["remaining_stop_time"]
            bus.stop_factor = bus_state["stop_factor"]
            bus.at_stop = bus_state["at_stop"]
            bus.next_stop = bus_state["next_stop"]
            bus.active = bus_state["active"]

        for roundabout, roundabout_state in zip(network.roundabouts, self.roundabout_state):
            for junction, junction_state in zip(roundabout.junctions, roundabout_state):
                if junction.queue_junction:
                    junction.secondary_in.queue_length = junction_state["queue_length"]
        return network
        


class RestartingRoadNetwork:
    '''
    Specific class for performing simulating a network for kvadraturen with e18, creating a new
    RoadNetwork class every time a new bus stop is reached
    '''

    def __init__(self, T, N, speed_limits, control_points, cycle_times, config):
        self.T = T
        self.N = N
        self.speed_limits = speed_limits
        self.control_points = control_points
        self.cycle_times = cycle_times
        self.config = config
        self.speed_grads = []
        self.light_grads = []

    def _combine_grads(self, speed_grads, light_grads):
        for i, s in enumerate(speed_grads):
            self.speed_grads[i] += s

        for i, l in enumerate(light_grads):
            self.light_grads[i] += l

    def _combine_quantities(self, densities, queues, bus_lengths, bus_delays, n_stops_reached,
                            densities_, queues_, bus_lengths_, bus_delays_, n_stops_reached_):
        
        road_keys = densities.keys()
        bus_keys = bus_lengths.keys()

        for road in road_keys:
            for t in densities_[road].keys():
                densities[road][t] = densities_[road][t]
                queues[road][t] = queues_[road][t]

        for bus in bus_keys:
            for t in bus_lengths_[bus].keys():
                bus_lengths[bus][t] = bus_lengths_[bus][t]
            
            for i in range(len(bus_delays[bus])):
                bus_delays[bus][i] += bus_delays_[bus][i]

        n_stops_reached += n_stops_reached_

        return  densities, queues, bus_lengths, bus_delays, n_stops_reached

    def _first_step(self):
        network = gk.generate_kvadraturen_from_config_e18(self.T, self.N, self.speed_limits, self.control_points, 
                                                          self.cycle_times, self.config, track_grad=True)

        densities, queues, bus_lengths, bus_delays, t, n_stops_reached = network.solve_until_stop_reached()
        # Store last of network as NetworkState object
        state = NetworkState(network)
        return densities, queues, bus_lengths, bus_delays, t, n_stops_reached, state
    
    def _cons_law_step(self, state, t):
        network = gk.generate_kvadraturen_from_config_e18(self.T, self.N, self.speed_limits, self.control_points, 
                                                          self.cycle_times, self.config, track_grad=True)
        # Update network using last state
        network = state.copy_to_network(network)
        densities, queues, bus_lengths, bus_delays, t, n_stops_reached = network.solve_until_stop_reached(t)
        # Store last of network as NetworkState object
        state = NetworkState(network)
        return densities, queues, bus_lengths, bus_delays, t, n_stops_reached, state

    def solve_cons_law(self):
        # Perform first step
        print("Simulating until first stop is reached:")
        densities, queues, bus_lengths, bus_delays, t, n_stops_reached, state = self._first_step()
        # Initialize gradients
        self.speed_grads = state.speed_limit_grads
        self.light_grads = state.traffic_light_grads

        while t < self.T:
            print()
            print("Simulating until next stop is reached:")
            densities_, queues_, bus_lengths_, bus_delays_, t, n_stops_reached_, state = self._cons_law_step(state, t)
            # Update gradients
            self._combine_grads(state.speed_limit_grads, state.traffic_light_grads)
            # Combine quantities
            densities, queues, bus_lengths, bus_delays, n_stops_reached = self._combine_quantities(densities, queues, bus_lengths, bus_delays, n_stops_reached,
                                                                                                   densities_, queues_, bus_lengths_, bus_delays_, n_stops_reached_)
        
        # print("Gradients after the full simulation:")
        # print(self.speed_grads)
        # print(self.light_grads)
        return densities, queues, bus_lengths, bus_delays, n_stops_reached, self.speed_grads, self.light_grads

if __name__ == "__main__":
    import json

    # f = open("kvadraturen_networks/with_e18/network_2.json")
    # data = json.load(f)
    # f.close()
    # T = data["T"]
    # # T = 60
    # # T = 359#7.2502
    # # 7.700253486633301
    # N = data["N"]
    # speed_limits = data["speed_limits"] # Nested list
    # control_points = data["control_points"] # Nested list
    # cycle_times = data["cycle_times"] # Nested list

    # f = open("kvadraturen_networks/with_e18/config_2_1.json")
    # config = json.load(f)
    # f.close()

    # restart_network = RestartingRoadNetwork(T, N, speed_limits, control_points, 
    #                                         cycle_times, config)
    
    # restart_network.solve_cons_law()


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
    config = json.load(f)
    f.close()

    restart_network = RestartingRoadNetwork(T, N, speed_limits, control_points, 
                                            cycle_times, config)
    
    _, _, _, bus_delays, n_stops_reached, speed_grads, light_grads = restart_network.solve_cons_law()
    objective = 0.0
    for i in range(len(bus_delays)):
        for delay in bus_delays[i]:
            objective += delay
    print(bus_delays)
    print(n_stops_reached)
    objective = objective / n_stops_reached
    gradient = speed_grads + light_grads

    print(f"Objective: {objective}")
    print(type(objective))
    print(f"Gradient: {gradient}")
    print(type(gradient))
