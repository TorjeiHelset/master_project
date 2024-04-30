import torch
import generate_kvadraturen as gk
import json


if __name__ == "__main__":
    f = open("kvadraturen_networks/with_e18/network_2.json")
    data = json.load(f)
    f.close()
    T = data["T"]
    T = 60
    # T = 359#7.2502
    # 7.700253486633301
    N = data["N"]
    speed_limits = data["speed_limits"] # Nested list
    control_points = data["control_points"] # Nested list
    cycle_times = data["cycle_times"] # Nested list

    f = open("kvadraturen_networks/with_e18/config_2_1.json")
    config = json.load(f)
    f.close()

    bus_network = gk.generate_kvadraturen_from_config_e18(T, N, speed_limits, control_points, cycle_times,
                                                  config, track_grad=True)
    
    bus_network.solve_cons_law_with_restarting()
