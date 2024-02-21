import generate_kvadraturen as gk
import bus
import network as nw

def smoothness_delay(T, v_strand_speeds, h_w_speed, tollbod_speed, elvegata_speed,
                     dronning_speed, festning_speed, lundsbro_speed):
    # Parameters for the busses:
    ids_bw = ["lundsbro_bw", "elvegata_bw", "tollbod_6bw", "tollbod_5bw", "tollbod_4bw", 
        "tollbod_3bw", "tollbod_2bw", "tollbod_1bw", "v_strand_7bw", "v_strand_6bw",
        "v_strand_5bw", "v_strand_4bw", "v_strand_3bw", "v_strand_2bw", "v_strand_1bw"] 
    stops_bw = [("tollbod_6bw", 50), ("tollbod_3bw", 90), ("tollbod_1bw", 30), ("v_strand_3bw", 25)]
    times_bw = [30, 90, 140, 220]

    ids_fw = ["v_strand_1fw", "v_strand_2fw", "v_strand_3fw", "h_w_2", "h_w_3",
    "h_w_4", "festning_4fw", "festning_5fw", "festning_6fw", "festning_7fw",
    "tollbod_4fw", "tollbod_5fw", "tollbod_6fw", "elvegata_fw", "lundsbro_fw"]

    stops_fw = [("h_w_3", 30), ("festning_5fw", 40), ("tollbod_4fw", 25), 
                    ("tollbod_6fw", 60)]
    times_fw = [40, 100, 150, 210]

    n = max(len(v_strand_speeds), len(h_w_speed), len(tollbod_speed), len(elvegata_speed),
            len(dronning_speed), len(festning_speed), len(lundsbro_speed))
    if n == 1:
        print("Not possible to calculate smoothness delay with only one speed for each street...")
        return None
    objectives = [0 for _ in range(n)]
    
    for i in range(n):
        if len(v_strand_speeds) == n:
            network = gk.generate_kvadraturen_small_w_params(T, v_strand_speeds[i], h_w_speed[0], tollbod_speed[0], 
                                                            elvegata_speed[0], dronning_speed[0], festning_speed[0],
                                                            lundsbro_speed[0])
        elif len(h_w_speed) == n:
            network = gk.generate_kvadraturen_small_w_params(T, v_strand_speeds[0], h_w_speed[i], tollbod_speed[0], 
                                                            elvegata_speed[0], dronning_speed[0], festning_speed[0],
                                                            lundsbro_speed[0])
        elif len(tollbod_speed) == n:
            network = gk.generate_kvadraturen_small_w_params(T, v_strand_speeds[0], h_w_speed[0], tollbod_speed[i], 
                                                            elvegata_speed[0], dronning_speed[0], festning_speed[0],
                                                            lundsbro_speed[0])
        elif len(elvegata_speed) == n:
            network = gk.generate_kvadraturen_small_w_params(T, v_strand_speeds[0], h_w_speed[0], tollbod_speed[0], 
                                                            elvegata_speed[i], dronning_speed[0], festning_speed[0],
                                                            lundsbro_speed[0])
        elif len(dronning_speed) == n:
            network = gk.generate_kvadraturen_small_w_params(T, v_strand_speeds[0], h_w_speed[0], tollbod_speed[0], 
                                                            elvegata_speed[0], dronning_speed[i], festning_speed[0],
                                                            lundsbro_speed[0])
        elif len(festning_speed) == n:
            network = gk.generate_kvadraturen_small_w_params(T, v_strand_speeds[0], h_w_speed[0], tollbod_speed[0], 
                                                            elvegata_speed[0], dronning_speed[0], festning_speed[i],
                                                            lundsbro_speed[0])
        else:
            network = gk.generate_kvadraturen_small_w_params(T, v_strand_speeds[0], h_w_speed[0], tollbod_speed[0], 
                                                            elvegata_speed[0], dronning_speed[0], festning_speed[0],
                                                            lundsbro_speed[i])
        
        # Add busses - only do two busses for now
        bus_bw = bus.Bus(ids_bw, stops_bw, times_bw, network, id = "19 bw")
        bus_fw = bus.Bus(ids_fw, stops_fw, times_fw, network, id = "19 fw")
        roads = network.roads
        junctions = network.junctions
        T = network.T
        bus_network = nw.RoadNetwork(roads, junctions, T, [bus_fw, bus_bw], store_densities = False)
        _, _, _, bus_delays = bus_network.solve_cons_law()
        objective = 0
        
        for l in range(len(bus_delays)): # 2
            for j in range(len(bus_delays[l])):
                try:
                    objective += bus_delays[l][j].detach()
                except:
                    objective += bus_delays[l][j]
        print(f"After simulation {i+1}, the objective was: {objective}")

        objectives[i] = objective
    return objectives


if __name__ == "__main__":
    # Check memory allocation:
    # import torch
    # with torch.autograd.profiler.profile() as prof:
    
    option = 7
    match option:
        case 0: # Vestre Strandgate
            # Check smoothness wrt speed on vestre strandgate:
            # Let speed vary from 30 to 60 km/h, with 5 km/h intervals
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            T = 400
            objectives = smoothness_delay(T, variable_speeds, [30.0], [30.0], [30.0], [50.0], [50.0], [50.0])
            print(objectives)
        
        case 1: # Visualize results from case 0
            # Plotting results from case 0
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            objectives = [242.0585, 225.5487, 225.2126, 224.9543, 224.7533, 224.2100, 223.8235]
            import matplotlib.pyplot as plt
            plt.plot(variable_speeds, objectives)
            plt.xlabel("Speed on Vestre Strandgate [km/h]")
            plt.ylabel("Objective value")
            plt.title("Objective value as a function of speed on Vestre Strandgate")
            plt.show()

        case 2: # Henrik Wergelands gate
            # Check smoothness wrt speed on henrik wergelands gate
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            T = 400
            objectives = smoothness_delay(T, [50.0], variable_speeds, [30.0], [30.0], [50.0], [50.0], [50.0])
            print(objectives)

        case 3: # Visualize results from case 2
            # Plotting results from case 2
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            objectives = [224.7533, 224.7535, 224.7537, 224.7539, 224.7433, 224.5243, 224.5075]
            import matplotlib.pyplot as plt
            plt.plot(variable_speeds, objectives)
            plt.xlabel("Speed on Henrik Wergelands gate [km/h]")
            plt.ylabel("Objective value")
            plt.title("Objective value as a function of speed on Henrik Wergelands gate")
            plt.show()

        case 4: # Tollbodgata
            # Check smoothness wrt speed on tollbodgata
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            T = 400
            objectives = smoothness_delay(T, [50.0], [30.0], variable_speeds, [30.0], [50.0], [50.0], [50.0])
            print(objectives)
        
        case 5: # Visualize results from case 4
            # Plotting results from case 4
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            objectives = [224.7533, 70.7778, 67.0494, 64.1555, 61.8448, 59.8606, 58.1628]
            import matplotlib.pyplot as plt
            plt.plot(variable_speeds, objectives)
            plt.xlabel("Speed on Tollbodgata [km/h]")
            plt.ylabel("Objective value")
            plt.title("Objective value as a function of speed on Tollbodgata")
            plt.show()
        
        case 6: # Elvegata
            # Check smoothness wrt speed on elvegata
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            T = 400
            objectives = smoothness_delay(T, [50.0], [30.0], [30.0], variable_speeds, [50.0], [50.0], [50.0])
            print(objectives)

        case 7: # Visualize results from case 6
            # Plotting results from case 6
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            objectives = [224.7533, 223.8207, 223.1177, 223.1175, 223.1174, 222.9969, 222.6492]
            import matplotlib.pyplot as plt
            plt.plot(variable_speeds, objectives)
            plt.xlabel("Speed on Elvegata [km/h]")
            plt.ylabel("Objective value")
            plt.title("Objective value as a function of speed on Elvegata")
            plt.show()

        case 8: # Dronningens gate
            # Check smoothness wrt speed on dronningens gate
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            T = 400
            objectives = smoothness_delay(T, [50.0], [30.0], [30.0], [30.0], variable_speeds, [50.0], [50.0])
            print(objectives)
        
        case 9: # Visualize results from case 8
            # Plotting results from case 8
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            objectives = [224.7125, 224.7150, 224.7278, 224.7455, 224.7533, 224.3718, 224.2061]
            import matplotlib.pyplot as plt
            plt.plot(variable_speeds, objectives)
            plt.xlabel("Speed on Dronningens gate [km/h]")
            plt.ylabel("Objective value")
            plt.title("Objective value as a function of speed on Dronningens gate")
            plt.show()
        
        case 10: # Festningsgata
            # Check smoothness wrt speed on festningsgata
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            T = 400
            objectives = smoothness_delay(T, [50.0], [30.0], [30.0], [30.0], [50.0], variable_speeds, [50.0])
            print(objectives)
        
        case 11: # Visualize results from case 10
            # Plotting results from case 10
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            objectives = [224.8055, 224.6883, 224.6926, 224.7186, 224.7533, 224.4084, 224.1792]
            import matplotlib.pyplot as plt
            plt.plot(variable_speeds, objectives)
            plt.xlabel("Speed on Festningsgata [km/h]")
            plt.ylabel("Objective value")
            plt.title("Objective value as a function of speed on Festningsgata")
            plt.show()
        
        case 12: # Lundsbroa
            # Check smoothness wrt speed on lundsbroa
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            T = 400
            objectives = smoothness_delay(T, [50.0], [30.0], [30.0], [30.0], [50.0], [50.0], variable_speeds)
            print(objectives)

        case 13: # Visualize results from case 12
            # Plotting results from case 12
            variable_speeds = [float(i) for i in range(30, 65, 5)]
            objectives = [232.2312, 228.8918, 226.5134, 225.2944, 224.7533, 223.6433, 222.8387]
            import matplotlib.pyplot as plt
            plt.plot(variable_speeds, objectives)
            plt.xlabel("Speed on Lundsbroa [km/h]")
            plt.ylabel("Objective value")
            plt.title("Objective value as a function of speed on Lundsbroa")
            plt.show()

