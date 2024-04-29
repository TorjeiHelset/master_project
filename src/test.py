def create_roundabouts_w_e18(v_strand_fw, v_strand_bw, festning_fw, festning_bw,
                             e18_west_out_fw, e18_west_out_bw, e18_east_out_fw, e18_east_out_bw,
                       speed1 = [torch.tensor(50.0)], speed2 = [torch.tensor(50.0)], 
                       controls1 = [], controls2 = [], N = 2,
                       initial_densities = None, inflow_configs = None, track_grad = True):

    # Shared parameters for both roundabouts:
    offset = 0.1
    tilt = 0.025
    L = 25 # Length of road

    # Initial distribution functions:
    if initial_densities is None:
        initial_fncs = [lambda x : torch.ones_like(x) * 0.2 for _ in range(7)]
    else:
        initial_fncs = [ibc.init_density(initial_densities[-2], 1), 
                        ibc.init_density(initial_densities[-1], 1)]
        
    # Inflow functions - with e18 there is only a need for one inflow function
    if inflow_configs is None:
        # Vestre strandgate:
        secondary_gamma = torch.tensor(50 / (25 * 3.6))
        max_inflow = fv.flux(torch.tensor(0.5), secondary_gamma)
        rho_1 = torch.tensor(0.15)
        inflow = lambda t : fv.flux(rho_1, secondary_gamma)
        inflows = [inflow]
        max_inflows = [max_inflow]

    else:
        # Use config to create the inflow conditions:
        inflows = []
        max_inflows = []
        speeds = inflow_configs["speeds"]
        configs = inflow_configs["inflows"]

        for s, c in zip(speeds, configs):
            max_dens = c["max_dens"]
            gamma = torch.tensor(s / (L*3.6))
            max_inflow = fv.flux(torch.tensor(0.5), gamma) * max_dens
            fluxes = [fv.flux(torch.tensor(rho), gamma) for rho in c["densities"]]
            inflow = ibc.boundary_conditions(3, max_dens, c["densities"], fluxes, c["jumps"],
                                             in_speed = s, L = L)
            inflows.append(inflow)
            max_inflows.append(max_inflow)
            

    ##############################################
    # Vestre Strandgate roundabout:
    ##############################################

    # Four arm roundabout with 4 junctions, where 2 roads are a part of simulation
    # Secondary roads can be described by a small object with a queue and an inflow condition
    # Creating the mainline
    if torch.is_tensor(speed1[0]):
        main_speed_limit = [v / 3.6 for v in speed1]
    else:
        main_speed_limit = [torch.tensor(v / 3.6) for v in speed1]
    for v in main_speed_limit:
        v.requires_grad = track_grad

    vs_main_1 = rd.Road(1, L, N, main_speed_limit, controls1, initial=initial_fncs[0], left_pos=(0.05,0), right_pos=(0.25, -0.25),
                        id="vs_mainline_1")
    vs_main_2 = rd.Road(1, L, N, main_speed_limit, controls1, initial=initial_fncs[0], left_pos=(0.25, -0.25), right_pos=(0.08, -0.55),
                        id="vs_mainline_2")
    vs_main_3 = rd.Road(1, L, N, main_speed_limit, controls1, initial=initial_fncs[0], left_pos=(0.05, -0.55), right_pos=(-0.15, -0.3),
                        id="vs_mainline_3")
    vs_main_4 = rd.Road(1, L, N, main_speed_limit, controls1, initial=initial_fncs[0], left_pos=(-0.13, -0.28), right_pos=(-0.02,0),
                        id="vs_mainline_4")

    # Creating the secondary incoming roads
    # These are not proper road objects, but rather a smal classs containing only
    # the queue length and the inflow function
    # Inflow function should be specified by some config object...

    # Need secondary speeds
    vs_secondary = rb.RoundaboutRoad(inflows[0], max_inflows[0])

    # Creating the roundabout junctions
    vs_jnc_1 = rb.RoundaboutJunction(vs_main_4, vs_main_1, 0.6, v_strand_bw, v_strand_fw, queue_junction = False)
    vs_jnc_2 = rb.RoundaboutJunction(vs_main_1, vs_main_2, 0.6, e18_west_out_bw[1], e18_west_out_fw[1], queue_junction = False)
    vs_jnc_3 = rb.RoundaboutJunction(vs_main_2, vs_main_3, 0.6, vs_secondary, None, queue_junction = True)
    vs_jnc_4 = rb.RoundaboutJunction(vs_main_3, vs_main_4, 0.6, e18_west_out_fw[0], e18_west_out_bw[0], queue_junction = False)
    vs_junctions = [vs_jnc_1, vs_jnc_2, vs_jnc_3, vs_jnc_4]
    vs_roundabout = rb.Roundabout([vs_main_1, vs_main_2, vs_main_3, vs_main_4],
                           [v_strand_fw, e18_west_out_bw[1], vs_secondary, e18_west_out_fw[0]],
                           [v_strand_bw, e18_west_out_fw[1], None, e18_west_out_bw[0]],vs_junctions)

    ##############################################
    # Festningsgate roundabout:
    ##############################################

    if torch.is_tensor(speed2[0]):
        secondary_speed_limit = [v / 3.6 for v in speed2]
    else:
        secondary_speed_limit = [torch.tensor(v / 3.6) for v in speed2]
        
    for v in secondary_speed_limit:
        v.requires_grad = track_grad

    # Similar to the vestre strandgate roundabout, but with only three incoming roads
    fn_main_1 = rd.Road(1, L, N, secondary_speed_limit, controls2, initial=initial_fncs[1], left_pos=(3+9*tilt+0.04, offset), right_pos=(3+9*tilt+0.15, offset-0.4),
                        id="fn_mainline_1")
    fn_main_2 = rd.Road(1, L, N, secondary_speed_limit, controls2, initial=initial_fncs[1], left_pos=(3+9*tilt+0.15, offset-0.5), right_pos=(3+9*tilt-0.15, offset-0.5),
                        id="fn_mainline_2")
    fn_main_3 = rd.Road(1, L, N, secondary_speed_limit, controls2, initial=initial_fncs[1], left_pos=(3+9*tilt-0.15, offset-0.4), right_pos=(3+9*tilt-0.04, offset),
                        id="fn_mainline_3")
    
    # Creating the roundabout junctions
    fn_jnc_1 = rb.RoundaboutJunction(fn_main_3, fn_main_1, 0.6, festning_bw, festning_fw, queue_junction = False)
    fn_jnc_2 = rb.RoundaboutJunction(fn_main_1, fn_main_2, 0.6, e18_east_out_bw[1], e18_east_out_fw[1], queue_junction = False)
    fn_jnc_3 = rb.RoundaboutJunction(fn_main_2, fn_main_3, 0.6, e18_east_out_fw[0], e18_east_out_bw[0], queue_junction = False)
    fn_junctions = [fn_jnc_1, fn_jnc_2, fn_jnc_3]
    fn_roundabout = rb.Roundabout([fn_main_1, fn_main_2, fn_main_3],
                                  [festning_fw, e18_east_out_bw[1], e18_east_out_fw[0]],
                                  [festning_bw, e18_east_out_fw[1], e18_east_out_bw[0]], fn_junctions)

    return [vs_main_1, vs_main_2, vs_main_3, vs_main_4,fn_main_1, fn_main_2, fn_main_3], [vs_roundabout, fn_roundabout]