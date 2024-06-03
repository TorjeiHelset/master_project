import json
import numpy as np

def interpolate_time(refined_densities, orig_densities):
    refined_times = [float(t) for t in list(refined_densities['0'].keys())]
    orig_times = [float(t) for t in list(orig_densities['0'].keys())]

    # Create container for new densities
    inter_densities = {i : {} for i in refined_densities.keys()}

    prev_idx = 0
    for t in refined_times:
        add_incr = 0
        for t_prev, t_next in zip(orig_times[prev_idx:-1], orig_times[prev_idx+1:]):
            if t_prev <= t and t_next >= t:
                # T somewhere between oldt_t_prev and old_t_next
                prev_dist = t - t_prev
                next_dist = t_next - t
                interval_length = t_next - t_prev
                prev_weight = (interval_length - prev_dist) / interval_length
                next_weight = (interval_length - next_dist) / interval_length
                # print(f"w_prev: {prev_weight}, w_next: {next_weight}")
                # print(f"t_prev: {t_prev}, t_next: {t_next}, t: {t}")
                # print()
                for road_id in refined_densities.keys():
                    inter_densities[road_id][str(t)] = [orig_densities[road_id][str(t_prev)][i]*prev_weight + orig_densities[road_id][str(t_next)][i]*next_weight for i in range(len(orig_densities[road_id][str(t_prev)]))]

                break
            else:
                add_incr += 1

            # if add_incr > 3:
            #     break
    
    return inter_densities


def calculate_l1(fine_densities, orig_densities, fine_dx):
    ratio = len(fine_densities['0']['0.0']) // len(orig_densities['0']['0.0'])

    time_diffs = {t: 0 for t in fine_densities['0'].keys()}
    for road_id in fine_densities.keys():
        # Go through every road
        for t in fine_densities[road_id].keys():
            # Go through every time
            # Calculate the L1 integral at time t and add to time_diffs
            time_diffs[t] = 0
            diff = 0
            for i in range(len(orig_densities[road_id][t])):
                for j in range(ratio):
                    diff += fine_dx * abs(orig_densities[road_id][t][i]
                                          - fine_densities[road_id][t][i*ratio + j])
            time_diffs[t] = time_diffs[t] + diff

    # Calculate full integral using trapezoidal rule
    times = np.array(list(time_diffs.keys()))

    l1_int = 0
    for t1, t2 in zip(times[:-1], times[1:]):
        dt = float(t2) - float(t1)
        error1 = time_diffs[t1]
        error2 = time_diffs[t2]

        l1_int = l1_int + dt * (error1 + error2) / 2

        
    return l1_int

def compare_grids(compare_files, refined_file, dx):
    # Load in the finest file
    f = open(refined_file)
    data = json.load(f)
    f.close()
    densities = data[0]

    l1_errors = []
    for compare_file in compare_files:
        # Load the new file
        f = open(compare_file)
        data = json.load(f)
        f.close()
        compare_densities = data[0]

        # Linear interpolation in time:
        interpolated = interpolate_time(densities, compare_densities)

        # Calculate L1 integral
        l1_error = calculate_l1(densities, interpolated, dx)
        l1_errors.append(l1_error)

    return l1_errors



