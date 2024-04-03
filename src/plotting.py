import matplotlib.pyplot as plt
import numpy as np


def plot_results(densities, queues, road_network):
    '''
    Plotting results from simulation
    '''
    n_roads = len(road_network.roads)
    n_rows = int(np.ceil(n_roads/2))
    

    times = list(densities[0].keys())

    # Creating plots
    density_fig, dens_axes = plt.subplots(n_rows, 2, figsize=(10,4))
    # caxes = [density_fig.add_axes([0.48, 0.15, 0.02, 0.7]),
    #          density_fig.add_axes([0.92, 0.15, 0.02, 0.7])]
             
    # queue_fig, queu_axes = plt.subplots(n_rows, 2)
    for i, road in enumerate(road_network.roads):
        x = np.linspace(0,road.b, len(densities[i][0]))
        # Go through each road        
        row = int(np.floor(i/2)) # Row of plotting
        col = int(np.ceil(i%2)) # Col of plotting

        queue_lengths = []
        road_densities = []
        for t in times:
            road_densities.append(densities[i][t].detach())
            # queue_lengths.append(queues[i][t].detach())
        
        if n_rows == 1:
            im = dens_axes[col].pcolor(x, times, road_densities, cmap="gray")
            dens_axes[col].set_title(f'Density of road {i+1}.')
            dens_axes[col].set_ylabel('Time [s]')
        else:
            im = dens_axes[row,col].pcolor(x, times, road_densities, cmap="gray")
            dens_axes[row,col].set_title(f'Density of road {i}.')
            dens_axes[row,col].set_ylabel('Time [s]')

        # cbar = density_fig.colorbar(im, cax=caxes[i])
        # cbar.set_label('Density')

        # queu_axes[row,col].plot(times, queue_lengths)
        # queu_axes[row,col].set_title(f'Queue length of road {i}.')
        # queu_axes[row,col].set_ylabel('Time [s]')

    plt.tight_layout()
    return density_fig, dens_axes


# Add method for plotting end density for a single road
def plot_first_road(densities, road_network):
    t = list(densities[0].keys())[-1]

    road = road_network.roads[0]
    x = np.linspace(0, road.b, len(densities[0][t])) - 1 


    road_densities = densities[0][t].detach()

    fig = plt.figure()
    plt.plot(x, road_densities)
    
    return fig