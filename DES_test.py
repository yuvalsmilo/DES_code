#!/usr/bin/env python
# coding: utf-8

# In[63]:


# Load components
import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid
from DE_Diffuser import DE_Diffuser
import sys
from landlab.components import LinearDiffuser
from landlab import imshow_grid

sys.setrecursionlimit(4000)

# Domain parameters
nrows = 500
ncols = 500
dx = 10

grid = RasterModelGrid((int(nrows), int(ncols)), xy_spacing=int(dx))
grid.set_closed_boundaries_at_grid_edges(False, False, False, False)
topo = grid.add_zeros('topographic__elevation', at='node')
grid.add_zeros('linear_diffusivity',
               at="node")
D = grid.at_node['linear_diffusivity']
diffusivity = 0.001
D[:] = diffusivity
D[grid.nodes[:,1]]= 0.0001

uplift_rate=0.0001
de_diffuser = DE_Diffuser(grid, diffusivity=D,
                          uplift_rate=uplift_rate,
                          delta_f_max = 1,
                          lamda_min=1,
                          w_lim=1,
                          cfl_factor=0.1)
# main loop
Ttot = 100000


t_clock = de_diffuser._t_clock
while t_clock < Ttot:
    de_diffuser.select_and_update_event()
    de_diffuser.event_synchronization()
    de_diffuser.event_scheduling()
    t_clock = de_diffuser._t_clock
    #print(t_clock)



fig, ax = plt.subplots(2,1,
                       figsize=(8,15))
xvec = np.cumsum(np.ones_like(grid.at_node['topographic__elevation'][grid.nodes][:,1]) * grid.dx)
for n in range(2):
    ax[n].plot(xvec, grid.at_node['topographic__elevation'][grid.nodes][:,1], color = 'black', linewidth=3)
    ax[n].set_xlabel('Distance [m]')
    ax[n].set_ylabel('Elevation [m]')
    if n ==1:
        ax[n].set_aspect('equal', 'box')

plt.show()

fig, ax = plt.subplots(2,1,
                       figsize=(8,15))
xvec = np.cumsum(np.ones_like(grid.at_node['topographic__elevation'][grid.nodes][5,:]) * grid.dx)
for n in range(2):
    ax[n].plot(xvec, grid.at_node['topographic__elevation'][grid.nodes][5,:], color = 'black', linewidth=3)
    ax[n].set_xlabel('Distance [m]')
    ax[n].set_ylabel('Elevation [m]')
    if n ==1:
        ax[n].set_aspect('equal', 'box')

plt.show()


imshow_grid(grid,grid.at_node['topographic__elevation'])
plt.show()



## 'Time-driven' simulation using Landlab LinearDiffuser component

grid = RasterModelGrid((int(nrows), int(ncols)), xy_spacing=int(dx))
grid.set_closed_boundaries_at_grid_edges(False, False, False, False)
topo = grid.add_zeros('topographic__elevation', at='node')
grid.add_zeros('linear_diffusivity',
               at="node")
DD = grid.at_node['linear_diffusivity']
DD[:] = D*10

Ttot = 100000
tl_diff = LinearDiffuser(grid, linear_diffusivity='linear_diffusivity')
t_clock = 0.0
dt = 50
while t_clock < Ttot:
    tl_diff.run_one_step(dt = dt)
    topo[grid.core_nodes] += uplift_rate*dt
    t_clock+=dt
    print(t_clock)



fig, ax = plt.subplots(2,1,
                       figsize=(8,15))
xvec = np.cumsum(np.ones_like(grid.at_node['topographic__elevation'][grid.nodes][:,1]) * grid.dx)
for n in range(2):
    ax[n].plot(xvec, grid.at_node['topographic__elevation'][grid.nodes][:,1], color = 'black', linewidth=3)
    ax[n].set_xlabel('Distance [m]')
    ax[n].set_ylabel('Elevation [m]')
    if n ==1:
        ax[n].set_aspect('equal', 'box')

plt.show()

fig, ax = plt.subplots(2,1,
                       figsize=(8,15))
xvec = np.cumsum(np.ones_like(grid.at_node['topographic__elevation'][grid.nodes][5,:]) * grid.dx)
for n in range(2):
    ax[n].plot(xvec, grid.at_node['topographic__elevation'][grid.nodes][5,:], color = 'black', linewidth=3)
    ax[n].set_xlabel('Distance [m]')
    ax[n].set_ylabel('Elevation [m]')
    if n ==1:
        ax[n].set_aspect('equal', 'box')

plt.show()



imshow_grid(grid,grid.at_node['topographic__elevation'])
plt.show()








# grid = RasterModelGrid((int(nrows), int(ncols)), xy_spacing=int(dx))
# grid.set_closed_boundaries_at_grid_edges(True, True, True, False)
# topo = grid.add_zeros('topographic__elevation', at='node')
# hillslope__diffusivity = grid.add_ones('hillslope__diffusivity', at='node')
# hillslope__diffusivity *= 0.001
# start_low_d = 4
# end_low_d = 9
# hillslope__diffusivity[grid.nodes[start_low_d:end_low_d, 1]] *= 0.4
# de_diffuser = DE_Diffuser(grid,
#                           diffusivity =hillslope__diffusivity)
#
# Ttot = 10000000
#
# t_clock = de_diffuser._t_clock
# while t_clock < Ttot:
#     de_diffuser.select_and_update_event()
#     de_diffuser.event_synchronization()
#     de_diffuser.event_scheduling()
#     t_clock = de_diffuser._t_clock
#     print('time = ', np.round(t_clock,1), ' -->  ', np.round(t_clock/Ttot,2), ' % out of Ttot')
#
# fig, ax = plt.subplots(2,1,
#                        figsize=(8,15))
# xvec = np.cumsum(np.ones_like(grid.at_node['topographic__elevation'][grid.nodes][:-1,1]) * grid.dx)
# for n in range(2):
#     ax[n].plot(xvec, grid.at_node['topographic__elevation'][grid.nodes][:-1,1], color = 'black', linewidth=3)
#     ax[n].plot(xvec[start_low_d: end_low_d],grid.at_node['topographic__elevation'][grid.nodes][start_low_d:end_low_d,1], color = 'red', linewidth=3)
#     ax[n].set_xlabel('Distance [m]')
#     ax[n].set_ylabel('Elevation [m]')
#     if n ==1:
#         ax[n].set_aspect('equal', 'box')
#
# plt.show()


# OLD  VERSION - BEFORE CREATING THE COMPONENT
# flux_capacitor = grid.add_zeros('flux__capacitor', at='node')
# delta_f_threhold = grid.add_zeros('delta_f__threhold', at='node')
# sediment_flux = grid.add_zeros('sediment__flux', at='link')
# R = grid.add_zeros('R', at='node')
# Qlist = pd.DataFrame(columns=('node_id','dts','update_time'))
#
# # In[66]:
#
#
# # Global parameters
# uplift_rate = 0.001
# diffusivity = 0.001
# _ALPHA = 0.7
# _CORE_NODES =  grid.core_nodes
# _flatten_nodes = grid.nodes.flatten()
# delta_f_max = 1 # delta_f_max > 0
# w_lim = 0.5 # 0 < w_lim <=1
# lamda_min = 1.1 # lambda_min >1
# _Wcfl = 0.5 # 0 < _Wcfl <= 1
# epsilon = 0.001
# _non_active_link = grid.status_at_link==4
# # In[67]:
# t_clock = 0.0 # simulation elapsed time
#
#
# # In[68]:
# topo_grad_at_link = grid.calc_grad_at_link('topographic__elevation', out=None)
# flux_at_link = diffusivity * topo_grad_at_link
# outlinks_at_node = grid.link_at_node_is_downwind(topo_grad_at_link)
#
#
# # In[69]:
# topo_grad_at_link = grid.calc_grad_at_link('topographic__elevation', out=None)
# CFL_prefactor = (
#             _ALPHA  * grid.length_of_link[: grid.number_of_links] ** 2.0
#         )
# dt_at_links = np.abs(
#     np.divide(CFL_prefactor,
#               diffusivity,
#              )
#     )
# cfl_dt_at_nodes  = grid.map_min_of_node_links_to_node(dt_at_links)
# sorted_ids = np.argsort(cfl_dt_at_nodes)
# sorted_ids = sorted_ids[np.isin(sorted_ids, grid.core_nodes)]
# Qlist['node_id'] = _flatten_nodes[sorted_ids]
# Qlist['t_p'] = np.zeros_like(cfl_dt_at_nodes[sorted_ids]) # dt at node
# Qlist['next_update_time'] = t_clock + Qlist['t_p']
# Qlist.set_index('node_id',inplace=True)
# Qlist.sort_values(by=['next_update_time'],inplace=True)
#
# # In[77]:
#
#
# ## Event processing
# # Select the event with the smallest timestamp
# def select_and_update_event(grid, Qlist, t_clock):
#     node_id_at_work = Qlist.index[0]
#     node_next_update_time  = Qlist.loc[node_id_at_work ,'next_update_time']
#     tp = Qlist.loc[node_id_at_work ,'t_p']
#
#     # Update global time
#     t_clock = node_next_update_time
#
#     fluxes_in, fluxes_out = calc_flux_at_node(grid, node_id_at_work)
#
#     R[node_id_at_work] = ((np.sum(fluxes_in) - np.sum(fluxes_out)) / grid.dx) + uplift_rate
#
#     # Execute event
#     topo[node_id_at_work] += R[node_id_at_work] * (t_clock - tp)
#     Qlist.loc[node_id_at_work, 't_p'] = t_clock
#
#     # Zero out flux capacitor
#     flux_capacitor[node_id_at_work] = 0.0
#
#     return node_id_at_work, topo, flux_capacitor, node_next_update_time,t_clock,Qlist
#
# def event_synchronization(grid, node_id_at_work, Qlist, t_clock, parents = None):
#
#     uplift_rate = 0.01
#     neighbors = grid.active_adjacent_nodes_at_node[node_id_at_work]
#     flux_capacitor = grid.at_node['flux__capacitor']
#     delta_f_threhold = grid.at_node['delta_f__threhold']
#     node_next_update_time = Qlist.loc[node_id_at_work, 'next_update_time']
#     if parents == None:
#         parents  = []
#     parents.append(node_id_at_work)
#     for neighbor in neighbors:
#         if grid.node_is_boundary(neighbor) == False and ~np.isin(neighbor, parents): # i.e., not a boundary
#
#             t_s = Qlist.loc[neighbor, 't_p']
#
#             # Calc out and in fluxes
#             fluxes_in, fluxes_out = calc_flux_at_node(grid, neighbor)
#
#             R[neighbor] = ((np.sum(fluxes_in) - np.sum(fluxes_out)) / grid.dx) + uplift_rate
#             delta_f = R[neighbor] * (t_clock - t_s )
#             delta_f_cap = flux_capacitor[neighbor] + delta_f
#             topo[neighbor] += delta_f
#             t_s = t_clock
#             Qlist.loc[neighbor, 't_p'] = t_s
#             if np.abs(delta_f_cap) >= delta_f_threhold[neighbor]:
#                 flux_capacitor[neighbor] = 0.0
#                 event_synchronization(grid, neighbor,Qlist, t_clock,parents)
#                 return
#             else:
#                 # link = find_common_link(grid, node_id_at_work, neighbor)
#                 # topo_grad_at_link[link] = topo[grid.node_at_link_head[link]] - topo[grid.node_at_link_tail[link]]
#                 # sediment_flux[link] = topo_grad_at_link[link] * grid.link_dirs_at_node[link] * diffusivity
#                 #
#                 # fluxes_out = sediment_flux[
#                 #     grid.links_at_node[neighbor][np.greater(sediment_flux[grid.links_at_node[neighbor]],
#                 #                                             0)]]
#                 #
#                 # fluxes_in = np.abs(sediment_flux[grid.links_at_node[neighbor][
#                 #     np.greater(-sediment_flux[grid.links_at_node[neighbor]],
#                 #                0)]])
#                 fluxes_in, fluxes_out = calc_flux_at_node(grid, neighbor)
#                 R[neighbor] = ((np.sum(fluxes_in) - np.sum(fluxes_out)) / grid.dx) + uplift_rate
#
#         if grid.node_is_boundary(neighbor) == True:
#             if grid._node_status[neighbor] == 1:
#                 fluxes_in, fluxes_out = calc_flux_at_node(grid, neighbor)
#     return
#     # In[56]:
#
#
# def find_common_link(grid, node_a, node_b):
#     try_tail = grid.node_at_link_tail[grid.links_at_node[node_a]] == node_b
#     try_head = grid.node_at_link_head[grid.links_at_node[node_a]] == node_b
#
#     if np.any(try_tail):
#         return grid.links_at_node[node_a][try_tail][0]
#     elif np.any(try_head):
#         return grid.links_at_node[node_a][try_head][0]
#     else:
#         return -1
#
#
# ## Event scheduling
# # Calc out and in fluxes
# def event_scheduling(grid,node_id_at_work,t_clock):
#
#     # Calc out and in fluxes
#     fluxes_in, fluxes_out = calc_flux_at_node(grid, node_id_at_work)
#
#
#     R[node_id_at_work] = ((np.sum(fluxes_in) - np.sum(fluxes_out)) / grid.dx) + uplift_rate
#
#     delta_f_p_threshold = compute_local_target_increment(grid, node_id_at_work)
#
#     delta_tp = np.divide(delta_f_p_threshold, np.abs(R[node_id_at_work]))
#
#     #Qlist.loc[node_id_at_work, 't_p'] = delta_tp  # dt at node
#     Qlist.loc[node_id_at_work, 'next_update_time'] = t_clock + delta_tp
#     return Qlist
#
# def compute_local_target_increment(grid, node_id_at_work):
#     neighbors = grid.active_adjacent_nodes_at_node[node_id_at_work]
#     neighbors = neighbors[np.isin(neighbors, grid.core_nodes)]
#     delta_f_cfl = np.abs(R[node_id_at_work]) * _Wcfl * cfl_dt_at_nodes[node_id_at_work]
#     if delta_f_cfl < epsilon:
#         delta_f_p_threshold = delta_f_cfl
#     else:
#         delta_f_p_threshold  = delta_f_cfl
#         topo_neighbors_min = np.min(topo[neighbors])
#         topo_neighbors_max = np.max(topo[neighbors])
#
#         lambda_t = np.min((np.divide(topo_neighbors_min, delta_f_p_threshold), lamda_min))
#         if lambda_t > 1:
#             delta_f_p_threshold = np.max((delta_f_p_threshold,np.min((np.divide(topo_neighbors_min,lambda_t),w_lim*(topo_neighbors_max-topo_neighbors_min)))))
#
#         delta_f_p_threshold = np.min((delta_f_max,delta_f_p_threshold))
#
#     return delta_f_p_threshold
#
#
# def calc_flux_at_node(grid, node_id_at_work):
#     # Calc out and in fluxes
#
#     topo_grad_at_link = grid.calc_grad_at_link('topographic__elevation', out=None)
#     topo_grad_at_link[_non_active_link ] = 0.0 # take care for close boundaries
#
#     sediment_flux[grid.links_at_node[node_id_at_work]] = topo_grad_at_link[grid.links_at_node[node_id_at_work]] * \
#                                                          grid.link_dirs_at_node[node_id_at_work] * diffusivity
#
#     fluxes_out = sediment_flux[
#         grid.links_at_node[node_id_at_work][np.greater(sediment_flux[grid.links_at_node[node_id_at_work]],
#                                                        0)]]
#
#     fluxes_in = np.abs(
#         sediment_flux[
#             grid.links_at_node[node_id_at_work][np.greater(-sediment_flux[grid.links_at_node[node_id_at_work]],
#                                                            0)]])
#     return fluxes_in, fluxes_out
#
#
# ## main loope
# Ttot = 500000
# while t_clock < Ttot:
#     Qlist.sort_values(by=['next_update_time'], inplace=True)
#
#     node_id_at_work, topo, flux_capacitor, node_next_update_time,t_clock,Qlist = select_and_update_event(grid, Qlist, t_clock)
#
#     event_synchronization(grid, node_id_at_work, Qlist, t_clock)
#
#     event_scheduling(grid, node_id_at_work,t_clock)
#
#
# plt.plot(topo[grid.nodes][:,1]),plt.show()
#
