import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid
import pandas as pd
from landlab import Component



class DE_Diffuser(Component):
    _name = "DE_Diffuser"
    _unit_agnostic = True


    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "flux__capacitor": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "ADD--",
        },
        "sediment__flux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "link",
            "doc": "Sediment flux at link",
        },
        "delta_f__threshold": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "ADD",
        },
        "R": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "ADD",
        },
        "hillslope__diffusivity": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "ADD",
        },
        "topographic__gradient": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "link",
            "doc": "ADD",
        },
    }

    def __init__(self,
                 grid,
                 uplift_rate=0.0001,
                 diffusivity = 0.01,
                 ALPHA = 0.7,
                 delta_f_max = 0.1,   # delta_f_max > 0
                 w_lim = 0.25,       # 0 < w_lim <=1
                 lamda_min = 1.1,   # lambda_min > 1
                 Wcfl = 1,        # 0 < _Wcfl <= 1
                 epsilon = 10**-14,
                 t_clock = 0.0,     # simulation time
                 ):

        super(DE_Diffuser, self).__init__(grid)

        if "topographic__elevation" not in grid.at_node:
            grid.add_field("topographic__elevation",
                           np.ones_like(self._grid.nodes.flatten()) * 0.0, at="node", dtype=float)


        if "hillslope__diffusivity" not in grid.at_node:
            grid.add_field("hillslope__diffusivity",
                           np.ones_like(self._grid.nodes.flatten()) * diffusivity, at="node", dtype=float)

        self.initialize_output_fields()
        self._non_active_link = self._grid.status_at_link == 4
        self._CORE_NODES = self._grid.core_nodes
        self._flatten_nodes = self._grid.nodes.flatten()
        self._ALPHA = ALPHA
        self._delta_f_max = delta_f_max
        self._lamda_min = lamda_min
        self._t_clock = t_clock
        self._uplift_rate = uplift_rate
        self._Qlist = pd.DataFrame(columns=('node_id', 'dts', 'update_time'))
        self._Wcfl = Wcfl
        self._w_lim = w_lim
        self._epsilon =epsilon

        # Inititalization of cfl and Qlist
        self._calc_cfl_dt()
        self._init_Qlist()



    def _calc_cfl_dt(self,):

        delta_f_threhold = self._grid.at_node['delta_f__threshold']
        diffusivity= self._grid.at_node["hillslope__diffusivity"]

        CFL_prefactor = (
                self._ALPHA * self._grid.length_of_link[: np.size(self._grid.nodes.flatten())] ** 2.0
        )

        self._cfl_dt_at_nodes  = np.abs(
            np.divide(CFL_prefactor,
                      diffusivity,
                      )
        )*0.1


        delta_f_threhold[:] = self._delta_f_max
    def _init_Qlist(self):

        sorted_ids = np.argsort(self._cfl_dt_at_nodes)
        sorted_ids = sorted_ids[np.isin(sorted_ids, self._grid.core_nodes)]
        self._Qlist['node_id'] = self._flatten_nodes[sorted_ids]
        self._Qlist['t_p'] = np.zeros_like(self._cfl_dt_at_nodes[sorted_ids])  # dt at node
        self._Qlist['next_update_time'] = self._cfl_dt_at_nodes[sorted_ids]
        self._Qlist.set_index('node_id', inplace=True)
        self._Qlist.sort_values(by=['next_update_time'], inplace=True)


    def select_and_update_event(self):

        # Pointers
        Qlist = self._Qlist
        grid = self._grid
        uplift_rate =self._uplift_rate
        R = self._grid.at_node['R']
        topo = self._grid.at_node['topographic__elevation']
        flux_capacitor = self._grid.at_node['flux__capacitor']

        # Start
        Qlist.sort_values(by=['next_update_time'], inplace=True)
        self._node_id_at_work = Qlist.index[0]
        node_id_at_work = self._node_id_at_work

        node_next_update_time = Qlist.loc[node_id_at_work, 'next_update_time']
        tp = Qlist.loc[node_id_at_work, 't_p']

        # Update global time
        self._t_clock = node_next_update_time

        fluxes_in, fluxes_out = self._calc_flux_at_node(node_id_at_work = node_id_at_work)
        R[node_id_at_work] = ((np.sum(fluxes_in) - np.sum(fluxes_out)) / grid.dx) + uplift_rate

        # Execute event
        topo[node_id_at_work] += R[node_id_at_work] * (self._t_clock - tp)
        Qlist.loc[node_id_at_work, 't_p'] = self._t_clock

        # Zero out flux capacitor
        flux_capacitor[node_id_at_work] = 0.0


    def event_synchronization(self, parent_node = None, parents=None):

        if parent_node is None:
            parent_node = self._node_id_at_work

        node_id_at_work = parent_node
        uplift_rate = self._uplift_rate
        grid = self._grid
        Qlist = self._Qlist
        R = self._grid.at_node['R']
        topo = self._grid.at_node['topographic__elevation']
        neighbors = grid.adjacent_nodes_at_node[node_id_at_work]
        flux_capacitor = grid.at_node['flux__capacitor']
        delta_f_threhold = grid.at_node['delta_f__threshold']


        if parents == None:
            parents = []
        parents.append(node_id_at_work)


        for neighbor in neighbors:
            if grid.node_is_boundary(neighbor) == False and ~np.isin(neighbor, parents):  # i.e., not a boundary and not a parent

                t_s = Qlist.loc[neighbor, 't_p']

                # Calc out and in fluxes
                fluxes_in, fluxes_out = self._calc_flux_at_node(node_id_at_work = neighbor)

                R[neighbor] = ((np.sum(fluxes_in) - np.sum(fluxes_out)) / grid.dx) + uplift_rate
                delta_f = R[neighbor] * (self._t_clock - t_s)
                delta_f_cap = flux_capacitor[neighbor] + delta_f
                topo[neighbor] += delta_f
                t_s = self._t_clock
                Qlist.loc[neighbor, 't_p'] = t_s
                if np.abs(delta_f_cap) >= delta_f_threhold[neighbor]:
                    flux_capacitor[neighbor] = 0.0
                    self.run_one_event(parent = neighbor, parents=parents)
                    return
                else:

                    self._correct_flux_between_nodes(node_s=neighbor, node_p=node_id_at_work)

            if grid.node_is_boundary(neighbor) == True & neighbor>0:
                self._correct_flux_between_nodes(node_s = neighbor, node_p = node_id_at_work)
        return

    def event_scheduling(self, parent = None):

        if parent == None:
            parent = self._node_id_at_work

        node_id_at_work = parent
        uplift_rate = self._uplift_rate
        grid = self._grid
        Qlist = self._Qlist
        R = self._grid.at_node['R']
        delta_f_threhold = grid.at_node['delta_f__threshold']

        # Calc out and in fluxes
        fluxes_in, fluxes_out = self._calc_flux_at_node(node_id_at_work)

        R[node_id_at_work] = ((np.sum(fluxes_in) - np.sum(fluxes_out)) / grid.dx) + uplift_rate
        temp_delta_f_p_threshold = self._compute_local_target_increment(node_id_at_work)
        if temp_delta_f_p_threshold== np.inf:
            Qlist.loc[node_id_at_work, 'next_update_time'] = np.inf
            return
        delta_f_threhold[node_id_at_work] =temp_delta_f_p_threshold
        delta_tp = np.divide(temp_delta_f_p_threshold, np.abs(R[node_id_at_work]))
        Qlist.loc[node_id_at_work, 'next_update_time'] = self._t_clock + delta_tp

    def _compute_local_target_increment(self, node_id_at_work):

        grid = self._grid
        R = self._grid.at_node['R']
        _Wcfl = self._Wcfl
        epsilon = self._epsilon
        topo = self._grid.at_node['topographic__elevation']
        cfl_dt_at_nodes = self._cfl_dt_at_nodes
        w_lim = self._w_lim
        lamda_min = self._lamda_min
        delta_f_max = self._delta_f_max

        neighbors = grid.adjacent_nodes_at_node[node_id_at_work]
        neighbors = neighbors[np.isin(neighbors, grid.core_nodes)]
        delta_f_cfl = np.abs(R[node_id_at_work]) * _Wcfl * cfl_dt_at_nodes[node_id_at_work]
        if delta_f_cfl < epsilon:
            return np.inf
        else:
            delta_f_p_threshold = np.copy(delta_f_cfl)
            topo_neighbors_min = np.min(topo[neighbors])
            topo_neighbors_max = np.max(topo[neighbors])

            lambda_t = np.min((np.divide(topo_neighbors_min, delta_f_p_threshold), lamda_min))
            if lambda_t > 1:
                delta_f_p_threshold = np.max((delta_f_p_threshold, np.min(
                    (np.divide(topo_neighbors_min, lambda_t), w_lim * (topo_neighbors_max - topo_neighbors_min)))))

            delta_f_p_threshold = np.min((delta_f_max, delta_f_p_threshold))

        return delta_f_p_threshold

    def _calc_flux_at_node(self, node_id_at_work):
        # Calc out and in fluxes
        grid = self._grid
        _non_active_link = self._non_active_link
        topo_grad_at_link = self._grid.at_link['topographic__gradient']
        sediment_flux = self.grid.at_link['sediment__flux']
        diffusivity_at_link = grid.map_node_to_link_linear_upwind(grid.at_node['hillslope__diffusivity'],topo_grad_at_link)

        topo_grad_at_link[:] = grid.calc_grad_at_link('topographic__elevation', out=None)

        sediment_flux[grid.links_at_node[node_id_at_work]] = topo_grad_at_link[grid.links_at_node[node_id_at_work]] * (grid.link_dirs_at_node[node_id_at_work] *
                                                               diffusivity_at_link[grid.links_at_node[node_id_at_work]])

        sediment_flux[_non_active_link] = 0.0  # take care for close boundaries
        fluxes_out = sediment_flux[
            grid.links_at_node[node_id_at_work][np.greater(sediment_flux[grid.links_at_node[node_id_at_work]],
                                                           0)]]

        fluxes_in = np.abs(
            sediment_flux[
                grid.links_at_node[node_id_at_work][np.greater(-sediment_flux[grid.links_at_node[node_id_at_work]],
                                                               0)]])
        return fluxes_in, fluxes_out

    def run_one_event(self, parent, parents):

        self.event_synchronization(parent_node=parent, parents=parents)
        self.event_scheduling(parent=parent)



    def _correct_flux_between_nodes(self, node_s, node_p):

        grid = self._grid
        topo = self._grid.at_node['topographic__elevation']
        gtopo = self._grid.at_link['topographic__gradient']
        sediment_flux = self._grid.at_link['sediment__flux']
        uplift_rate = self._uplift_rate
        R = self.grid.at_node['R']


        try_tail = grid.node_at_link_tail[grid.links_at_node[node_s]] == node_p
        try_head = grid.node_at_link_head[grid.links_at_node[node_s]] == node_p

        if np.any(try_tail):
            link =  grid.links_at_node[node_s][try_tail][0]
        elif np.any(try_head):
            link =  grid.links_at_node[node_s][try_head][0]
        else:
            link = -1

        if node_s > 0:
            if np.any([grid._node_status[node_s]==4, grid._node_status[node_p]== 4]):
                sediment_flux[link] = 0.0
                R[node_s] = 0.0
                return

        gtopo[link] = (topo[grid.node_at_link_head[link]] - topo[grid.node_at_link_tail[link]]) / grid.dx
        diffusivity_at_link = grid.map_node_to_link_linear_upwind(grid.at_node['hillslope__diffusivity'],
                                                                  gtopo)
        sediment_flux[grid.links_at_node[node_s]] = gtopo[grid.links_at_node[node_s]] * grid.link_dirs_at_node[node_s] * diffusivity_at_link[grid.links_at_node[node_s]]
        sediment_flux[self._non_active_link] = 0.0  # take care for close boundaries
        fluxes_out = sediment_flux[
            grid.links_at_node[node_s][
                np.greater(sediment_flux[grid.links_at_node[node_s]],
                           0)]]

        fluxes_in = np.abs(
            sediment_flux[
                grid.links_at_node[node_s][
                    np.greater(-sediment_flux[grid.links_at_node[node_s]],
                               0)]])
        R[node_s] = ((np.sum(fluxes_in) - np.sum(fluxes_out)) / grid.dx) + uplift_rate



