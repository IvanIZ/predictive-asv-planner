""" A path evaluator by using the trained prediction models """
import os, sys

import numba
import numpy as np

from submodules.src.primitives import Primitives
from submodules.src.ship import Ship
from submodules.src.swath import Swath
from networks.network_modules import UNet_Ice
from submodules.src.occupancy_grid.ice_model_utils import crop_window, stitch_window, encode_swath, compute_ship_footprint_planner, view_swath, update_costmap, boundary_cost, get_boundary_map
import torch
from matplotlib import pyplot as plt
from torch import nn

class PredictivePathEvaluator:
    def __init__(self, prim: Primitives, cmap_scale, concentration):
        self.prim = prim
        self.cmap_scale = cmap_scale

        self.vertical_shift = 0     # NOTE this shoud be the same to how the ice model is trained!!!
        self.win_h = 40
        self.win_w = 40

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.diff_criterion = nn.MSELoss()  # allow reduction here as we are not doing batch prediction
        if concentration == 0.4:
            self.occ_diff_scale = 800
        else:
            self.occ_diff_scale = 1000

        self.ice_model = UNet_Ice(input_includes_centroids=False, output_includes_centroids=False).to(self.device)
        model_path = os.path.join('./models', 'c' + str(int(concentration * 100)), 'ice_model.pth')
        print("Path evaluator model: ", model_path)
        self.ice_model.load_state_dict(torch.load(model_path))
            

    def eval_path(self, occ_map, node_path, ship_pose, swath_ins, horizontal_shifts, ship_vertices, path_len_keys, debug=False):
        """
        This function computes a predictive evaluation of a path. Given a path, current ship position, and 
        the current occ observation, returns the path cost based on predictive occ diff
        :param occ_map: given occupancy observation, assume to be most recent
        :param node_path: assume to be of shape (n_nodes, 3), 3 --> (x, y, theta), where (x, y) is in global costmap frame
        :param ship_pose: current ship position in (x, y, theta) where (x, y) is in global costmap frame
        """

        swath_costs = []
        for i in range(node_path.shape[0] - 1):
            node_src = node_path[i]
            node_target = node_path[i + 1]

            # if ship already possed this segment (e.g. above both start & end), cost 0
            if ship_pose[1] > node_src[1] and ship_pose[1] > node_target[1]:
                cost = 0
            
            # all other cases, compute cost
            else:
                swath_in = swath_ins[i]
                horizontal_shift = horizontal_shifts[i]

                # compute global footprint
                transformed_node = (node_src[0], node_src[1], 0)
                theta_0 = np.pi / 2
                global_footprint = compute_ship_footprint_planner(node=transformed_node, theta_0=theta_0, ship_vertices=ship_vertices, occ_map_height=occ_map.shape[0], occ_map_width=occ_map.shape[1], scale=self.cmap_scale)

                occ_map_in, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win = crop_window(occ_map, node_src, win_width=self.win_w, win_height=self.win_h, horizontal_shift=horizontal_shift, vertical_shift=self.vertical_shift)
                footprint_in, _, _, _, _, _, _, _, _ = crop_window(global_footprint, node_src, win_width=self.win_w, win_height=self.win_h, horizontal_shift=horizontal_shift, vertical_shift=self.vertical_shift)

                x = np.concatenate((np.array([occ_map_in]), np.array([footprint_in]), np.array([swath_in])))
                x = torch.Tensor(x)     # (3 x W x H)
                x = x.unsqueeze(dim=0)      # (1 x 3 x W x H)
                x = x.to(self.device)

                # obtain prediction
                y = self.ice_model(x)
                occ_map_hat = y                                         # (1 x 1 x W x H)
                occ_map_hat = occ_map_hat.squeeze().detach()        # (W x H)

                # compute occ diff
                occ_before = x[0, 0, :, :]      # (W, H)
                occ_diff = self.diff_criterion(occ_map_hat, occ_before)
                occ_map_hat = occ_map_hat.cpu().numpy()

                # compute swath cost
                swath_cost = occ_diff.item() * self.occ_diff_scale
                origin, e = path_len_keys[i]
                temp_path_length = self.prim.path_lengths[(origin, e)]
                cost = swath_cost + temp_path_length

                # update global occ map
                occ_map = stitch_window(occ_map, occ_map_hat, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win)


                if debug:
                    self.save_debug_plots(i, occ_map, global_footprint, occ_map_in, occ_map_hat, swath_in)
            
            swath_costs.append(cost)

        return swath_costs
    

    def save_debug_plots(self, i, occ_map, global_footprint, occ_map_in, occ_map_hat, swath_in):
        fig, ax = plt.subplots()
        save_dir = "testbedSim_c04_r13"

        ax.clear()
        occ_map_render = np.flip(occ_map, axis=0)
        ax.imshow(occ_map_render, cmap='gray')
        fig.savefig("./dataset/" + save_dir + "/lattice_debug/" + str(i) + "_occ_map.png", bbox_inches='tight', transparent=False, pad_inches=0)

        ax.clear()
        footprint_render = np.flip(global_footprint, axis=0)
        ax.imshow(footprint_render, cmap='gray')
        fig.savefig("./dataset/" + save_dir + "/lattice_debug/" + str(i) + "_footprint.png", bbox_inches='tight', transparent=False, pad_inches=0)

        ax.clear()
        occ_map_in_render = np.flip(occ_map_in, axis=0)
        ax.imshow(occ_map_in_render, cmap='gray')
        fig.savefig("./dataset/" + save_dir + "/lattice_debug/" + str(i) + "_occ_map_in.png", bbox_inches='tight', transparent=False, pad_inches=0)
        
        ax.clear()
        occ_map_hat_render = np.flip(occ_map_hat, axis=0)
        ax.imshow(occ_map_hat_render, cmap='gray')
        fig.savefig("./dataset/" + save_dir + "/lattice_debug/" + str(i) + "_occ_map_hat.png", bbox_inches='tight', transparent=False, pad_inches=0)

        ax.clear()
        swath_in_render = np.flip(swath_in, axis=0)
        ax.imshow(swath_in_render, cmap='gray')
        fig.savefig("./dataset/" + save_dir + "/lattice_debug/" + str(i) + "_swath_in.png", bbox_inches='tight', transparent=False, pad_inches=0)
        

