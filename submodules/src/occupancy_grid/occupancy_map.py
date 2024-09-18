import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
from skimage.draw import draw

# from submodules.src.image_process.watershed import get_ice_edges_simulated, get_ice_binary_occgrid_simulated
from submodules.src.geometry.polygon import poly_centroid

class OccupancyGrid:

    def __init__(self, grid_width, grid_height, map_width, map_height, ship_body=None) -> None:
        """
        grid_width, grid_height, map_width, map_height are in meter units
        ship body info at starting position
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.map_width = map_width
        self.map_height = map_height
        self.occ_map_width = int(self.map_width / self.grid_width)         # number of grids in x-axis
        self.occ_map_height = int(self.map_height / self.grid_height)      # number of grids in y-axis

        print("Occupancy map resolution: ", grid_width, "; occupancy map dimension: ", (self.occ_map_width, self.occ_map_height))

        self.occ_map = np.zeros((self.occ_map_height, self.occ_map_width))
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        self.target_encoding = np.zeros((self.occ_map_height, self.occ_map_width))
        self.obstacle_centroids = np.zeros((self.occ_map_height, self.occ_map_width))
        self.swath = np.zeros((self.occ_map_height, self.occ_map_width))

        self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))
        self.con_ax.set_xlabel('')
        self.con_ax.set_xticks([])
        self.con_ax.set_ylabel('')
        self.con_ax.set_yticks([])

        self.local_window_width = 16
        self.local_window_height = 6

        if ship_body is None:
            self.cur_grid_x, self.cur_grid_y = 0, 0
        else:
            self.cur_grid_x, self.cur_grid_y = self.get_current_grid(ship_body=ship_body)

        self.con_target_grid_x = None
        self.con_target_grid_y = None
        self.con_grid_entry = None
        self.con_grid_exit = None
        self.t = 0

        self.con_labels = np.zeros((self.occ_map_height, 6))   # 10 step x (prev_obs, label, target_x, target_y, entry, exit)
        self.flow_field = np.zeros((self.occ_map_height, self.occ_map_height, self.occ_map_width, 2))  # flow field labels (t_dim, y_dim, x_dim, 2)

        self.occ_observations = []
        self.swath_observations = []
        self.footprint_observations = []
        self.target_observations = []
        self.centroids_observations = []
        self.cur_grids = []                 # keeps track of cur_grid_x, cur_grid_y in each timestep

        # data collection status
        self.con_collection = True


    def compute_occ_img(self, obstacles, ice_binary_w=235, ice_binary_h=774):
        meter_to_pixel_scale = ice_binary_h / self.map_height

        raw_ice_binary = np.zeros((ice_binary_h, ice_binary_w))

        for obstacle in obstacles:
            obstacle = np.asarray(obstacle) * meter_to_pixel_scale

            # get pixel coordinates on costmap that are contained inside obstacle/polygon
            rr, cc = draw.polygon(obstacle[:, 1], obstacle[:, 0], shape=raw_ice_binary.shape)

            # skip if 0 area
            if len(rr) == 0 or len(cc) == 0:
                continue

            raw_ice_binary[rr, cc] = 1.0
        
        # occ_val = np.sum(raw_ice_binary) / (raw_ice_binary.shape[0] * raw_ice_binary.shape[1])
        # print("occ concentration: ", occ_val)
        
        return raw_ice_binary
    

    def compute_con_gridmap(self, raw_ice_binary, save_fig_dir=None, visualize=False):
        """
        Compute concentration grid map
        """
        meter_to_pixel_scale_y = raw_ice_binary.shape[0] / self.map_height
        meter_to_pixel_scale_x = raw_ice_binary.shape[1] / self.map_width

        for i in range(self.occ_map_height):
            y_low = int(i * self.grid_height * meter_to_pixel_scale_y)
            y_high = int((i + 1) * self.grid_height * meter_to_pixel_scale_y)
            if y_high >= raw_ice_binary.shape[0]:
                y_high = raw_ice_binary.shape[0] - 1

            for j in range(self.occ_map_width):
                x_low = int(j * self.grid_width * meter_to_pixel_scale_x)
                x_high = int((j + 1) * self.grid_width * meter_to_pixel_scale_x)
                if x_high >= raw_ice_binary.shape[1]:
                    x_high = raw_ice_binary.shape[1] - 1
                cropped_region = raw_ice_binary[y_low:y_high, x_low:x_high]

                self.occ_map[i, j] = np.mean(cropped_region)

        # self.con_ax.clear()

        if (save_fig_dir is not None) and visualize:
            occ_map_render = np.copy(self.occ_map)
            occ_map_render = np.flip(occ_map_render, axis=0)
            self.con_ax.imshow(occ_map_render, cmap='gray')
            self.con_ax.axis('off')
            fp = os.path.join(save_fig_dir, str(self.t) + '_con.png')
            self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)


    def compute_swath(self, body, ship_vertices, padding=0.25):
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        # # apply padding as in a* search
        # ship_vertices = np.asarray(
        #     [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in ship_vertices]
        # )

        # ship vertices in meter
        heading = body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(body.position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)

        rr, cc = draw.polygon(r=r, c=c)
        self.swath[rr, cc] = 1.0


    def compute_ship_footprint(self, body, ship_vertices, padding=0.25):
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        # # apply padding as in a* search
        # ship_vertices = np.asarray(
        #     [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in ship_vertices]
        # )

        # ship vertices in meter
        heading = body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(body.position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)

        rr, cc = draw.polygon(r=r, c=c)
        self.footprint[rr, cc] = 1.0

    
    def compute_ship_footprint_planner(self, ship_state, ship_vertices, padding=0.25):
        """
        NOTE this function computes current ship footprint similarily to self.compute_ship_footprint()
        but is intended for generating observations for planners 
        :param ship_state: (x, y, theta) where x, y are in meter and theta in radian
        :param ship_vertices: original unscaled, unpadded ship vertices
        """
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        position = ship_state[:2]
        angle = ship_state[2]

        # # apply padding as in a* search
        # ship_vertices = np.asarray(
        #     [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in ship_vertices]
        # )

        # ship vertices in meter
        heading = angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)
        
        # it is possible that the ship state is outside of the grid map
        if len(r) == 0 or len(c) == 0:
            print("ship outside the costmap!!!")
            return
        
        rr, cc = draw.polygon(r=r, c=c)
        
        self.footprint[rr, cc] = 1.0


    def encode_target_grid(self, save_fig_dir):
        self.target_encoding = np.zeros((self.occ_map_height, self.occ_map_width))
        # self.target_encoding[self.con_target_grid_y, self.con_target_grid_x] = 1.0

        if self.cur_grid_y >= self.occ_map_height or self.cur_grid_x >= self.occ_map_width:
            return
        
        self.target_encoding[self.cur_grid_y, self.cur_grid_x] = 1.0


    def encode_centroids(self, obstacles):
        self.obstacle_centroids = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height
        
        for obs in obstacles:
            center_x, center_y = np.abs(poly_centroid(obs))
            x = int(center_x * meter_to_grid_scale_x)
            y = int(center_y * meter_to_grid_scale_y)

            # skip out of window obstacles
            if x < 0 or x > self.occ_map_width - 1:
                continue
            if y < 0 or y > self.occ_map_height - 1:
                continue
            
            self.obstacle_centroids[y, x] = 1.0

    
    def new_step(self, ship_body):
        """
        Determine if this is a new step. If so, the main loop will save obstacle map and invole step()
        """
        grid_x, grid_y = self.get_current_grid(ship_body=ship_body)

        # check local window terminal condition
        if grid_y + self.local_window_height > self.occ_map_height:
            self.con_collection = False

        if grid_y > self.cur_grid_y:
            self.cur_grid_x = grid_x
            self.cur_grid_y = grid_y
            return True
        else:
            return False


    def step(self, save_fig_dir, suffix, ship_body, ship_vertices, path, work, visualize, cur_obstacles):

        # generate observation for current step
        img_vis = self.get_observation(save_fig_dir=save_fig_dir, suffix=suffix, visualize=visualize)

        raw_ice_binary = self.compute_occ_img(obstacles=cur_obstacles, ice_binary_w=235, ice_binary_h=774)
        self.compute_con_gridmap(raw_ice_binary=raw_ice_binary, save_fig_dir=save_fig_dir, visualize=visualize)
        self.occ_observations.append(np.copy(self.occ_map))

        # if concentration collection is already finished, return
        # if not self.con_collection:
        #     return

        # compute swath
        self.compute_swath(body=ship_body, ship_vertices=ship_vertices)
        self.swath_observations.append(np.copy(self.swath))

        # compute ship footprint
        self.compute_ship_footprint(body=ship_body, ship_vertices=ship_vertices)
        self.footprint_observations.append(np.copy(self.footprint))

        # encode target grid
        # self.encode_target_grid(save_fig_dir=save_fig_dir)
        # self.target_observations.append(np.copy(self.target_encoding))

        # encode obstacle centroids
        self.encode_centroids(obstacles=cur_obstacles)
        self.centroids_observations.append(self.obstacle_centroids)

        # keep track of cur_grid_x, cur_grid_y info
        self.cur_grids.append([self.cur_grid_x, self.cur_grid_y])

        self.t += 1


    def _step_con(self, save_fig_dir, suffix, ship_body, path):

        # generate concentration labels for the previous step target
        if (self.con_target_grid_y is not None) and (self.con_target_grid_y - self.cur_grid_y == 1):
            con_label = self.occ_map[self.con_target_grid_y, self.con_target_grid_x]
            self.con_labels[self.t - 1, 1] = con_label

        # get next concentration target grid coordinates (spatio query)
        target_grid_lower = (self.cur_grid_y + 2) * self.grid_height
        target_grid_upper = (self.cur_grid_y + 3) * self.grid_height
        within_grid_coords = path[(path[:, 1] > target_grid_lower) & (path[:, 1] < target_grid_upper)]   # (N x 3); 3 --> (x, y, z)
        self.con_target_grid_x = int(np.mean(within_grid_coords[:, 0]) // self.grid_width)
        self.con_target_grid_y = int(self.cur_grid_y + 2)
        self.con_labels[self.t, 2] = self.con_target_grid_x
        self.con_labels[self.t, 3] = self.con_target_grid_y

        # current con observation for current target
        cur_con_obs = self.occ_map[self.con_target_grid_y, self.con_target_grid_x]
        self.con_labels[self.t, 0] = cur_con_obs

        # get action query for concentration (entry and exit on the grid before the target grid, i.e. cur grid + 1)
        entry_grid_lower = (self.cur_grid_y + 1) * self.grid_height
        entry_grid_upper = (self.cur_grid_y + 2) * self.grid_height
        entry_grid_coords = path[(path[:, 1] > entry_grid_lower) & (path[:, 1] < entry_grid_upper)]   # (N x 3); 3 --> (x, y, z)
        entry_grid_x = int(np.mean(entry_grid_coords[:, 0]) // self.grid_width)
        self.con_grid_entry = entry_grid_coords[0, 0] - entry_grid_x * self.grid_width
        self.con_grid_exit = entry_grid_coords[-1, 0] - entry_grid_x * self.grid_width
        if self.con_grid_entry < 0:
            self.con_grid_entry = 0
        elif self.con_grid_entry > self.grid_width:
            self.con_grid_entry = self.grid_width
        if self.con_grid_exit < 0:
            self.con_grid_exit = 0
        elif self.con_grid_exit > self.grid_width:
            self.con_grid_exit = self.grid_width
        self.con_labels[self.t, 4] = self.con_grid_entry
        self.con_labels[self.t, 5] = self.con_grid_exit


    def get_observation(self, save_fig_dir, suffix, visualize):
        # fp = os.path.join(save_fig_dir, str(suffix) + '_obs.png')
        # img = cv2.imread(fp)
        if visualize:
            fp = os.path.join(save_fig_dir, str(suffix) + '.png')
            img_vis = cv2.imread(fp)
        else:
            # img_vis = None
            return None

        # crop image NOTE this is an ad-hoc procedure. Cropping extend is manually adjusted
        if self.map_height == 20 and self.map_width == 8:
            # dimension for 20x8 map
            desired_height = 757
            desired_width = 305
            # img = img[12:12+desired_height, 16:16+desired_width]
            if visualize:
                img_vis = img_vis[12:12+desired_height, 16:16+desired_width]

        elif self.map_height == 40 and self.map_width == 8:
            # dimension for 40x8 map
            desired_height = 757 - 8
            desired_width = 155 - 6
            # img = img[12:12+desired_height, 16:16+desired_width]
            if visualize:
                img_vis = img_vis[12:12+desired_height, 16:16+desired_width]

        elif self.map_height == 40 and self.map_width == 12:
            # dimension for 40x4 map
            # print("original size: ", img.shape)         # original size (774, 235)
            desired_height = 774 - 4
            desired_width = 235 - 4
            # img = img[9:9+desired_height, 15:15+desired_width]
            if visualize:
                img_vis = img_vis[9:9+desired_height, 15:15+desired_width]

        else:
            # dimension for 40x4 map
            # print("original size: ", img.shape)         # original size (774, 81)
            desired_height = 774 - 4
            desired_width = 81 - 4
            # img = img[9:9+desired_height, 15:15+desired_width]
            if visualize:
                img_vis = img_vis[9:9+desired_height, 15:15+desired_width]

        # segmented_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # raw_ice_binary = get_ice_binary_occgrid_simulated(segmented_image)
        # raw_ice_edges = get_ice_edges_simulated(segmented_image)
        # return raw_ice_binary, raw_ice_edges, img, img_vis
        return img_vis
            

    def export(self, save_fig_dir):
        occ_observations = np.array(self.occ_observations)
        swath_observations = np.array(self.swath_observations)
        footprint_observations = np.array(self.footprint_observations)
        # target_observations = np.array(self.target_observations)
        centroids_observations = np.array(self.centroids_observations)
        cur_grids = np.array(self.cur_grids)

        print("Occupancy bservation shape: ", occ_observations.shape)
        print("Swath observation shape: ", swath_observations.shape)
        print("Footprint observation shape: ", footprint_observations.shape)
        # print("Target observation shape: ", target_observations.shape)
        print("Centroids observation shape: ", centroids_observations.shape)
        print("Cur Grids shape: ", cur_grids.shape)

        np_path = os.path.join(save_fig_dir, 'occ_observations.npy')
        np.save(np_path, occ_observations)

        np_path = os.path.join(save_fig_dir, 'swath_observations.npy')
        np.save(np_path, swath_observations)

        np_path = os.path.join(save_fig_dir, 'footprint_observations.npy')
        np.save(np_path, footprint_observations)

        # np_path = os.path.join(save_fig_dir, 'target_observations.npy')
        # np.save(np_path, target_observations)

        np_path = os.path.join(save_fig_dir, 'centroids_observations.npy')
        np.save(np_path, centroids_observations)

        np_path = os.path.join(save_fig_dir, 'cur_grids.npy')
        np.save(np_path, cur_grids)


    def get_current_grid(self, ship_body):
        return (int(ship_body.position.x // self.grid_width), int(ship_body.position.y // self.grid_height))