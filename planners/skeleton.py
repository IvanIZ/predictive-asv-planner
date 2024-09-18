import logging
import os, sys
sys.path.append("/home/n5zhong/ASV/predictive-asv-planner")
import pickle
import time

import cv2
import networkx as nx
from scipy.signal import savgol_filter
import sknw
from skimage import draw
from skimage.morphology import skeletonize

import numpy as np
from matplotlib import pyplot as plt

from submodules.src.cost_map import CostMap
from submodules.src.ship import Ship
from submodules.src.geometry.polygon import poly_centroid
from submodules.src.utils.plot import Plot
from submodules.src.utils.utils import Path
from submodules.src.utils.utils import DotDict, resample_path
import threading
import copy

# ROS Humble Related
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32
from geometry_msgs.msg import Pose2D, Point, Polygon, Point32
from custom_msgs.msg import PolygonArray

SCALE = 10         # scales the image by this factor when applying skeletonize
SCALE_ORIGINAL = 10
WD_LEN = 51       # length of filter window for path smoothing
GOAL_EXTEND = 8   # in metres, the amount to extend the goal y coordinate by,
                  # avoids having to pick an arbitrary goal point since we
                  # can just cut path at a fixed y position
SHRINK = 0.2      # amount to take off each obstacle vertex to ensure path is always found

class SkeletonNode(Node):

    def __init__(self, cfg, start_trial_idx=0):
        super().__init__('skeleton_node')

        self.cfg = cfg

        if cfg.planner != 'skeleton':
            raise Exception("Wrong planner. Please run the planner node specified in the config file.")

        # publishers
        self.path_publisher = self.create_publisher(Float32MultiArray, 'path', 1)
        self.cur_trial_idx_pub = self.create_publisher(Int32, 'high_level_trial_idx', 1)

        # subscribers
        self.poly_sub = self.create_subscription(PolygonArray, 'polygons', self.poly_callback, 1)
        self.obs = None

        self.ship_pose_sub = self.create_subscription(Pose2D, 'ship_pose', self.ship_pose_callback, 1)
        self.ship_pos = None        # ship pose (x, y, theta) in meter scale
        self.ship_pos_scaled = None     # ship pose (x, y, theta) in costmap scalse

        self.goal_sub = self.create_subscription(Point, 'goal', self.goal_callback, 1)
        self.goal = None

        self.trial_idx_sub = self.create_subscription(Int32, 'trial_idx', self.trial_idx_callback, 1)
        self.cur_trial_idx = start_trial_idx       # current trial the planner is running on
        self.sim_trial_idx = None       # trial the sim node is running on

        # self.occ_fig, self.occ_ax = plt.subplots(figsize=(10, 10))
        # self.occ_ax.set_xlabel('')
        # self.occ_ax.set_xticks([])
        # self.occ_ax.set_ylabel('')
        # self.occ_ax.set_yticks([])

        # frequency control
        self.wait_rate = self.create_rate(5)

        # replan distance
        self.replan_dist = 0.0
        self.prev_replan_pos = None

        print("Skeleton planner node initialzed --> Skeleton scale: ", SCALE, "; replan distance: ", self.replan_dist)


    def poly_callback(self, msg):
        polygons = msg.polygons
        obs = []
        for poly in polygons:
            verts = []
            points = poly.points
            for pt in points:
                verts.append([pt.x, pt.y])
            verts = np.array(verts)     # each polygon of shape (n, 2)
            
            obs.append(verts)
        self.obs = obs

    def trial_idx_callback(self, msg):
        self.sim_trial_idx = msg.data

    def goal_callback(self, goal_msg):
        self.goal = (goal_msg.x, goal_msg.y)

    def ship_pose_callback(self, pose_msg):
        self.ship_pos = np.array([pose_msg.x, pose_msg.y, pose_msg.theta])
        self.ship_pos_scaled = np.array([pose_msg.x * self.cfg.costmap.scale, pose_msg.y * self.cfg.costmap.scale, pose_msg.theta])

    
    def skeleton_planner(self, cfg):

        # instantiate main objects
        # the costmap and ship objects are only used for the purposes
        # of plotting and computing metrics, the skeleton planner
        # does not interface with either of these components
        costmap = CostMap(horizon=cfg.a_star.horizon,
                            ship_mass=cfg.ship.mass, **cfg.costmap)
        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
        
        # set distance threshold for removing edges near channel border to 2 boat widths
        vertices = np.asarray(cfg.ship.vertices)
        dist_thres = min(vertices[:, 0].max() - vertices[:, 0].min(),
                        vertices[:, 1].max() - vertices[:, 1].min()) * 2

        # keep track of the planning count
        replan_count = 0
        # keep track of planner rate
        compute_time = []
        num_model_calls = []
        curr_goal = np.asarray([0, np.inf, 0])

        # check first planning instance success
        initial_plan_success = False

        # start main planner loop
        print("Skeleton planner ROS Running...")
        while replan_count < cfg.get('max_replan', np.infty) and rclpy.ok():

            # wait for all planning information to be available
            while rclpy.ok() and ((self.ship_pos is None) or (self.goal is None) or (self.obs is None) or (self.sim_trial_idx is None)):
                self.wait_rate.sleep()

            # check if sim node has already start a new trial
            if self.cur_trial_idx != self.sim_trial_idx:

                # clear everything, and reiterate to ensure info is updated
                print("Starting new trial: ", self.sim_trial_idx)
                self.cur_trial_idx = self.sim_trial_idx
                replan_count = 0
                self.ship_pos = None
                self.goal = None
                curr_goal = None
                self.obs = None
                initial_plan_success = False
                self.prev_replan_pos = None
                continue
            
            # currently in the correct trial but initial plan done without replan
            elif (not cfg.a_star.replan) and initial_plan_success:
                print("Initial plan done without replan! Waiting...")
                self.wait_rate.sleep()
                continue
            
            # replan after a fixed distance
            # if self.prev_replan_pos is not None:
            #     dist_passed = ((self.ship_pos[0] - self.prev_replan_pos[0])**2 + (self.ship_pos[1] - self.prev_replan_pos[1])**2)**(0.5)
            #     if dist_passed < self.replan_dist:
            #         continue

            # start timer
            t0 = time.time()

            if self.goal is not None:
                curr_goal = self.goal

            # stop planning if the remaining total distance is less than a ship length in meter
            if self.goal[1] - self.ship_pos[1] <= 2:
                continue

            # scale x and y by the scaling factor
            # self.ship_pos[:2] = self.ship_pos[:2] * cfg.costmap.scale

            # check if there is new obstacle information
            if self.obs is not None:
                # update costmap
                costmap.update(self.obs, self.ship_pos_scaled[1] - ship.max_ship_length / 2)
            
            global SCALE, SCALE_ORIGINAL
            assert SCALE == SCALE_ORIGINAL, print("Did not reset SCALE properly: ", SCALE, SCALE_ORIGINAL)

            ship_pos = copy.deepcopy(self.ship_pos)
            obs = copy.deepcopy(self.obs)
            self.prev_replan_pos = copy.deepcopy(self.ship_pos)
            print("ship position: ", self.prev_replan_pos[:2], "; start planning...")
            path = self.morph_skeleton(map_shape=(cfg.costmap.m, cfg.costmap.n),
                              state_data={'ship_state': ship_pos, 'goal': curr_goal, 'obstacles': obs},
                              dist_thres=dist_thres, debug=False)
            
            if path is False:

                curr_shrink = SHRINK
                retry_num = 0

                for i in range(50):

                    # global SCALE
                    SCALE = SCALE + 2
    
                    ship_pos = copy.deepcopy(self.ship_pos)
                    obs = copy.deepcopy(self.obs)
                    path = self.morph_skeleton(map_shape=(cfg.costmap.m, cfg.costmap.n),
                                        state_data={'ship_state': ship_pos, 'goal': curr_goal, 'obstacles': obs},
                                        dist_thres=dist_thres, debug=False, shrink=SHRINK)
                    
                    curr_shrink += SHRINK
                    retry_num += 1

                    if path is not False:
                        break

                SCALE = SCALE_ORIGINAL
                
                # ship_pos = copy.deepcopy(self.ship_pos)
                # obs = copy.deepcopy(self.obs)
                # path = self.morph_skeleton(map_shape=(cfg.costmap.m, cfg.costmap.n),
                #                     state_data={'ship_state': ship_pos, 'goal': curr_goal, 'obstacles': obs},
                #                     dist_thres=dist_thres, debug=False, shrink=SHRINK)

            # check if sim node has already start a new trial, discard result
            if self.cur_trial_idx != self.sim_trial_idx:
                continue
            
            # fail to find path
            if path is False:
                print("Shrinking failed to help planner find a path!")
                replan_count += 1
                self.ship_pos = None
                continue
            else:
                print("Planning success: ", replan_count)
                initial_plan_success = True

            path = resample_path(path.T, step_size=cfg.prim.step_size)

            # in skeleton planner, path is already in world scale
            path_true_scale = path

            # # in skeleton planner, path is already in world scale
            # path_true_scale = path.T

            # send path
            path_msg = Float32MultiArray()
            dim_h = MultiArrayDimension()
            dim_h.label = 'height'
            dim_h.size = path_true_scale.shape[0]
            path_msg.layout.dim.append(dim_h)
            dim_w = MultiArrayDimension()
            dim_w.label = 'width'
            dim_w.size = path_true_scale.shape[1]
            path_msg.layout.dim.append(dim_w)
            path_msg.data = path_true_scale.flatten().tolist()
            self.path_publisher.publish(path_msg)

            compute_time.append((time.time() - t0))
            replan_count += 1

            # reset planning information
            self.ship_pos = None
            self.obs = None



    def morph_skeleton(self, map_shape, state_data, dist_thres=None, debug=False, shrink=None):
        im = np.zeros(((map_shape[0] + GOAL_EXTEND) * SCALE, map_shape[1] * SCALE), dtype='uint8')
        obs = []
        # scale the obstacles
        for x in state_data['obstacles']:
            x = np.asarray(x)

            if shrink:
                # apply shrinking to each obstacle
                centre = np.abs(poly_centroid(x))
                x = np.asarray(
                    [[np.sign(a) * (abs(a) - shrink), np.sign(b) * (abs(b) - shrink)] for a, b in x - centre]
                ) + centre

            x = (x * SCALE).astype(np.int32)
            obs.append(x)

        cv2.fillPoly(im, obs, (255, 255, 255))
        im = im.astype(bool)
        ske = skeletonize(~im).astype(np.uint16)

        # remove the pixels on skeleton that are too close to channel borders
        if dist_thres:
            ske[:, :int(dist_thres * SCALE)] = 0
            ske[:, -int(dist_thres * SCALE):] = 0
        
        # add a pixel for the start and final position
        goal = [state_data['goal'][0],
                state_data['goal'][1] + GOAL_EXTEND]  # extend goal y coordinate
        ship_state = state_data['ship_state']

        min_dlist = []  # for debugging
        # 8 nearest neighbours + centre
        nn = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
        # draw line segment to closest pixel in skeleton
        print("ship state 1:", ship_state, "\n")
        for p in [ship_state, goal]:
            p = np.asarray(p) * SCALE

            # get min distance to pixel on skeleton
            ske_c = ske.copy()
            ske_c = ske_c.astype(bool).astype('uint8')
            ske_c[ske_c == 0] = 255  # assumes unoccupied cells are 0
            ske_c[ske_c == 1] = 0  # everything else is considered occupied
            dist = cv2.distanceTransform(src=ske_c,
                                        distanceType=cv2.DIST_L2,
                                        maskSize=cv2.DIST_MASK_PRECISE)

            if p[1] > ske.shape[0] or p[0] > ske.shape[1]:
                p = (ske.shape[1] - 10, ske.shape[0] - 10)
            
            min_d = dist[int(round(p[1])), int(round(p[0]))]
            min_dlist.append(min_d)

            # check edge case when p is already a pixel on skeleton
            if ske[int(round(p[1])), int(round(p[0]))]:
                # find a neighbouring pixel that is not yet on skeleton
                for j in nn:
                    x, y = p[0] + j[0], p[1] + j[1]
                    if 0 < round(x) < im.shape[1] and 0 < round(y) < im.shape[0] and not ske[int(round(y)), int(round(x))]:
                        ske[int(round(y)), int(round(x))] = 1
                        break

            else:
                # find a pixel on skeleton in circle of radius min_d + [-1, 0, 1, 2]
                for inc in [0, 1, -1, 2]:
                    rr, cc = draw.disk((p[1], p[0]), min_d + inc, shape=ske.shape)
                    if ske[rr, cc].sum() != 0:
                        break

                assert ske[rr, cc].sum() != 0
                # iterate over pixels inside of circle
                x_best, y_best, d_best = None, None, np.inf
                for r, c in zip(rr, cc):
                    d = ((p[0] - c) ** 2 + (p[1] - r) ** 2) ** 0.5
                    if d < d_best and ske[r, c]:
                        # make sure this is not the point p itself
                        if c == p[0] and r == p[1]:
                            continue
                        d_best = d
                        x_best, y_best = c, r

                if x_best is None or y_best is None:
                    raise ValueError

                # draw the line
                lin = draw.line_nd([p[1], p[0]], [y_best, x_best])
                assert ske[lin].sum() != len(lin[0])
                ske[lin] = 1

        # build graph from skeleton
        graph = sknw.build_sknw(ske)

        # find the key for the start and goal nodes in the graph
        s_key, g_key = (ship_state[0] * SCALE, ship_state[1] * SCALE), (goal[0] * SCALE, goal[1] * SCALE)
        s_node, g_node = {s_key: None, 'dist': np.inf}, {g_key: None, 'dist': np.inf}
        print("ship state 2:", ship_state, "\n")
        for i in graph.nodes:
            curr_node = graph.nodes[i]['o']
            for node, key in zip([s_node, g_node], [s_key, g_key]):
                if node['dist'] == 0:
                    continue  # already found the node in the graph so skip
                d = ((curr_node[0] - key[1]) ** 2 + (curr_node[1] - key[0]) ** 2) ** 0.5
                if d < node['dist']:
                    node[key] = i
                    node['dist'] = d

        s_node, g_node = s_node[s_key], g_node[g_key]
        assert s_node is not None and g_node is not None

        # skip if start and goal nodes are the same
        if s_node == g_node:
            return False

        # define heuristic function
        h = lambda a, b: ((graph.nodes[a]['o'][0] - graph.nodes[b]['o'][0]) ** 2 + (
                graph.nodes[a]['o'][1] - graph.nodes[b]['o'][1]) ** 2) ** 0.5
        try:
            path = nx.algorithms.astar_path(graph, s_node, g_node, heuristic=h)
        except nx.NetworkXNoPath:
            return False

        # now build path
        full_path = []
        for i, j in zip(path[:-1], path[1:]):
            # get edge
            edge = graph.edges[i, j]['pts']

            # reverse order of pts if necessary
            if tuple(edge[0]) not in {tuple(item) for item in graph.nodes[i]['pts']}:
                edge = edge[::-1]

            full_path.extend(edge)

        # convert to 2d numpy array
        full_path = np.asarray(full_path).T[::-1]  # shape is 2 x n

        # apply quadratic smoothing
        smooth_path = np.asarray([savgol_filter(full_path[0], WD_LEN, 2, mode='nearest'),
                                savgol_filter(full_path[1], WD_LEN, 2, mode='nearest')])

        # truncate path up until original goal
        smooth_path = smooth_path[..., smooth_path[1] <= (goal[1] - GOAL_EXTEND) * SCALE]

        if debug:
            f, ax = plt.subplots(1, 4)
            ax[0].imshow(im, cmap='gray', origin='lower')
            ax[0].set_title('Original image')

            ax[1].imshow(ske, cmap='gray', origin='lower')
            for p, min_d in zip([ship_state, goal], min_dlist):
                p = np.asarray(p) * SCALE
                ax[1].plot(np.cos(np.arange(0, 2 * np.pi, 0.1)) * min_d + p[0],
                        np.sin(np.arange(0, 2 * np.pi, 0.1)) * min_d + p[1])
            ax[1].set_title('Skeleton')

            ax[2].imshow(im, cmap='gray', origin='lower')
            # draw edges by pts
            for (s, e) in graph.edges():
                ps = graph[s][e]['pts']
                ax[2].plot(ps[:, 1], ps[:, 0], 'green')

            # draw node by o
            nodes = graph.nodes()
            ps = np.array([nodes[i]['o'] for i in nodes])
            ax[2].plot(ps[:, 1], ps[:, 0], 'r.')
            ax[2].set_title('Graph')

            # show on plots start and goal points
            for a in ax:
                a.plot(ship_state[0] * SCALE, ship_state[1] * SCALE, 'gx')
                a.plot(goal[0] * SCALE, goal[1] * SCALE, 'gx')

            # draw path
            ax[3].imshow(im, cmap='gray', origin='lower')
            ax[3].plot(full_path[0], full_path[1], 'm--')
            ax[3].plot(smooth_path[0], smooth_path[1], 'c')
            ax[3].set_title('Path')

            plt.show()

        # compute the heading along path
        theta = [
            np.arctan2(j[1] - i[1], j[0] - i[0])
            for i, j in zip(smooth_path.T[:-1], smooth_path.T[1:])
        ]

        # transform path to original scaling and then return
        return np.c_[smooth_path.T[:-1] / SCALE, theta].T  # shape is 3 x n



if __name__ == '__main__':
    rclpy.init(args=None)
    cfg_file = 'configs/sim2d_config.yaml'
    cfg = cfg = DotDict.load_from_file(cfg_file)

    node = SkeletonNode(cfg=cfg, start_trial_idx=0)

    # Spin in a separate thread NOTE: doing this to unblock rate. Should figure out a better way!
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    try:
        # rclpy.spin(node)
        node.skeleton_planner(cfg=cfg)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
