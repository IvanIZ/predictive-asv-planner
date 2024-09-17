import os, sys
sys.path.append("/home/n5zhong/ASV/predictive-asv-planner")
import time

import numpy as np
from matplotlib import pyplot as plt

from ship_ice_planner.src.path_evaluator import PredictivePathEvaluator
from ship_ice_planner.src.primitives import Primitives
from ship_ice_planner.src.ship import Ship
from ship_ice_planner.src.swath import generate_swath, view_all_swaths
from ship_ice_planner.src.utils.plot import Plot
from ship_ice_planner.src.utils.utils import Path
from ship_ice_planner.src.occupancy_grid.occupancy_map import OccupancyGrid
from ship_ice_planner.src.cost_map_occ import CostMap_Occupancy
from ship_ice_planner.src.a_star_predictive import AStar_Predictive
from ship_ice_planner.src.utils.utils import DotDict
import threading
import copy

# ROS Humble Related
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32
from geometry_msgs.msg import Pose2D, Point, Polygon, Point32
from custom_msgs.msg import PolygonArray

# global vars for dir/file names
PLOT_DIR = 'plots'
PATH_DIR = 'paths'
METRICS_FILE = 'metrics.txt'


class PredictiveNode(Node):

    def __init__(self, cfg, start_trial_idx=0):
        super().__init__('lattice_node')

        self.cfg = cfg

        # publishers
        self.path_publisher = self.create_publisher(Float32MultiArray, 'path', 1)
        self.cur_trial_idx_pub = self.create_publisher(Int32, 'high_level_trial_idx', 1)

        # subscribers
        self.occ_sub = self.create_subscription(Float32MultiArray, 'occ_map', self.occ_callback, 1)
        self.callback_count = 0
        self.occ_map = None

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

        # frequency control
        self.wait_rate = self.create_rate(5)

        # replan distance
        self.replan_dist = 5.0
        self.prev_replan_pos = None


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
    
    def occ_callback(self, msg):
        # get dimensions from layout
        height = msg.layout.dim[0].size
        width = msg.layout.dim[1].size

        # convert flat data back to occ map
        self.occ_map = np.array(msg.data, dtype=np.float32).reshape((height, width))

    
    def predictive_planner(self, cfg):

        # instantiate main objects
        if cfg.planner != 'predictive':
            raise Exception("Wrong planner. Please run the planner node specified in the config file.")
        
        costmap = CostMap_Occupancy(cfg=cfg, horizon=cfg.a_star.horizon,
                        ship_mass=cfg.ship.mass, **cfg.costmap)
        use_ice_model = True

        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
        prim = Primitives(cache=False, **cfg.prim)
        swath_dict = generate_swath(ship, prim, cache=False,  model_inference=False)
        debug_ice_model = False

        ship_no_padding = Ship(scale=cfg.costmap.scale, vertices=cfg.ship.vertices, padding=0, mass=cfg.ship.mass)
        swath_dict_no_padding = generate_swath(ship_no_padding, prim, cache=False, model_inference=True)
        # view_all_swaths(swath_dict_no_padding); exit()

        occupancy = OccupancyGrid(grid_width=cfg.occ.grid_size, grid_height=cfg.occ.grid_size, map_width=cfg.occ.map_width, map_height=cfg.occ.map_height, ship_body=None)

        a_star = AStar_Predictive(cmap=costmap,
                ke_map=None,
                concentration=cfg.concentration,
                prim=prim,
                ship=ship,
                swath_dict=swath_dict,
                swath_dict_no_padding=swath_dict_no_padding,
                ship_no_padding=ship_no_padding,
                use_ice_model=use_ice_model,
                **cfg.a_star)
    
        path_eval = PredictivePathEvaluator(prim=prim, cmap_scale=cfg.costmap.scale, concentration=cfg.concentration)

        path_obj = Path()

        # keep track of the planning count
        replan_count = 0
        # keep track of planner rate
        compute_time = []
        num_model_calls = []
        last_goal_y = np.inf

        # check first planning instance success
        initial_plan_success = False

        # set prediction horizon
        prediction_horizon = 10 * cfg.costmap.scale
        prediction_horizon = None

        # start main planner loop
        print("Lattice planner ROS Running...")
        while replan_count < cfg.get('max_replan', np.infty) and rclpy.ok():

            # wait for all planning information to be available
            while rclpy.ok() and ((self.ship_pos is None) or (self.goal is None) or (self.occ_map is None and self.obs is None) or (self.sim_trial_idx is None)):
            # while rclpy.ok() and ((self.ship_pos is None) or (self.goal is None) or (self.occ_map is None) or (self.obs is None) or (self.sim_trial_idx is None)):
                self.wait_rate.sleep()

            # check if sim node has already start a new trial
            if self.cur_trial_idx != self.sim_trial_idx:

                # clear everything, and reiterate to ensure info is updated
                print("Starting new trial: ", self.sim_trial_idx)
                self.cur_trial_idx = self.sim_trial_idx
                path_obj = Path()
                replan_count = 0
                self.ship_pos = None
                self.goal = None
                self.occ_map = None
                self.obs = None
                initial_plan_success = False
                self.prev_replan_pos = None
                continue
            
            # currently in the correct trial but initial plan done without replan
            elif (not cfg.a_star.replan) and initial_plan_success:
                print("Initial plan done without replan! Waiting...")
                self.wait_rate.sleep()

            # start timer
            t0 = time.time()

            # get ice model observation
            footprint = None

            # get footprint  NOTE: here we want unscaled, unpadded vertices
            occupancy.compute_ship_footprint_planner(ship_state=self.ship_pos, ship_vertices=cfg.ship.vertices)
            footprint = np.copy(occupancy.footprint)

            # stop planning if the remaining total distance is less than a ship length in meter
            if self.goal[1] - self.ship_pos[1] <= 2:
                continue

            # compute next goal NOTE: could be simplified in ROS version
            if self.goal is not None:
                goal_y = min(self.goal[1], (self.ship_pos[1] + cfg.a_star.horizon)) * cfg.costmap.scale
                last_goal_y = self.goal[1]
            else:
                goal_y = min(last_goal_y, (self.ship_pos[1] + cfg.a_star.horizon)) * cfg.costmap.scale

            # check if there is new obstacle information
            costmap.update(occ_map=self.occ_map)
            
            # replan after a fixed distance
            if self.prev_replan_pos is not None:
                dist_passed = ((self.ship_pos[0] - self.prev_replan_pos[0])**2 + (self.ship_pos[1] - self.prev_replan_pos[1])**2)**(0.5)
                if dist_passed < self.replan_dist:
                    continue

            # compute path to goal
            ship_pos = copy.deepcopy(self.ship_pos_scaled)     # probably don't need it but just to be safe
            self.prev_replan_pos = copy.deepcopy(self.ship_pos)
            print("ship position: ", self.prev_replan_pos[:2], "; start planning...")
            occ_map = np.copy(self.occ_map)
            plan_start = time.time()
            search_result = a_star.search(
                start=(ship_pos[0], ship_pos[1], ship_pos[2]),
                goal_y=goal_y,
                occ_map=occ_map,
                centroids=None,
                footprint=footprint,
                ship_vertices=cfg.ship.vertices, 
                use_ice_model=use_ice_model,
                debug=debug_ice_model, 
                prediction_horizon=prediction_horizon, 
            )
            print("planning time: ", time.time() - plan_start)

            # check if sim node has already start a new trial, discard result
            if self.cur_trial_idx != self.sim_trial_idx:
                continue
            
            # fail to find path
            if not search_result:
                print("Planner failed to find a path!")
                replan_count += 1
                self.ship_pos = None
                self.occ_map = None
                continue
            else:
                print("Planning success: ", replan_count)
                initial_plan_success = True
            
            # unpack result
            (full_path, full_swath), \
            (node_path, node_path_length), \
            (node_path_smth, new_nodes), \
            (nodes_expanded, g_score, swath_cost, length, edge_seq, swath_costs, swath_ins, horizontal_shifts, path_len_keys) = search_result
            x1, y1, _ = node_path  # this is the original node path prior to smoothing
            x2, y2 = new_nodes if len(new_nodes) != 0 else (0, 0)  # these are the nodes added from smoothing

            # get swath costs for old path on new observation using predictive occ diff (NOTE only do this when this is not the first path)
            if path_obj.node_path is not None:

                old_swath_costs = path_eval.eval_path(occ_map=np.copy(self.occ_map), node_path=path_obj.node_path.T, ship_pose=self.ship_pos_scaled, 
                                                    swath_ins=path_obj.swath_ins, horizontal_shifts=path_obj.horizontal_shifts, ship_vertices=cfg.ship.vertices,
                                                    path_len_keys=path_obj.path_len_keys, debug=False)
                
                new_swath_costs = path_eval.eval_path(occ_map=np.copy(self.occ_map), node_path=node_path.T, ship_pose=self.ship_pos_scaled, 
                                                    swath_ins=swath_ins, horizontal_shifts=horizontal_shifts, ship_vertices=cfg.ship.vertices,
                                                    path_len_keys=path_len_keys, debug=False)

            else:
                old_swath_costs = None
                new_swath_costs = None
            
            drift_threshold = 0.25
            send_new_path, old_cost, new_cost = path_obj.update_occDiff(old_swath_costs=old_swath_costs, node_path=node_path.T, swath_costs=new_swath_costs, ship_pos=self.ship_pos_scaled,
                                            threshold_dist=cfg.get('threshold_dist', 0) * length,
                                            threshold_cost=cfg.get('threshold_cost'), 
                                            drift_threshold=drift_threshold)
            print("send path: ", send_new_path, "; old cost: ", old_cost, "; new cost: ", new_cost, "; drift threshold: ", drift_threshold)

            # update path object
            if send_new_path:
                path_obj.node_path = node_path
                path_obj.swath_ins = swath_ins
                path_obj.horizontal_shifts = horizontal_shifts
                path_obj.path_len_keys = path_len_keys
                path_obj.path = full_path

            # send path, return path in original scale
            # shape will be n x 3
            path_true_scale = np.c_[(path_obj.path[:2] / cfg.costmap.scale).T, path_obj.path[2]]  # TODO: confirm heading is ok

            if send_new_path:
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
            else:
                ...

            compute_time.append((time.time() - t0))
            replan_count += 1

            # reset planning information
            self.ship_pos = None
            self.occ_map = None
            self.obs = None




if __name__ == '__main__':
    rclpy.init(args=None)
    cfg_file = 'configs/sim2d_config.yaml'
    cfg = cfg = DotDict.load_from_file(cfg_file)

    node = PredictiveNode(cfg=cfg, start_trial_idx=0)

    # Spin in a separate thread NOTE: doing this to unblock rate. Should figure out a better way!
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    try:
        # rclpy.spin(node)
        node.predictive_planner(cfg=cfg)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
