import logging
import os, sys
sys.path.append("/home/n5zhong/ASV/predictive-asv-planner")
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt

from ship_ice_planner.src.utils.utils import DotDict
import threading
import copy

# ROS Humble Related
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32
from geometry_msgs.msg import Pose2D, Point, Polygon, Point32
from custom_msgs.msg import PolygonArray

STEP_SIZE = 0.0125  # step size when sampling path, should be similar to other planners for fair comparison

class StraightNode(Node):

    def __init__(self, cfg, start_trial_idx=0):
        super().__init__('straight_node')

        self.cfg = cfg

        if cfg.planner != 'straight':
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

        print("Straight planner node initialzed")


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

    
    def straight_planner(self, cfg):

        # keep track of the planning count
        replan_count = 0
        curr_goal = np.asarray([0, np.inf, 0])
        # keep track of planner rate
        compute_time = []

        # check first planning instance success
        initial_plan_success = False

        # start main planner loop
        print("Straight planner ROS Running...")
        while replan_count < cfg.get('max_replan', np.infty) and rclpy.ok():

            # wait for all planning information to be available
            while rclpy.ok() and ((self.ship_pos is None) or (self.goal is None) or (self.sim_trial_idx is None)):
                # if self.ship_pos is None:
                #     print("waiting for ship pose")
                # if self.goal is None:
                #     print("waiting for goal")
                # if self.sim_trial_idx is None:
                #     print("waiting for trial idx")
                self.wait_rate.sleep()

            # check if sim node has already start a new trial
            if self.cur_trial_idx != self.sim_trial_idx:

                # clear everything, and reiterate to ensure info is updated
                print("Starting new trial: ", self.sim_trial_idx)
                self.cur_trial_idx = self.sim_trial_idx
                replan_count = 0
                self.ship_pos = None
                self.goal = None
                initial_plan_success = False
                continue
            
            # currently in the correct trial but initial plan done without replan
            elif (not cfg.a_star.replan) and initial_plan_success:
                print("Initial plan done without replan! Waiting...")
                self.wait_rate.sleep()

            # start timer
            t0 = time.time()

            if self.goal is not None:
                curr_goal = self.goal

            # stop planning if the remaining total distance is less than a ship length in meter
            if self.goal[1] - self.ship_pos[1] <= 2:
                continue

            # scale x and y by the scaling factor
            # self.ship_pos[:2] = self.ship_pos[:2] * cfg.costmap.scale

            # compute straight path
            ship_pos = copy.deepcopy(self.ship_pos)
            y = np.arange(ship_pos[1], curr_goal[1], STEP_SIZE)
            path = np.asarray([[ship_pos[0]] * len(y), y, [np.pi / 2] * len(y)])
            
            # check if sim node has already start a new trial, discard result
            if self.cur_trial_idx != self.sim_trial_idx:
                continue

            # in straight planner, path is already in world scale
            path_true_scale = path.T

            # send path only once per trial
            if not initial_plan_success:
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

                initial_plan_success = True

            compute_time.append((time.time() - t0))
            replan_count += 1

            # reset planning information
            self.ship_pos = None



if __name__ == '__main__':
    rclpy.init(args=None)
    cfg_file = 'configs/sim2d_config.yaml'
    cfg = DotDict.load_from_file(cfg_file)

    node = StraightNode(cfg=cfg, start_trial_idx=0)

    # Spin in a separate thread NOTE: doing this to unblock rate. Should figure out a better way!
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    try:
        # rclpy.spin(node)
        node.straight_planner(cfg=cfg)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
