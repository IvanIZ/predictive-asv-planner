""" Main script for running simulation experiments with autonomous ship navigation in ice """
import os, sys
sys.path.append("/home/n5zhong/ASV/predictive-asv-planner")
import pickle
import random
from operator import le, ge

import numpy as np
import pymunk
import pymunk.constraints
from pymunk import Vec2d
import threading
from matplotlib import pyplot as plt, patches

from ship_ice_planner.src.controller.dp import DP
from ship_ice_planner.src.cost_map import CostMap
from ship_ice_planner.src.evaluation.metrics import tracking_error, total_work_done
from ship_ice_planner.src.geometry.polygon import poly_area
from ship_ice_planner.src.ship import Ship
from ship_ice_planner.src.utils.plot import Plot
from ship_ice_planner.src.utils.sim_utils import generate_sim_obs
from ship_ice_planner.src.utils.utils import DotDict
from ship_ice_planner.src.occupancy_grid.occupancy_map import OccupancyGrid

# ROS Humble Related
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32
from geometry_msgs.msg import Pose2D, Point, Polygon, Point32
from custom_msgs.msg import PolygonArray


class Sim2DNode(Node):

    def __init__(self, cfg=None):
        super().__init__('simulation_node')

        self.occupancy = OccupancyGrid(grid_width=cfg.occ.grid_size, grid_height=cfg.occ.grid_size, map_width=cfg.occ.map_width, map_height=cfg.occ.map_height, ship_body=None)
        self.cfg = cfg

        # publishers
        self.occ_publisher = self.create_publisher(Float32MultiArray, 'occ_map', 1)
        self.ship_pose_publisher = self.create_publisher(Pose2D, 'ship_pose', 1)
        self.goal_publisher = self.create_publisher(Point, 'goal', 1)
        self.trial_idx_publisher = self.create_publisher(Int32, 'trial_idx', 1)
        self.poly_publisher = self.create_publisher(PolygonArray, 'polygons', 1)

        # subscribers
        self.path = self.create_subscription(Float32MultiArray, 'path', self.path_callback, 1)
        self.path = None
        self.new_path = None
        self.new_path_received = False
        
        # frequency control
        self.wait_path_rate = self.create_rate(5)
        self.rate_dt = self.create_rate(50)         # based on controller dt = 0.02
        

    def path_callback(self, path_msg):
        # get dimensions from layout
        height = path_msg.layout.dim[0].size
        width = path_msg.layout.dim[1].size

        # convert flat data back to occ map
        self.new_path = np.array(path_msg.data, dtype=np.float32).reshape((height, width))
        self.new_path_received = True


    def sim(self, trial_idx=0, cfg_file=None, init_queue=None):

        # load config
        cfg = DotDict.load_from_file(cfg_file)
        cfg.cfg_file = cfg_file

        # get important params
        steps = cfg.sim.steps
        t_max = cfg.sim.t_max if cfg.sim.t_max else np.inf
        horizon = cfg.a_star.horizon
        replan = cfg.a_star.replan
        seed = cfg.get('seed', None)
        dt = cfg.controller.dt

        np.random.seed(seed)  # seed to make sims deterministic
        random.seed(seed)

        # setup pymunk environment
        space = pymunk.Space()  # threaded=True causes some issues
        space.iterations = cfg.sim.iterations
        space.gravity = cfg.sim.gravity
        space.damping = cfg.sim.damping

        # keep track of running total of total kinetic energy / total impulse
        # computed using pymunk api call, source code here
        # https://github.com/slembcke/Chipmunk2D/blob/edf83e5603c5a0a104996bd816fca6d3facedd6a/src/cpArbiter.c#L158-L172
        system_ke_loss = []   # https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.total_ke
                                # source code in Chimpunk2D cpArbiterTotalKE
        total_ke = [0, []]  # keep track of both running total and ke at each collision
        total_impulse = [0, []]
        # keep track of running total of work
        total_work = [0, []]

        total_dis = 0 
        prev_state = None   

        # keep track of all the obstacles that collide with ship
        clln_obs = set()

        # keep track of contact points
        contact_pts = []

        # setup a collision callback to keep track of total ke
        # def pre_solve_handler(arbiter, space, data):
        #     nonlocal ship_ke
        #     ship_ke = arbiter.shapes[0].body.kinetic_energy
        #     print('ship_ke', ship_ke, 'mass', arbiter.shapes[0].body.mass, 'velocity', arbiter.shapes[0].body.velocity)
        #     return True
        # # http://www.pymunk.org/en/latest/pymunk.html#pymunk.Body.each_arbiter

        # setup pymunk collision callbacks
        def pre_solve_handler(arbiter, space, data):
            ice_body = arbiter.shapes[1].body
            ice_body.pre_collision_KE = ice_body.kinetic_energy  # hacky, adding a field to pymunk body object
            return True

        def post_solve_handler(arbiter, space, data):
            nonlocal total_ke, system_ke_loss, total_impulse, clln_obs
            ship_shape, ice_shape = arbiter.shapes

            system_ke_loss.append(arbiter.total_ke)

            total_ke[0] += arbiter.total_ke
            total_ke[1].append(arbiter.total_ke)

            total_impulse[0] += arbiter.total_impulse.length
            total_impulse[1].append(list(arbiter.total_impulse))

            if arbiter.is_first_contact:
                clln_obs.add(arbiter.shapes[1])

            # max of two sets of points, easy to see with a picture with two overlapping convex shapes
            # find the impact locations in the local coordinates of the ship
            for i in arbiter.contact_point_set.points:
                contact_pts.append(list(arbiter.shapes[0].body.world_to_local((i.point_b + i.point_a) / 2)))

        # handler = space.add_default_collision_handler()
        handler = space.add_collision_handler(1, 2)
        # from pymunk docs
        # post_solve: two shapes are touching and collision response processed
        handler.pre_solve = pre_solve_handler
        handler.post_solve = post_solve_handler

        _, start, obs_dicts = init_queue.values()
        
        # filter out obstacles that have zero area
        obs_dicts[:] = [ob for ob in obs_dicts if poly_area(ob['vertices']) != 0]
        obstacles = [ob['vertices'] for ob in obs_dicts]

        # we don't case about goal_x; goal_y + 1 to overshot the planner a bit for avoiding stopping early
        goal = (0, cfg.goal_y + 1)

        # if running baseline, send polygons
        if cfg.planner == 'skeleton' or cfg.planner == 'straight' or cfg.planner == 'lattice':
            poly_msg = PolygonArray()
            ros_polygons = []
            for obstacle in obstacles:
                poly = Polygon()
                points = []
                for vert in obstacle:
                    pt = Point32()
                    pt.x = vert[0]
                    pt.y = vert[1]
                    points.append(pt)
                poly.points = points
                ros_polygons.append(poly)
            
            poly_msg.polygons = ros_polygons
            self.poly_publisher.publish(poly_msg)

        elif cfg.planner == 'predictive':

            # prepare obstacle occupancy information
            if self.occupancy.map_height == 40:
                raw_ice_binary = self.occupancy.compute_occ_img(obstacles=obstacles, ice_binary_w=235, ice_binary_h=774)
            else:
                raw_ice_binary = self.occupancy.compute_occ_img(obstacles=obstacles, ice_binary_w=235, ice_binary_h=1355)
            self.occupancy.compute_con_gridmap(raw_ice_binary=raw_ice_binary, save_fig_dir=None)
            occ_map = np.copy(self.occupancy.occ_map)

            occ_msg = Float32MultiArray()
            dim_h = MultiArrayDimension()
            dim_h.label = 'height'
            dim_h.size = occ_map.shape[0]
            occ_msg.layout.dim.append(dim_h)
            dim_w = MultiArrayDimension()
            dim_w.label = 'width'
            dim_w.size = occ_map.shape[1]
            occ_msg.layout.dim.append(dim_w)
            occ_msg.data = occ_map.flatten().tolist()
            self.occ_publisher.publish(occ_msg)

        # send ship pose
        ship_state_msg = Pose2D()
        ship_state_msg.x = start[0]
        ship_state_msg.y = start[1]
        ship_state_msg.theta = start[2]
        self.ship_pose_publisher.publish(ship_state_msg)

        # send goal
        goal_msg = Point()
        goal_msg.x = float(goal[0])
        goal_msg.y = float(goal[1])
        self.goal_publisher.publish(goal_msg)

        # send trial idx
        trial_idx_msg = Int32()
        trial_idx_msg.data = trial_idx
        self.trial_idx_publisher.publish(trial_idx_msg)

        # clear current path to receive new path
        self.new_path = None
        self.path = None
        self.new_path_received = False

        polygons = generate_sim_obs(space, obs_dicts, cfg.sim.obstacle_density)
        for p in polygons:
            p.collision_type = 2

        # initialize ship sim objects
        ship_body, ship_shape = Ship.sim(cfg.ship.vertices, start)
        ship_shape.collision_type = 1
        space.add(ship_body, ship_shape)
        # run initial simulation steps to let environment settle
        for _ in range(1000):
            space.step(dt / steps)
        prev_obs = CostMap.get_obs_from_poly(polygons)

        # Wait for the first path. Keeping publishing while waiting
        while rclpy.ok() and (self.new_path is None):
            self.trial_idx_publisher.publish(trial_idx_msg)
            if cfg.planner == 'predictive':
                self.occ_publisher.publish(occ_msg)
            else:
                self.poly_publisher.publish(poly_msg)
            self.ship_pose_publisher.publish(ship_state_msg)
            self.goal_publisher.publish(goal_msg)
            self.wait_path_rate.sleep()

        self.path = np.copy(self.new_path[self.new_path[:, 1] < horizon + self.new_path[0, 1]])
        self.new_path_received = False

        # setup dp controller
        cx = self.path.T[0]
        cy = self.path.T[1]
        ch = self.path.T[2]
        dp = DP(x=start[0], y=start[1], yaw=start[2],
                cx=cx, cy=cy, ch=ch, output_dir=cfg.output_dir,
                **cfg.controller)
        state = dp.state

        # initialize plotting/animation
        if cfg.anim.show:
            plot = Plot(
                np.zeros((cfg.costmap.m, cfg.costmap.n)), obs_dicts, path=self.path.T,
                ship_pos=start, ship_vertices=np.asarray(ship_shape.get_vertices()),
                horizon=horizon, map_figsize=None, y_axis_limit=cfg.plot.y_axis_limit,
                target=tuple(dp.setpoint[:2]), inf_stream=True, goal=goal[1]
            )
            plt.ion()  # turn on interactive mode
            plt.show()

        ship_state = ([], [])  # keep track of ship path
        past_path = ([], [])  # keep track of planned path behind ship
        t = 0  # start time tick
        goal_op = ge if not cfg.get('reverse_dir') else le

        try:
            visualize = False
            work = 0.0

            # get updated obstacles
            plot.animate_sim(save_fig_dir=os.path.join(cfg.output_dir, 't' + str(trial_idx))
                                if (cfg.anim.save and cfg.output_dir) else None, suffix=t)
            
            # main simulation loop
            while t < t_max:
                t += 1
                if t >= t_max:
                    print('Reached max time: ', t_max)
                    break

                # for len=40 environment without replan, we set goal as 30 to agree with planning horizon
                if goal_op(ship_body.position.y, cfg.goal_y):
                    break

                if t % cfg.anim.plan_steps == 0:
                    print('Simulation time {} / {}, ship position x={} y={}'
                      .format(t, t_max, ship_body.position.x, ship_body.position.y), end='\r')

                    # get updated obstacles
                    obstacles = CostMap.get_obs_from_poly(polygons)

                    # update work metric
                    work = total_work_done(prev_obs, obstacles)
                    total_work[0] += work
                    total_work[1].append(work)
                    prev_obs = obstacles

                    if replan:

                        # send new information for replan
                        # send trial idx
                        trial_idx_msg = Int32()
                        trial_idx_msg.data = trial_idx
                        self.trial_idx_publisher.publish(trial_idx_msg)

                        # if running baseline, send polygons
                        if cfg.planner == 'skeleton' or cfg.planner == 'straight' or cfg.planner == 'lattice':
                            poly_msg = PolygonArray()
                            ros_polygons = []
                            for obstacle in obstacles:
                                poly = Polygon()
                                points = []
                                for vert in obstacle:
                                    pt = Point32()
                                    pt.x = vert[0]
                                    pt.y = vert[1]
                                    points.append(pt)
                                poly.points = points
                                ros_polygons.append(poly)
                            
                            poly_msg.polygons = ros_polygons
                            self.poly_publisher.publish(poly_msg)

                        elif cfg.planner == 'predictive':

                            # prepare obstacle occupancy for replan
                            if self.occupancy.map_height == 40:
                                raw_ice_binary = self.occupancy.compute_occ_img(obstacles=obstacles, ice_binary_w=235, ice_binary_h=774)
                            else:
                                raw_ice_binary = self.occupancy.compute_occ_img(obstacles=obstacles, ice_binary_w=235, ice_binary_h=1355)
                            self.occupancy.compute_con_gridmap(raw_ice_binary=raw_ice_binary, save_fig_dir=None)
                            occ_map = np.copy(self.occupancy.occ_map)

                            msg = Float32MultiArray()
                            dim_h = MultiArrayDimension()
                            dim_h.label = 'height'
                            dim_h.size = occ_map.shape[0]
                            msg.layout.dim.append(dim_h)
                            dim_w = MultiArrayDimension()
                            dim_w.label = 'width'
                            dim_w.size = occ_map.shape[1]
                            msg.layout.dim.append(dim_w)
                            msg.data = occ_map.flatten().tolist()
                            self.occ_publisher.publish(msg)

                        # send ship pose
                        ship_state_msg = Pose2D()
                        ship_state_msg.x = state.x
                        ship_state_msg.y = state.y
                        ship_state_msg.theta = state.yaw
                        self.ship_pose_publisher.publish(ship_state_msg)

                    # check for path
                    if self.new_path_received:
                        # confirm path is a minimum of 2 points
                        if len(self.new_path) > 1:
                            self.path = np.copy(self.new_path)
                            cx = self.path.T[0]
                            cy = self.path.T[1]
                            ch = self.path.T[2]
                            dp.target_course.update(cx, cy, ch)
                        self.new_path_received = False

                if self.cfg.planner == 'skeleton':
                    # update DP controller
                    dp(ship_body.position.x,
                    ship_body.position.y,
                    ship_body.angle)

                    # apply velocity commands to ship body
                    ship_body.angular_velocity = state.r * np.pi / 180
                    x_vel, y_vel = state.get_global_velocity()  # get velocities in global frame
                    ship_body.velocity = Vec2d(x_vel, y_vel)

                else:
                    # call ideal controller
                    omega, global_velocity = dp.ideal_control(ship_body.position.x,
                    ship_body.position.y,
                    ship_body.angle)

                    # apply velocity commands to ship body from ideal controller
                    ship_body.angular_velocity = omega
                    ship_body.velocity = Vec2d(global_velocity[0], global_velocity[1])

                # move simulation forward
                for _ in range(steps):
                    space.step(dt / steps)

                # update ship pose
                state.update_pose(ship_body.position.x,
                                ship_body.position.y,
                                ship_body.angle)

                ship_state[0].append(state.x)
                ship_state[1].append(state.y)

                if prev_state is not None:
                    dis = ((state.x - prev_state[0])**2 + (state.y - prev_state[1])**2)**(0.5)
                    total_dis += dis
                prev_state = [state.x, state.y]

                # log updates including tracking error
                (e_x, e_y, e_yaw), track_idx = tracking_error([state.x, state.y, state.yaw], self.path, get_idx=True)
                past_path[0].append(self.path[track_idx][0])
                past_path[1].append(self.path[track_idx][1])

                # update setpoint
                x_s, y_s, h_s = dp.get_setpoint()
                dp.setpoint = np.asarray([x_s, y_s, np.unwrap([state.yaw, h_s])[1]])

                if t % cfg.anim.plot_steps == 0 and cfg.anim.show:
                    plt.pause(0.001)

                    # update animation
                    plot.update_path(self.path[track_idx:].T, target=(x_s, y_s), ship_state=ship_state,
                                    past_path=past_path, start_y=self.path[0, 1])
                    plot.update_ship(ship_body, ship_shape, move_yaxis_threshold=cfg.anim.move_yaxis_threshold)
                    plot.update_obstacles(obstacles=CostMap.get_obs_from_poly(polygons))
                    # get updated obstacles
                    plot.animate_sim(save_fig_dir=os.path.join(cfg.output_dir, 't' + str(trial_idx))
                                    if (cfg.anim.save and cfg.output_dir) else None, suffix=t)
                    plot.sim_fig.suptitle('velocity ({:.2f}, {:.2f}, {:.2f}) [m/s, m/s, rad/s]'
                                        '\nsim iteration {:d}'
                                        .format(ship_body.velocity.x, ship_body.velocity.y, ship_body.angular_velocity, t))

                # frequency control to ensure the simulation does not exceed real-world time
                self.rate_dt.sleep()

        finally:
            print('Done simulation\nCleaning up...')
            print('Total KE', total_ke[0])
            print('Total impulse', total_impulse[0])
            print('Total work {}'.format(total_work[0]))
            print('Total distance: ', total_dis)
            plt.ioff()
            plt.close('all')

        return  total_ke[0], total_impulse[0], total_work[0], total_dis

        

    def run_sim(self):

        if self.cfg.concentration not in [0.2, 0.3, 0.4, 0.5]:
            raise Exception("Invalid concentration value. Please check the config file. ") 

        ddict = pickle.load(open('experiments/experiments_' + str(int(self.cfg.concentration * 100)) + '_200_r06_d40x12.pk', 'rb'))
        print("Planner: ", self.cfg.planner, "; Concentration: ", self.cfg.concentration)

        ke_list = []
        impulse_list = []
        work_list = []
        dis_list = []
        cost_list = []
        fig, ax = plt.subplots()
        for trial_idx in range(0, 200):
            exp = ddict['exp'][self.cfg.concentration][trial_idx]  # 0.5 is the concentration, 0 is the trial number

            total_ke, total_impulse, total_work, total_dis = self.sim(
                trial_idx=trial_idx, 
                cfg_file='configs/sim2d_config.yaml',
                # bottom two flags are for planner
                init_queue={
                    **exp
                })
            
            ke_list.append(total_ke)
            impulse_list.append(total_impulse)
            work_list.append(total_work)
            dis_list.append(total_dis)

            np.save(os.path.join(self.cfg.output_dir, self.cfg.planner + "_ke.npy"), np.array(ke_list))
            np.save(os.path.join(self.cfg.output_dir, self.cfg.planner + "_impulse.npy"), np.array(impulse_list))
            np.save(os.path.join(self.cfg.output_dir, self.cfg.planner + "_work.npy"), np.array(work_list))
            np.save(os.path.join(self.cfg.output_dir, self.cfg.planner + "_dis.npy"), np.array(dis_list))



if __name__ == '__main__':
    rclpy.init(args=None)
    cfg_file = 'configs/sim2d_config.yaml'
    cfg = cfg = DotDict.load_from_file(cfg_file)
    node = Sim2DNode(cfg=cfg)

    # Spin in a separate thread NOTE: doing this to unblock rate. Should figure out a better way!
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    try:
        node.run_sim()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
