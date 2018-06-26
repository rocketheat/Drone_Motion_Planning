import argparse
import time
import msgpack
from enum import Enum, auto

import networkx as nx
from queue import PriorityQueue

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from planning_utils import\
a_star,\
graph_a_star,\
heuristic,\
create_grid,\
create_grid_and_edges,\
collinearity_bresenham,\
plot_map,\
plot_map_graph

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection, target, module='grid',
                 pruning=True, plot=False):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

        # Target Position in the shape [latitude, longitude, altitude]
        self.target = target

        # Module used, Search Algorithm, and Pruning methods:
        self.module = module
        self.pruning = pruning
        self.plot = plot

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values

        # the lat0 and lon0 are loacated in line[0] of the colliders file
        with open('./colliders.csv', mode='r') as f:
            lines = f.readlines()
            coordinates = lines[0]
            coordinates = coordinates.replace(',', ' ').split()
            lat0 = float(coordinates[1])
            lan0 = float(coordinates[3])

        # TODO: set home position to (lon0, lat0, 0)

        # setting home position to (lan0, lat0, 0)
        self.set_home_position(lan0, lat0, 0)

        # TODO: retrieve current global position
        current_gloabl_position = self.global_position

        # TODO: convert to current local position using global_to_local()
        current_local_position = global_to_local(current_gloabl_position, self.global_home)

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        def grid_search(data, TARGET_ALTITUDE, SAFETY_DISTANCE):
            """
            This function defines the grid search. It mostly returns path.
            The rest variables are returned for plotting.
            """

            # Define a grid for a particular altitude and safety margin around obstacles
            grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
            print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
            # Define starting point on the grid (this is just grid center)
            grid_start = (-north_offset, -east_offset)

            # TODO: convert start position to current position rather than map center
            grid_start = (int(np.ceil(current_local_position[0]-north_offset)), int(np.ceil(current_local_position[1] - east_offset)))

            # Set goal as some arbitrary position on the grid
            # TODO: adapt to set goal as latitude / longitude position and convert
            goal_north, goal_east, goal_alt = global_to_local(self.target, self.global_home)
            grid_goal = (int(np.ceil(goal_north - north_offset)), int(np.ceil(goal_east - east_offset)))

            # Run A* to find a path from start to goal
            # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
            # or move to a different search space such as a graph (not done here)
            # Diagnoal actions are added in planning_utils

            print('Local Start and Goal: ', grid_start, grid_goal)
            path, _ = a_star(grid, heuristic, grid_start, grid_goal)
            # print('The path is: ', path)

            return path, grid, north_offset, east_offset, grid_start, grid_goal

        # End of grid_search function

        def graph_search(data, TARGET_ALTITUDE, SAFETY_DISTANCE):
            """
            This function defines the graph search. It mostly returns path.
            The rest variables are returned for plotting.
            """
            grid, edges, north_offset, east_offset = create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

            # Define starting point on the grid (this is just grid center)
            grid_start = (-north_offset, -east_offset)

            # TODO: convert start position to current position rather than map center
            grid_start = (int(np.ceil(current_local_position[0]-north_offset)), int(np.ceil(current_local_position[1] - east_offset)))

            # Set goal as some arbitrary position on the grid
            # TODO: adapt to set goal as latitude / longitude position and convert
            goal_north, goal_east, goal_alt = global_to_local(self.target, self.global_home)
            grid_goal = (int(np.ceil(goal_north - north_offset)), int(np.ceil(goal_east - east_offset)))

            # set to the Euclidean distance between the points
            G = nx.Graph()
            for e in edges:
                p1 = e[0]
                p2 = e[1]
                dist = LA.norm(np.array(p2) - np.array(p1))
                G.add_edge(p1, p2, weight=dist)

            def heuristic_graph(n1, n2):
                return LA.norm(np.array(n2) - np.array(n1))

            def closest_point(graph, current_point):
                """
                Compute the closest point in the `graph`
                to the `current_point`.
                """
                closest_point = None
                dist = 100000
                for p in graph.nodes:
                    d = LA.norm(np.array(p) - np.array(current_point))
                    if d < dist:
                        closest_point = p
                        dist = d
                return closest_point

            start_ne_g = closest_point(G, grid_start)
            goal_ne_g = closest_point(G, grid_goal)
            print(start_ne_g)
            print(goal_ne_g)

            path, cost = graph_a_star(G, heuristic_graph, start_ne_g, goal_ne_g)
            path = [(int(np.ceil(p[0])), int(np.ceil(p[1]))) for p in path]


            return path, grid, north_offset, east_offset, grid_start, grid_goal,\
                   start_ne_g, goal_ne_g, edges

        # End of graph_search function

        # TODO: prune path to minimize number of waypoints
        if self.module == 'grid':
            path, grid, north_offset, east_offset, grid_start, grid_goal = grid_search(
                                    data, TARGET_ALTITUDE, SAFETY_DISTANCE)

            if self.plot:
                plot_map(grid=grid, start_ne=grid_start, goal_ne=grid_goal, path=path)

        elif self.module == 'graph':
            path, grid, north_offset, east_offset, grid_start, grid_goal, \
                            start_ne_g, goal_ne_g, edges = graph_search( \
                            data, TARGET_ALTITUDE, SAFETY_DISTANCE)

            if self.plot:
                plot_map_graph(grid=grid, edges=edges, grid_start=grid_start,
                               grid_goal=grid_goal, start_ne_g=start_ne_g,
                               goal_ne_g=goal_ne_g, path=path)

        # I pruned twice to double check all extra nodes are pruned.
        if self.pruning:
            path = collinearity_bresenham(grid=grid, path=path, epsilon=1e-6)

        if self.pruning:
            path = collinearity_bresenham(grid=grid, path=path, epsilon=1e-6)


        # TODO (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


# Testing examples
target_examples = [[-122.39292549, 37.7902035, -0.147],
                   [-122.40195876, 37.79673913, -0.147],
                   [-122.40199327, 37.79245808, -0.147]]


class Fly(object):
    """
    I created this class to simplify the interaction with the drone.
    My goal here is to allow possible combinations of situations for example
    test graph with breadth first search and combination of pruning methods

    Parameters:
    target_examples: three options 1, 2, 3
    module: graph or grid
    pruning: True or False
    plot: True or False
    """
    def __init__(self, conn, target = target_examples[1], module = 'grid',
                 pruning = True, plot = True):

        self.conn = conn
        self.target = target
        self.module = module
        self.pruning = pruning
        self.plot = plot

    def get_plan(self):
        self.drone = MotionPlanning(connection=self.conn, target=self.target,
                                    module=self.module, pruning=self.pruning, plot=self.plot)

    def fly(self):
        self.get_plan()
        self.drone.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = Fly(conn)
    time.sleep(1)

    drone.fly()
