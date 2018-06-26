from enum import Enum
from queue import PriorityQueue
import numpy as np
from scipy.spatial import Voronoi
from bresenham import bresenham
import matplotlib.pyplot as plt


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Initialize an empty list for Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # location of obstacle centres
    graph = Voronoi(points)

    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]

        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        hit = False

        for c in cells:
            # First check if we're off the map
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                hit = True
                break
            # Next check if we're in collision
            if grid[c[0], c[1]] == 1:
                hit = True
                break

        # If the edge does not hit on obstacle
        # add it to the list
        if not hit:
            # array to tuple for future graph creation step)
            p1 = (p1[0], p1[1])
            p2 = (p2[0], p2[1])
            edges.append((p1, p2))

    return grid, edges, int(north_min), int(east_min)


# Assume all actions cost the same.

class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    # diagonal Actions

    SOUTHEAST = (1, 1, np.sqrt(2))
    EASTNORTH = (1, -1, np.sqrt(2))
    NORTHWEST = (-1, -1, np.sqrt(2))
    WESTSOUTH = (-1, 1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)

    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)

    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)

    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)


    # Diagnoal Valid Actions. Basically constructed based on combinations
    # of the above
    if (x + 1 > n or y + 1 > m or grid[x + 1, y] == 1 or grid[x, y + 1] == 1):
        valid_actions.remove(Action.SOUTHEAST)

    if (y + 1 > m or x - 1 < 0 or grid[x, y + 1] == 1 or grid[x - 1, y] == 1):
        valid_actions.remove(Action.EASTNORTH)

    if (x - 1 < 0 or y - 1 < 0 or grid[x - 1, y] == 1 or grid[x, y - 1] == 1):
        valid_actions.remove(Action.NORTHWEST)

    if (y - 1 < 0 or x + 1 > n or grid[x, y - 1] == 1 or grid[x + 1, y] == 1):
        valid_actions.remove(Action.WESTSOUTH)

    return valid_actions


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


def graph_a_star(graph, h, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                branch_cost = current_cost + cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node)
                    queue.put((queue_cost, next_node))


    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost



def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def collinearity_bresenham(grid, path, epsilon=1e-6):
    """ Assess collinearity through path points """

    def point(p):
        """ converting points into a square matrix """
        return np.array([p[0], p[1], 1.])

    def collinearity_test(p1, p2, p3, epsilon=epsilon):
        """ Calculating the determinant and checking if < epsilon """
        collinear = False
        mat = np.vstack((point(p1), point(p2), point(p3)))
        det = np.linalg.det(mat)
        if det < epsilon:
            collinear = True

        return collinear

    def find_in_grid(point):
        """ Used by bresenham to find if the grid cell is an obstacle """
        if grid[point[0], point[1]] == 1:
            return 1
        else:
            return 0

    def prune_bresenham(p1, p3):
        """ Uses bresenham algorithm to prune the path """
        error_margin = 1 # represent only one grid is overlapped with an obstacle.
        prune = True
        cells = list(bresenham(p1[0], p1[1], p3[0], p3[1]))
        ingrid = np.array(list(map(find_in_grid, cells)), dtype=np.float64)
        if np.sum(ingrid) > error_margin:
            prune = False
        return prune

    num =0
    while num < len(path) -2:
        # I had some difficulty using a for loop so I ended up just using a
        # a while loop here
        if (collinearity_test(p1=path[num], p2=path[num+1], p3=path[num+2],\
            epsilon=epsilon) and prune_bresenham(p1=path[num], p3=path[num+2])) or\
            prune_bresenham(p1=path[num], p3=path[num+2]):
            path.pop(num+1)
            # we need to subtract one here because if we prune point 2 in a
            # a list [1,2,3], on the next iteration we will miss point 3 as
            # a being a possible point to prune withouth the num -= 1 below.
            num -= 1

        num += 1

    return path


def plot_map(grid, start_ne, goal_ne, path):
    """ Used to plot grids """

    plt.imshow(grid, cmap='Greys', origin='lower')

    # For the purposes of the visual the east coordinate lay along
    # the x-axis and the north coordinates long the y-axis.
    plt.plot(start_ne[1], start_ne[0], 'x')
    plt.plot(goal_ne[1], goal_ne[0], 'x')

    pp = np.array(path)
    plt.plot(pp[:, 1], pp[:, 0], 'g')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.show()


def plot_map_graph(grid, edges, grid_start, grid_goal, start_ne_g, goal_ne_g, path):
    """ Used to plot graphs """

    plt.imshow(grid, origin='lower', cmap='Greys')

    for e in edges:
        p1 = e[0]
        p2 = e[1]
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')

    plt.plot([grid_start[1], start_ne_g[1]], [grid_start[0], start_ne_g[0]], 'r-')
    for i in range(len(path)-1):
        p1 = path[i]
        p2 = path[i+1]
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-')
    plt.plot([grid_goal[1], goal_ne_g[1]], [grid_goal[0], goal_ne_g[0]], 'r-')

    plt.plot(grid_start[1], grid_start[0], 'gx')
    plt.plot(grid_goal[1], grid_goal[0], 'gx')

    plt.xlabel('EAST', fontsize=20)
    plt.ylabel('NORTH', fontsize=20)
    plt.show()
