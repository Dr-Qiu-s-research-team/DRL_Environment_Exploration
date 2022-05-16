import time
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from generate_trajectory import traj_mwpts

def plot_env(env, waypoints):
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlabel('Z', color='k')
    ax.set_ylabel('Y', color='k')
    ax.set_xlabel('X', color='k')
    ax.set_xlim(0, 10)
    ax.set_xticks(np.arange(0,10,1))
    ax.set_ylim(0, 10)
    ax.set_yticks(np.arange(0,10,1))
    ax.set_zlim(0, 10)
    #ax.set_title("")
    ax.legend(loc='lower right')
    plot_obstacles(fig, env)
    plot_x, plot_y, plot_z = trajectory(waypoints)
    waypoints_t = np.transpose(np.array(waypoints))+0.5
    ax.plot(waypoints_t[0], waypoints_t[1], waypoints_t[2], 'o')
    ax.plot([waypoints_t[0][0]], [waypoints_t[1][0]], [waypoints_t[2][0]], 'o',color='g')
    ax.plot([waypoints_t[0][-1]], [waypoints_t[1][-1]], [waypoints_t[2][-1]], 'o',color='r')
    ax.plot(plot_x+0.5, plot_y+0.5, plot_z+0.5, 'r-',linewidth=2)

    ax.view_init(40, -130)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    pyplot.savefig('./%s.png'%time_str)
    pyplot.grid()
    pyplot.show()

def trajectory(points):
    vi = np.array((0, 0, 0.001))
    ai = np.array((0, 0, 0))
    zi = np.array((0, 0, 0))
    trajectory = []
    t = [0]
    num = len(points)
    for i in range(num - 1):
        t = np.hstack((t, 6 * (i + 1)))
    trajectory, f, norm_f, velocity_next, acceler_next, gerk_next = traj_mwpts(t, np.asarray(points).T, np.array([vi]).T, np.array([ai]).T, np.array([zi]).T)
    plot_x = trajectory[0]
    plot_y = trajectory[1]
    plot_z = trajectory[2]
    return plot_x, plot_y, plot_z

def plot_obstacles(fig, env):
    ax = fig.gca(projection='3d')
    obs_list = env.objs_info['obst_list']
    for temp_obst in obs_list:
        plot_cube(fig, int(temp_obst[0]), int(temp_obst[1]), int(temp_obst[2]))

def plot_cube(fig, cur_x, cur_y, cur_z):
    ax = fig.gca(projection='3d')
    N = 11
    #left and right
    for i in range(2):
        x = (cur_x + i) * np.ones((N,N))
        y = np.arange(cur_y, cur_y + 1 + 0.1, 0.1)
        z = np.arange(cur_z, cur_z + 1 + 0.1, 0.1)
        y, z = np.meshgrid(y, z)
        ax.plot_surface(x, y, z, color='b')
    #front and back
    for i in range(2):
        x = np.arange(cur_x, cur_x + 1 + 0.1, 0.1)
        y = (cur_y + i) * np.ones((N,N))
        z = np.arange(cur_z, cur_z + 1 + 0.1, 0.1)
        x, z = np.meshgrid(x, z)
        ax.plot_surface(x, y, z, color='b')
    #up and down
    for i in range(2):
        x = np.arange(cur_x, cur_x + 1 + 0.1, 0.1)
        y = np.arange(cur_y, cur_y + 1 + 0.1, 0.1)
        z = (cur_z + i) * np.ones((N,N))
        y, x = np.meshgrid(y, x)
        ax.plot_surface(x, y, z, color='b')
