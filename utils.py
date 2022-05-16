import os,sys,random
import numpy as np
import datetime
from path import *
import pdb

'''
Five kinds of obstacles:
    obs0 :      #  (1, 1, 10)
    obs1 :      #  (3, 1, 3) # 1
    obs2 :      #  (3, 1, 3) # 2
    obs3 :      #  (1, 1, 1), (1, 1, 1), (3, 1, 1) # 1
    obs4 :      #  (1, 1, 1), (1, 1, 1), (3, 1, 1) # 2
'''

def ifObsValid(env_bits,index, position):
    try:
        if index == 0:
            if env_bits[position[0], position[1], 0] == 0 and env_bits[position[0], position[1], 1] == 0:
                for i in range(10):
                    env_bits[position[0], position[1], i] = 1
                return True
        elif index == 1 or index == 3:
            if env_bits[position[0], position[1], 0] == 0 and \
                    env_bits[position[0] + 1, position[1], 0] == 0 and \
                    env_bits[position[0] + 2, position[1], 0] == 0 and \
                    env_bits[position[0], position[1], 1] == 0 and \
                    env_bits[position[0] + 1, position[1], 1] == 0 and \
                    env_bits[position[0] + 2, position[1], 1] == 0:

                if index == 1:
                    for i in range(3):
                        env_bits[position[0], position[1], i] = 1
                        env_bits[position[0] + 1, position[1], i] = 1
                        env_bits[position[0] + 2, position[1], i] = 1

                else:
                    env_bits[position[0], position[1], 0] = 1
                    env_bits[position[0] + 2, position[1], 0] = 1
                    env_bits[position[0]: position[0] + 3, position[1], 1] = 1

                return True

        elif index == 2 or index == 4:
            if env_bits[position[0], position[1], 0] == 0 and \
                    env_bits[position[0], position[1] + 1, 0] == 0 and \
                    env_bits[position[0], position[1] + 2, 0] == 0 and \
                    env_bits[position[0], position[1], 1] == 0 and \
                    env_bits[position[0], position[1] + 1, 1] == 0 and \
                    env_bits[position[0], position[1] + 2, 1] == 0:
                if index == 2:
                    for i in range(3):
                        env_bits[position[0], position[1], i] = 1
                        env_bits[position[0], position[1] + 1, i] = 1
                        env_bits[position[0], position[1] + 2, i] = 1
                else:
                    env_bits[position[0], position[1], 0] = 1
                    env_bits[position[0], position[1] + 2, 0] = 1
                    env_bits[position[0], position[1]: position[1] + 3, 1] = 1

                return True
        return False
    except:
        return False

def setUavLocation(env_bits):
    random.seed(datetime.datetime.now())
    x = random.randint(0, 9)
    y = random.randint(0, 9)
    z = random.randint(0, 9)
    while env_bits[x][y][z] == 1:
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        z = random.randint(0, 9)
    return x,y,z

def create_objs(obs_num=10):
    env_bits = np.zeros((10, 10, 10))
    obs_list=[]
    info=[]
    info.append('the number of obs:%s'%str(obs_num)+'\n')

    # obs_number = int(obs_num)
    # print('The number of obstacles: ' + str(obs_number))
    tmp=[]
    pillar_num = random.randint(1, 5)
    assert pillar_num <= obs_num
    for i in range(pillar_num):
        position = (random.randint(0, 9), random.randint(0, 9))
        random.seed(datetime.datetime.now())
        while not ifObsValid(env_bits, 0, position) :
            position = (random.randint(0, 9), random.randint(0, 9))
        info.append('obstacle # : ' + str(0) + ' position :' + str(position)+'\n')
        # add into env
        tmp=[0] + list(position) + [0]
        obs_list.append(tmp)

    other_num = obs_num - pillar_num
    for i in range(int(other_num)):
        obs_id = random.randint(1, 4)   # get the shape of obstacle randomly
        position = (random.randint(0, 9), random.randint(0, 9))
        random.seed(datetime.datetime.now())
        while not ifObsValid(env_bits, obs_id, position) :
            obs_id = random.randint(1, 4)
            position = (random.randint(0, 9), random.randint(0, 9))
        info.append('obstacle # : ' + str(obs_id) + ' position :' + str(position)+'\n')
        # add into env
        tmp=[obs_id] + list(position) + [0]
        obs_list.append(tmp)

    groundtruth = []
    grids_pos = np.where(env_bits == 1)
    for idx in range(grids_pos[0].shape[0]):
        temp = np.asarray([grid[idx] for grid in grids_pos])
        groundtruth.append(temp.astype(np.float32))
    groundtruth = np.array(groundtruth)
    return groundtruth

def plot_obstacles(fig, env):
    ax = fig.gca(projection='3d')
    obs_list = env.objs_info['obst_list']
    for temp_obst in obs_list:
        _plot_cube(fig, int(temp_obst[0]), int(temp_obst[1]), int(temp_obst[2]))

def _plot_cube(fig, cur_x, cur_y, cur_z):
    ax = fig.gca(projection='3d')
    N = 11
    # right
    x = (cur_x + 1) * np.ones((N,N))
    y = np.arange(cur_y, cur_y + 1 + 0.1, 0.1)
    z = np.arange(cur_z, cur_z + 1 + 0.1, 0.1)
    y, z = np.meshgrid(y, z)
    ax.plot_surface(x, y, z, color='b')
    # left
    x = cur_x * np.ones((N,N))
    y = np.arange(cur_y, cur_y + 1 + 0.1, 0.1)
    z = np.arange(cur_z, cur_z + 1 + 0.1, 0.1)
    y, z = np.meshgrid(y, z)
    ax.plot_surface(x, y, z, color='b')
    # back
    y = (cur_y + 1) * np.ones((N,N))
    x = np.arange(cur_x, cur_x + 1 + 0.1, 0.1)
    z = np.arange(cur_z, cur_z + 1 + 0.1, 0.1)
    x, z = np.meshgrid(x, z)
    ax.plot_surface(x, y, z, color='b')
    # front
    y = cur_y * np.ones((N,N))
    x = np.arange(cur_x, cur_x + 1 + 0.1, 0.1)
    z = np.arange(cur_z, cur_z + 1 + 0.1, 0.1)
    x, z = np.meshgrid(x, z)
    ax.plot_surface(x, y, z, color='b')
    # up
    z = (cur_z + 1) * np.ones((N,N))
    y = np.arange(cur_y, cur_y + 1 + 0.1, 0.1)
    x = np.arange(cur_x, cur_x + 1 + 0.1, 0.1)
    y, x = np.meshgrid(y, x)
    ax.plot_surface(x, y, z, color='b')
    # down
    z = cur_z * np.ones((N,N))
    y = np.arange(cur_y, cur_y + 1 + 0.1, 0.1)
    x = np.arange(cur_x, cur_x + 1 + 0.1, 0.1)
    y, x = np.meshgrid(y, x)
    ax.plot_surface(x, y, z, color='b')

if __name__ == '__main__':
    create_environment(10)
