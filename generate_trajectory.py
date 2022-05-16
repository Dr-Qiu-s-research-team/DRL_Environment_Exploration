from numpy import *
from scipy.sparse.linalg import expm
from scipy.linalg import norm
import pdb
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
import numpy as np
import random
import torch
import csv

import pdb

def traj_mwpts(t, b, vi, ai, zi):
    # Optimal trajectory passing thru multiple waypoints
    # Continuous dynamics: \dot{x} = Ax + Bu; y = Cx;
    m = 4.34
    A = np.array(np.vstack((np.hstack((np.zeros((3,3)), np.eye(3), np.zeros((3,3)), np.zeros((3,3)))),
                            np.hstack((np.zeros((3,3)), np.zeros((3,3)), np.multiply(np.divide(-1., m), np.eye(3)), np.zeros((3,3)))),
                            np.hstack((np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3))),
                            np.hstack((np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)))))))
    B = np.array(np.hstack((np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3)))).T
    C = np.array(np.hstack((np.eye(3), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)))))

    # Discrete dynamics: x_{i+1} = Adx_{i} + Bdu_{i}; y_{i} = Cdu_{i};
    h = 0.1
    Ad = expm(A*h)
    Bd = np.array(np.vstack((np.hstack((h*np.eye(3), ((pow(h,2))/2)*np.eye(3), ((pow(h,3))/6)*np.eye(3), ((pow(h,4))/24)*np.eye(3))),
                            np.hstack((np.zeros((3, 3)), h*np.eye(3), ((pow(h,2))/2)*np.eye(3), ((pow(h,3))/6)*np.eye(3))),
                            np.hstack((np.zeros((3, 3)), np.zeros((3, 3)),  h*np.eye(3), ((pow(h,2))/2)*np.eye(3))),
                            np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), h*np.eye(3))))))
    Bd = Bd.dot(B)
    Cd = C.copy()

    # Physical parameters
    g = 9.8 # Acceleration due to gravity
    e3 = array([[0],[0],[1]])

    # Initial state
    xi = np.vstack((np.expand_dims(b[:,0], axis=1), vi, ai, zi))

    # Gain Matrices
    R_bar = 25 * eye(3)
    Q_bar = 0.01 * eye(12)
    S = 35 * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 10]])

    Pf = Cd.T.dot(S).dot(C)/h
    etaf =  -1*Cd.T.dot(S).dot(np.expand_dims(b[:,-1], axis=1))/h

    N = int(np.round((t[-1] - t[0])/h))
    nm = zeros(len(t), dtype=np.int)
    for i in range(len(t)-1):
        nm[i+1] = nm[i] + int(np.round((t[i+1]-t[i])/h))

    P = np.zeros((12, 12, N+1))
    P[:, :, N] = Pf
    eta = np.zeros((12, N+1))
    eta[:, N] = etaf.squeeze()
    k1 = 10

    R = np.zeros((3, 3, N))
    Q = np.zeros((12, 12, N))
    for j in range(len(nm)-1, 0, -1):
        alfa = np.zeros((1, nm[j] - nm[j-1] - 1))
        for k in range(int(nm[j] - nm[j-1] - 1)):
            alfa[0][k] = ((k + 1) * h) / (t[j] - t[j - 1])

        for i in range(nm[j]-1, nm[j-1], -1):
            R[:,:,i] = (k1-alfa[0][i-1-nm[j-1]]) * R_bar
            Q[:,:,i] = alfa[0][i-1-nm[j-1]] * Q_bar
            P[:,:,i] = Q[:,:,i]+Ad.T.dot(P[:,:,i+1]).dot(Ad)-Ad.T.dot(P[:,:,i+1]).dot(Bd).dot(np.linalg.inv(R[:,:,i]+Bd.T.dot(P[:,:,i+1]).dot(Bd))).dot(Bd.T).dot(P[:,:,i+1]).dot(Ad)
            eta[:,i] = Ad.T.dot(np.eye(12)-P[:,:,i+1].dot(Bd).dot(np.linalg.inv(R[:,:,i]+Bd.T.dot(P[:,:,i+1]).dot(Bd))).dot(Bd.T)).dot(eta[:,i+1])

        k = nm[j-1]
        alfa = 1
        R[:,:,k] = (k1-alfa)*R_bar
        Q[:,:,k] = alfa*Q_bar
        P[:,:,k] = Q[:,:,k]+Ad.T.dot(P[:,:,k+1]).dot(Ad)-Ad.T.dot(P[:,:,k+1]).dot(Bd).dot(np.linalg.inv(R[:,:,k]+Bd.T.dot(P[:,:,k+1]).dot(Bd))).dot(Bd.T).dot(P[:,:,k+1]).dot(Ad)+Cd.T.dot(S).dot(Cd)/h
        eta[:, k] = Ad.T.dot(np.eye(12)-P[:,:,k+1].dot(Bd).dot(np.linalg.inv(R[:,:,k] + Bd.T.dot(P[:,:,k+1]).dot(Bd))).dot(Bd.T)).dot(eta[:,k+1]) - (Cd.T.dot(S).dot(np.expand_dims(b[:,j-1], axis=1)/h)).squeeze()

    # Determining state and control
    xt = np.zeros((12, N))
    xt[:,0] = xi.squeeze()
    tme = np.zeros((1,N))
    tme[:,0] = t[0]
    thrst = np.zeros((3, N))
    thrst[:,0] = (m * (g * e3 + np.expand_dims(xt[6:9, 0], axis=1))).squeeze()
    nthr = np.zeros((1, N))
    nthr[:,0] = norm(thrst[:, 0])
    spd = np.zeros((1, N))
    spd[:, 0] = norm(xt[3:6, 0])

    ut = np.zeros((3, N-1))
    yt = np.zeros((3, N))
    for j in range(N-1):
        tme[:,j+1] = t[0]+(j+1)*h
        ut[:,j] =  -1*np.linalg.inv(R[:,:,j]+Bd.T.dot(P[:,:,j+1]).dot(Bd)).dot(Bd.T).dot(P[:,:,j+1].dot(Ad).dot(xt[:,j])+eta[:,j+1])
        xt[:,j+1] = Ad.dot(xt[:,j])+Bd.dot(ut[:,j])
        yt[:,j+1] = Cd.dot(xt[:,j+1])
        thrst[:,j+1] = (m * (g * e3 + np.expand_dims(xt[6:9, j+1], axis=1))).squeeze()
        nthr[:,j+1] = norm(thrst[:,j+1])
        spd[:,j+1] = norm(xt[3:6, 0])

    lamb = np.zeros((12, N))
    for j in range(N):
        lamb[:, j] = P[:,:,j].dot(xt[:,j])+eta[:,j]

    return [xt[0:3, :], thrst, nthr, xt[3:6, -1], xt[6:9, -1], xt[9:12, -1]]


def traj_mwpts_test():
    # Test Part
    b = [[0],[0],[0]]
    b = hstack((b,[[0],[1],[1]]))
    b = hstack((b,[[0],[2],[1]]))
    # b = hstack((b,[[1],[1],[2]]))
    # b = hstack((b,[[2],[2],[3]]))
    # b = hstack((b,[[2],[1],[4]]))
    # b = hstack((b,[[1],[2],[5]]))


    t = array([0, 6,12])
    vi = array([[0],[0],[0.0025]])
    ai = array([[0],[0],[0]])
    zi = array([[0],[0],[0]])

    x_final, f, nf, v_final, a_final, z_final, v_total = traj_mwpts(t, b, vi, ai, zi)
    plot_x = x_final[0]
    plot_y = x_final[1]
    plot_z = x_final[2]
    print(v_final)
    for i in range(v_total.T.shape[0]):
        with open('./velocity.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([v_total.T[i][0], v_total.T[i][1], v_total.T[i][2]])
    pdb.set_trace()

    # path = np.asarray([[0], [0], [0]])
    # for idx in range(b.shape[1] - 1):
    #     waypoints = b[:,idx:idx+2]
    #     t = array([idx * 6, (idx + 1) * 6])
    #     x_final, f, nf, v_final, a_final, z_final, v_total = traj_mwpts(t, waypoints, vi, ai, zi)
    #     vi = np.array([v_final]).T 
    #     ai = np.array([a_final]).T 
    #     zi = np.array([z_final]).T 
    #     path = concatenate((path, x_final), axis=1)
    #     for i in range(v_total.T.shape[0]):
    #         with open('./velocity.csv', 'a', newline='') as csvfile:
    #             writer = csv.writer(csvfile, delimiter=',',
    #                                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #             writer.writerow([v_total.T[i][0], v_total.T[i][1], v_total.T[i][2]])
    #     print(v_final)
    # pdb.set_trace()
    # plot_x = path[0,1:]
    # plot_y = path[1,1:]
    # plot_z = path[2,1:]

    # def animate(num):
    #     graph.set_data (plot_x[num:num+1], plot_y[num:num+1])
    #     graph.set_3d_properties(plot_z[num:num+1])
    #     title.set_text('Time: {}, Data: {:.2f} {:.2f} {:.2f}'.format(num, plot_x[num], plot_y[num], plot_z[num]))
    #     return title, graph,

    fig = pyplot.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 5)
    # title = ax.set_title('Time: ')
    ax.plot(plot_x, plot_y, plot_z, linestyle="-")
    # pdb.set_trace()

    for i in range(b.T.shape[0]):
        ax.scatter(b.T[i][0],b.T[i][1],b.T[i][2] , s=20*4)

    # graph, = ax.plot(plot_x[:1], plot_y[:1], plot_z[:1], linestyle="", marker="o")
    # ani = FuncAnimation(fig, animate, frames=600, interval=10, blit=True)
    pyplot.show()

if __name__ == "__main__":
    traj_mwpts_test()
