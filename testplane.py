#!/usr/bin/python  -u

from cell import *
from animate import *
import initialconfig
# import cProfile
import matplotlib.pyplot as plt
npr.seed(seed=0)

#######################################################

if __name__ == '__main__':

    Lmax = 5
    Lsubs = 10
    N = None
    Nsubs = None

    bend = 10.0
    twist = 1.0
    dt = 0.01
    nmax = 300
    qmin = 0.001
    dims = 3

    d0min = 0.8  # min distance between cells
    d0max = 2.  # max distance connected by links
    d0_0 = 1.  # equilibrium distance of links
    p_add = 1.  # rate to add links
    p_add_subs = 1.
    p_del = 0.1  # base rate to delete links
    p_del_subs = 1.
    chkx = False  # check if links overlap?
    """
    # substrate

    config = initialconfig.square_wsubs(L=Lmax, Lsubs=Lsubs, N=N, Nsubs=Nsubs, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0,
                                        p_add=p_add, p_add_subs=p_add_subs, p_del=p_del, p_del_subs=p_del_subs,
                                        chkx=chkx, d0max=d0max, d0min=d0min, dims=dims)

    config.mynodes.updateDists(config.mynodes.nodesX)
    config.mysubs.updateDists(config.mynodes.nodesX)

    allnodes = np.concatenate((config.mynodes.nodesX, config.mysubs.nodesX), axis=0)

    for i, j in VoronoiNeighbors(allnodes, vodims=3):
        if np.linalg.norm(allnodes[i] - allnodes[j]) <= d0max:
            if (i < config.N) and (j < config.N):
                config.mynodes.addlink(i, j)
            elif (i >= config.N) and (j >= config.N):
                continue
            else:
                config.mysubs.addlink(i, j - config.N, config.mynodes.nodesPhi[i])
    
    configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces = \
        config.timeevo(50., record=True)

    config.savedata()
    animateconfigs(configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces, showsubs=False)
    mlab.show()
    """
    """
    # bilayer

    config = initialconfig.bilayer(L=Lmax, N=N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0,
                                      p_add=p_add, p_del=p_del, chkx=chkx, d0min=d0min, d0max=d0max, dims=dims)

    config.mynodes.updateDists(config.mynodes.nodesX)

    for i, j in VoronoiNeighbors(config.mynodes.nodesX):
        if np.linalg.norm(config.mynodes.nodesX[i] - config.mynodes.nodesX[j]) <= d0max:
            config.mynodes.addlink(i, j)

    # cProfile.run('config.oneequil()', sort='cumtime')



    configs, links, nodeforces, linkforces, ts = config.timeevo(40., record=True)
    # config.savedata()
    animateconfigs(configs, links, nodeforces, linkforces, ts)
    mlab.show()
    """

    """
    # double rod
    
    R = []
    for i in range(12):
        r = (0.1 - 0.2 * npr.random())
        R.append([i, r, 0])
        r = (0.1 - 0.2 * npr.random())
        R.append([i + 0.5, r + 0.5, 0])
    R = np.array(R)

    config = initialconfig.fromdefault(R, L=Lmax, N=N, dt=dt, nmax=nmax, qmin=qmin,
                                    d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max, dims=dims)

    config.mynodes.updateDists(config.mynodes.nodesX)
    
    for i, j in VoronoiNeighbors(config.mynodes.nodesX):
        if np.linalg.norm(config.mynodes.nodesX[i] - config.mynodes.nodesX[j]) <= d0max:
            config.mynodes.addlink(i, j)

    # cProfile.run("config.timeevo(40.)", sort="cumtime")

    configs, links, nodeforces, linkforces, ts = config.timeevo(40., record=True)
    # config.savedata()
    animateconfigs(configs, links, nodeforces, linkforces, ts)
    mlab.show()
    """

    #2D square
    config = initialconfig.square(L=Lmax, N=N, dt=dt, nmax=nmax, qmin=qmin,
                                  d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max, d0min=d0min, dims=dims)

    config.mynodes.updateDists(config.mynodes.nodesX)

    for i, j in VoronoiNeighbors(config.mynodes.nodesX):
        if np.linalg.norm(config.mynodes.nodesX[i] - config.mynodes.nodesX[j]) <= d0max:
            config.mynodes.addlink(i, j)

    configs, links, nodeforces, linkforces, ts = config.timeevo(10., record=True)
    config.savedata()
    animateconfigs(configs, links, nodeforces, linkforces, ts)
    mlab.show()


