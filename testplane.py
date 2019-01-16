#!/usr/bin/python  -u

from cell import *
from animate import *
import cProfile
import matplotlib.pyplot as plt
npr.seed(seed=0)

#######################################################


def generatePoint(L):
    X0 = (npr.rand() - .5) * L
    Y0 = (npr.rand() - .5) * L
    Z0 = 0.
    return np.array([X0, Y0, Z0])


def rand3d(L):
    return (np.random.random(3) - 0.5) * L


def generate_initial_random(L, N, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0max, dims):
    if N is None:
        N = int(L ** 2)

    c = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max,
                      dims=dims, issubs=False)

    for ni in range(N):
        while True:
            R1 = generatePoint(L)
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(c.mynodes.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        c.mynodes.nodesX[ni] = R1
    return c


def generate_initial_random_wsubs(L, N, Nsubs, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0max, dims):
    if N is None:
        N = int(L ** 2)
    if Nsubs is None:
        Nsubs = N

    c = CellMech(N, num_subs=Nsubs, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max,
                      dims=dims, issubs=True)

    for ni in range(N):
        while True:
            R1 = generatePoint(L)
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(c.mynodes.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        c.mynodes.nodesX[ni] = R1

    for ni in range(N):
        while True:
            R1 = generatePoint(L)
            R1[2] = -d0_0
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(c.mysubs.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        c.mysubs.nodesX[ni] = R1

    return c


def generate_initial_bilayer(L, N, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0min, d0max, dims):
    if N is None:
        N = int(2 * (L ** 2))

    c = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max,
                      dims=dims, issubs=False)

    for ni in range(N):
        while True:
            R1 = generatePoint(L)
            if ni >= N / 2:
                R1[2] = d0_0
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(c.mynodes.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        c.mynodes.nodesX[ni] = R1
    return c


def generate_default_initial(L=10, N=None):
    if N is None:
        N = int(L ** 2)
    R = []
    for ni in range(N):
        while True:
            R1 = generatePoint(L)
            OK = True
            for r in R:
                d = np.linalg.norm(R1 - r)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        R.append(R1)
    R = np.array(R)
    np.save("Rinit", R)

    return R


def generate_from_default(R, L, N, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0max, dims):
    N = len(R)
    c = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max,
                      dims=dims, issubs=False)
    for ni in range(N):
        c.mynodes.nodesX[ni] = R[ni]

    return c


def generate_initial_cube(L, N, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0max, dims, stretch=1.):
    if N is not None:
        print "N decission overriden because of possible conflict with Lmax"
    N = int(L ** 3)

    c = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max,
                      dims=dims, issubs=False)

    x0 = - stretch * L / 2.
    for ni in range(L):
        for nj in range(L):
            for nk in range(L):
                c.mynodes.nodesX[L * L * ni + L * nj + nk] = \
                    np.array([x0 + stretch * ni, x0 + stretch * nj, x0 + stretch * nk])
    return c

if __name__ == '__main__':

    Lmax = 5
    N = None

    bend = 10.0
    twist = 1.0
    dt = 0.01
    nmax = 300
    qmin = 0.001
    dims = 3
#
    d0min = 0.8  # min distance between cells
    d0max = 2.  # max distance connected by links
    d0_0 = 1.  # equilibrium distance of links
    p_add = 1.0  # rate to add links
    p_del = 0.1  # base rate to delete links
    chkx = False  # check if links overlap?


    # substrate

    config = generate_initial_random_wsubs(L=Lmax, N=N, Nsubs=N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0,
                                          p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max, dims=dims)

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
    animateconfigs(configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces)
    mlab.show()

    """
    # bilayer

    config = generate_initial_bilayer(L=Lmax, N=N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0,
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
    
    R = [[i, 0, 0] for i in range(13)]
    for i in range(12):
        R.append([i + 0.5, 0.5, 0])
    R = np.array(R)

    config = generate_config_from_default(R, L=Lmax, N=N, dt=dt, nmax=nmax, qmin=qmin,
                                          d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max, dims=dims)
                                              config.mynodes.updateDists(config.mynodes.nodesX)
    
    for i, j in VoronoiNeighbors(config.mynodes.nodesX):
        if np.linalg.norm(config.mynodes.nodesX[i] - config.mynodes.nodesX[j]) <= d0max:
            config.mynodes.addlink(i, j)
                
    configs, links, nodeforces, linkforces, ts = config.timeevo(40., record=True)
    config.savedata()
    animateconfigs(configs, links, nodeforces, linkforces, ts)
    mlab.show()
    """