from cell import *
from animate import *

npr.seed(seed=0)


def generatePoint(L):
    """
    Produce random 3-dimensional coordinate on x-y-plane confined to square
    :param L: float, length of side of confining square
    :return: numpy array of shape (3,)
    """
    X0 = (npr.rand() - .5) * L
    Y0 = (npr.rand() - .5) * L
    Z0 = 0.
    return np.array([X0, Y0, Z0])


if __name__ == '__main__':

    # run simulation of cells in 2d initialized in square with interruption and relaunch after half-time

    ####################

    Lmax = 5                # Length of confining square
    N = None                # Number of cells. If None: int(Lmax**2)
    runtime = 100.          # Length of simulation run
    dims = 2                # Number of dimensions for the given problem

    dtrec = 0.              # Periodicity of making configuration snapshots (done after every plasticity step if 0)
    savedata = True         # Whether to write simulation results to file
    savedir = "res"         # Directory to save the simulation results
    dtsave = 10.            # Periodicity of writing data to hard drive (done only after end of runtime if None)

    dt = 0.01               # fundamental time unit, relevant only in combination with nmax
    nmax = 1000             # dt * nmax is the maximum time for mechanical equilibration
    qmin = 0.001            # Threshhold tension beneath which the system is in mechanical equilibrium

    d0min = 0.8             # min distance between cells when initialized
    d0max = 2.              # max distance connected by links
    d0_0 = 1.               # equilibrium distance of links (fundamental scaling of space)
    p_add = 1.              # rate to add cell-cell links
    p_del = 0.2             # rate to delete cell-cell links

    chkx = False            # check whether links overlap

    ####################

    if N is None:
        N = int(Lmax ** 2)

    config = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del,
                      chkx=chkx, d0max=d0max, dims=dims, issubs=False)

    # initialize random positions for cells in bilayer
    for ni in range(N):
        while True:
            R1 = generatePoint(Lmax)
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(config.mynodes.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        config.mynodes.nodesX[ni] = R1

    # add links between cells adjacent in Voronoi tesselation and closer than d0min
    for i, j in VoronoiNeighbors(config.mynodes.nodesX):
        if np.linalg.norm(config.mynodes.nodesX[i] - config.mynodes.nodesX[j]) <= d0max:
            config.mynodes.addlink(i, j)

    # run and save simulation
    simdata = config.timeevo(runtime/2., dtrec=dtrec, savedata=savedata, savedir=savedir, dtsave=dtsave)

    config2 = relaunch_CellMech(savedir, N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del,
                      chkx=chkx, d0max=d0max, dims=dims, issubs=False)

    simdata2 = config2.timeevo(runtime/2., isinit=False, dtrec=dtrec, savedata=savedata, savedir=savedir, dtsave=dtsave)

    # animate results
    animateconfigs(simdata2)
    mlab.show()
