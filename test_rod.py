from cell import *
from animate import *

npr.seed(seed=0)


if __name__ == '__main__':

    # run simulation in 2d of cells initalized in the shape of double-rod

    ####################

    runtime = 100.              # Length of simulation run
    dims = 2                    # Number of dimensions for the given problem

    dtrec = 0.              # Periodicity of making configuration snapshots (done after every plasticity step if 0)
    savedata = True         # Whether to write simulation results to file
    savedir = "res"         # Directory to save the simulation results
    dtsave = 10.            # Periodicity of writing data to hard drive (done only after end of runtime if None)

    dt = 0.01                   # fundamental time unit, relevant only in combination with nmax
    nmax = 1000                 # dt * nmax is the maximum time for mechanical equilibration
    qmin = 0.001                # Threshhold tension beneath which the system is in mechanical equilibrium

    d0min = 0.8                 # min distance between cells when initialized
    d0max = 2.                  # max distance connected by links
    d0_0 = 1.                   # equilibrium distance of links (fundamental scaling of space)
    p_add = 1.                  # rate to add cell-cell links
    p_del = 0.2                 # rate to delete cell-cell links

    chkx = False                # check whether links overlap

    ####################

    # initialize positions for cells in rod
    R = []
    for i in range(12):
        r = (0.1 - 0.2 * npr.random())
        R.append([i, r, 0])
        r = (0.1 - 0.2 * npr.random())
        R.append([i + 0.5, r + 0.5, 0])
    R = np.array(R)

    N = len(R)

    config = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, p_add_subs=p_add,
                      p_del_subs=p_del, chkx=chkx, d0max=d0max, dims=dims, issubs=False)

    config.mynodes.nodesX = R

    # add links between cells adjacent in Voronoi tesselation and closer than d0min
    for i, j in VoronoiNeighbors(config.mynodes.nodesX):
        if np.linalg.norm(config.mynodes.nodesX[i] - config.mynodes.nodesX[j]) <= d0max:
            config.mynodes.addlink(i, j)

    # run and save simulation
    config.timeevo(runtime, dtrec=dtrec, savedata=savedata, savedir=savedir, dtsave=dtsave)
    dump, simdata = fetchdata(savedir)

    # animate results
    animateconfigs(simdata)
    mlab.show()
