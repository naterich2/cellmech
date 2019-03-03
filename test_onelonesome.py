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

    # run simulation of one cell diffusing across a substrate

    ####################

    Lmax = 5                # Length of confining square for substrate cells
    subs_dense = 1.         # substrate densities to test
    runtime = 100.          # Length of simulation run
    dims = 3                # Number of dimensions for the given problem

    savedir = "res"         # Directory to save the simulation results

    dt = 0.01               # fundamental time unit, relevant only in combination with nmax
    nmax = 1000             # dt * nmax is the maximum time for mechanical equilibration
    qmin = 0.001            # Threshhold tension beneath which the system is in mechanical equilibrium

    d0min = 0.8             # min distance between cells when initialized
    d0max = 2.              # max distance connected by links
    d0_0 = 1.               # equilibrium distance of links (fundamental scaling of space)
    p_add = 1.              # rate to add cell-cell links
    p_del = 0.2             # rate to delete cell-cell links

    ####################

    Nsubs = int(Lmax * Lmax * subs_dense)           # number in substrate cells
    N = 1                                           # per definition only one migrating cell
    d0min_subs = round(0.8 / sqrt(subs_dense), 3)   # minimum distance of substrate cells in initialization

    config = CellMech(N, num_subs=Nsubs, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del,
                      p_add_subs=p_add, p_del_subs=p_del, d0max=d0max, dims=dims, issubs="lonesome")

    config.mynodes.nodesX[0] = [0, 0, 0]            # position of tissue cell

    # initialize random positions of substrate tissue
    for ni in range(Nsubs):
        while True:
            R1 = generatePoint(Lmax)
            R1[2] = -d0_0
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(config.mysubs.nodesX[nj] - R1)
                if d < d0min_subs:
                    OK = False
                    break
            if OK:
                break
        config.mysubs.nodesX[ni] = R1

    # add links between cells adjacent in Voronoi tesselation and closer than d0min
    for j in range(config.mysubs.Nsubs):
        if np.linalg.norm(config.mynodes.nodesX[0] - config.mysubs.nodesX[j]) <= d0max:
            config.mysubs.addlink(0, j, config.mynodes.nodesPhi[0])
            break

    # run and save simulation
    configs, trashlinks, nodeforces, trashlinkforces, ts, subs, subslinks, subsnodeforces, subslinkforces = \
        config.timeevo(runtime, record=True)

    links = None
    linkforces = None
    config.savedata(savedir)

    # animate results
    animateconfigs(configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces,
                   showsubs=False)
    mlab.show()
