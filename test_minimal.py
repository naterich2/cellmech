from cell import *
from animate import *

npr.seed(seed=0)


if __name__ == '__main__':

    # run one mechanical equilibration for cells with stretched links

    ####################

    N = 5                       # Number of cells. If None: int(2 * Lmax**2)
    dims = 3                    # Number of dimensions for the given problem

    dt = 0.01                   # fundamental time unit, relevant only in combination with nmax
    nmax = 1000                 # dt * nmax is the maximum time for mechanical equilibration
    qmin = 0.001                # Threshhold tension beneath which the system is in mechanical equilibrium

    d0min = 0.8                 # min distance between cells when initialized
    d0max = 2.                  # max distance connected by links
    d0_0 = 1.                   # equilibrium distance of links (fundamental scaling of space)
    p_add = 1.                  # rate to add cell-cell links
    p_del = 0.2                 # rate to delete cell-cell links

    ####################

    config = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del,
                      d0max=d0max, dims=dims, issubs=False)

    # add pre-defined nodes and links between them

    Xnodes = np.array([[0, 0, 1.], [1, 0, 1.], [0, 1., 1.], [0, 0, -1.], [0, 0, 2]])

    config.mynodes.nodesX = Xnodes

    config.mynodes.addlink(0, 1, d0=1.)
    config.mynodes.addlink(0, 2, d0=1.)
    config.mynodes.addlink(1, 2, d0=1.)
    config.mynodes.addlink(0, 3, d0=1.)
    config.mynodes.addlink(1, 3, d0=1.)
    config.mynodes.addlink(2, 3, d0=1.)
    config.mynodes.addlink(0, 4, d0=1.)
    config.mynodes.addlink(1, 4, d0=1.)
    config.mynodes.addlink(2, 4, d0=1.)

    # run and save simulation
    configs, links, nodeforces, linkforces, ts = config.oneequil()
    config.savedata()

    # animate results
    animateconfigs(configs, links, nodeforces, linkforces, ts)
    mlab.show()
