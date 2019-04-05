from cell import *
from animate import *

npr.seed(seed=0)


if __name__ == '__main__':

    # run one mechanical equilibration for cells with stretched links

    ####################

    N = 5                       # Number of cells. If None: int(2 * Lmax**2)
    dims = 3                    # Number of dimensions for the given problem

    savedir = "res"             # Directory to save the simulation results

    dt = 0.01                   # fundamental time unit, relevant only in combination with nmax
    nmax = 1000                 # dt * nmax is the maximum time for mechanical equilibration
    qmin = 0.001                # Threshhold tension beneath which the system is in mechanical equilibrium

    d0min = 0.8                 # min distance between cells when initialized
    d0max = 2.                  # max distance connected by links
    d0_0 = 1.                   # equilibrium distance of links (fundamental scaling of space)

    ####################

    config = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, d0max=d0max, dims=dims, issubs=False)

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
    simdata = config.oneequil()
    config.savedata(savedir)

    # animate results
    animateconfigs(simdata)
    mlab.show()
