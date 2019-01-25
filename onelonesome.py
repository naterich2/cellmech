from cell import *
from animate import *
import initialconfig
import matplotlib.pyplot as plt
from math import floor

npr.seed(0)

if __name__ == '__main__':

    L = 5
    subs_densities = [2.]

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
    p_del = 1.  # base rate to delete links
    chkx = False  # check if links overlap?

    # onelonesome

    for dense in subs_densities:
        config = initialconfig.onelonesome(L=L, dense=dense, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0,
                                           p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max, dims=dims)

        config.mysubs.updateDists(config.mynodes.nodesX)

        for j in range(config.mysubs.Nsubs):
            if np.linalg.norm(config.mynodes.nodesX[0] - config.mysubs.nodesX[j]) <= d0max:
                config.mysubs.addlink(0, j, config.mynodes.nodesPhi[0])
                break

        configs, trashlinks, nodeforces, trashlinkforces, ts, subs, subslinks, subsnodeforces, subslinkforces = \
            config.timeevo(20., record=True)

        links = None
        linkforces = None
        config.savedata()
        animateconfigs(configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces,
                       showsubs=False)
        mlab.show()

