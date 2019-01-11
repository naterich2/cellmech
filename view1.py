from testplane import *

if __name__ == '__main__':

    Lmax = 5
    N = None

    bend = 10.0
    twist = 1.0
    dt = 0.01
    nmax = 3000
    qmin = 0.001
    dims = 3

    d0min = 0.8  # min distance between cells
    d0max = 2.  # max distance connected by links
    d0_0 = 1.  # equilibrium distance of links
    p_add = 1.0  # rate to add links
    p_del = 0.1  # base rate to delete links
    chkx = False  # check if links overlap?

    # config = generate_initial_random_wsubs(L=Lmax, N=N, Nsubs=N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0,
    #                                       p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max, dims=dims)

    config = generate_initial_bilayer(L=Lmax, N=N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0,
                                      p_add=p_add, p_del=p_del, chkx=chkx, d0min=d0min, d0max=d0max, dims=dims)

    config.mynodes.updateDists(config.mynodes.nodesX)

    for i, j in VoronoiNeighbors(config.mynodes.nodesX):
        if np.linalg.norm(config.mynodes.nodesX[i] - config.mynodes.nodesX[j]) <= d0max:
            config.mynodes.addlink(i, j)

    config.timeevo(22.9, isfinis=False)

    configs, links, nodeforces, linkforces, ts = config.oneequil()

    np.save("nodesr1", configs)
    np.save("links1", links)
    np.save("nodesf1", nodeforces)
    np.save("linksf1", linkforces)
    np.save("ts1", ts)
    np.save("dx1", dx)
    np.save("dphi1", dphi)
    np.save("k1", K)

    upto = 400

    animateconfigs(configs[:upto], links[:upto], nodeforces[:upto], linkforces[:upto], ts[:upto])
    mlab.show()