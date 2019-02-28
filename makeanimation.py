from animate import *


if __name__ == '__main__':

    # produce animation from previously saved simulation results

    ####################

    skip = 1            # only use every skip-th simulation step for animation
    dir = "res"         # location of simulation results
    showsubs = False    # whether or not to visualize substrate nodes

    ####################

    configs = np.load(dir + "/nodesr.npy")[::skip]
    links = np.load(dir + "/links.npy")[::skip]
    nodeforces = np.load(dir + "/nodesf.npy")[::skip]
    linkforces = np.load(dir + "/linksf.npy")[::skip]
    ts = np.load(dir + "/ts.npy")[::skip]

    try:     # try to include substrate details if they exists
        subs = np.load(dir + "/subsnodesr.npy")[::skip]
        subslinks = np.load(dir + "/subslinks.npy")[::skip]
        subsnodeforces = np.load(dir + "/subsnodesf.npy")[::skip]
        subslinkforces = np.load(dir + "/subslinksf.npy")[::skip]

        animateconfigs(configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces,
                       showsubs=False)

    except IOError: # if no substrate results exist
        animateconfigs(configs, links, nodeforces, linkforces, ts)

    mlab.show()
