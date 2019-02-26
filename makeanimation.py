import numpy as np
from animate import *

configs = np.load("res/nodesr.npy")
links = np.load("res/links.npy")
nodeforces = np.load("res/nodesf.npy")
linkforces = np.load("res/linksf.npy")
ts = np.load("res/ts.npy")
skip=10
animateconfigs(configs[::skip], links[::skip], nodeforces[::skip], linkforces[::skip], ts[::skip]) # , subs, subslinks, subsnodeforces, subslinkforces)
"""
configs = np.load("res/nodesr.npy")
links = np.load("res/links.npy")
nodeforces = np.load("res/nodesf.npy")
linkforces = np.load("res/linksf.npy")
ts = np.load("res/ts.npy")
subs = np.load("res/subsnodesr.npy")
subslinks = np.load("res/subslinks.npy")
subsnodeforces = np.load("res/subsnodesf.npy")
subslinkforces = np.load("res/subslinksf.npy")

animateconfigs(configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces, showsubs=False)
"""
"""
configs = np.load("res/nodesr1.npy")
links = np.load("res/links1.npy")
nodeforces = np.load("res/nodesf1.npy")
linkforces = np.load("res/linksf1.npy")
ts = np.load("res/ts1.npy")

upto = -1

animateconfigs(configs[:upto], links[:upto], nodeforces[:upto], linkforces[:upto], ts[:upto])
"""
mlab.show()