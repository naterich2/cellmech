import numpy as np
from animate import *
"""
configs = np.load("nodesr.npy")
links = np.load("links.npy")
nodeforces = np.load("nodesf.npy")
linkforces = np.load("linksf.npy")
ts = np.load("ts.npy")
"""
"""
subs = np.load("subsnodesr.npy")
subslinks = np.load("subslinks.npy")
subsnodeforces = np.load("subsnodesf.npy")
subslinkforces = np.load("subslinksf.npy")

animateconfigs(configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces)
"""

configs = np.load("nodesr1.npy")
links = np.load("links1.npy")
nodeforces = np.load("nodesf1.npy")
linkforces = np.load("linksf1.npy")
ts = np.load("ts1.npy")

upto = 400

animateconfigs(configs[:upto], links[:upto], nodeforces[:upto], linkforces[:upto], ts[:upto])

mlab.show()