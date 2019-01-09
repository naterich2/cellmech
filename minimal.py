from cell import *
from animate import *
import cProfile
import matplotlib.pyplot as plt
npr.seed(seed=0)

Lmax = 5
N = 3
Nsubs = 4

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

c = CellMech(N, num_subs=Nsubs, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx,
             d0max=d0max, dims=dims, issubs=True)

Xnodes = np.array([[0, 0, 1.], [1, 0, 1.], [0, 1, 1.]])
Xsubs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
for i in range(len(Xnodes)):
    c.mynodes.nodesX[i] = Xnodes[i]
for i in range(len(Xsubs)):
    c.mysubs.nodesX[i] = Xsubs[i]

c.mynodes.updateDists(c.mynodes.nodesX)
c.mysubs.updateDists(c.mynodes.nodesX)
c.mynodes.addlink(0, 1, d0=1.)
c.mynodes.addlink(0, 2, d0=1.)
c.mynodes.addlink(1, 2, d0=1.)

c.mysubs.addlink(0, 0, c.mynodes.nodesPhi[0], d0=1.)
c.mysubs.addlink(1, 0, c.mynodes.nodesPhi[1], d0=1.)
c.mysubs.addlink(2, 0, c.mynodes.nodesPhi[2], d0=1.)
c.mysubs.addlink(0, 1, c.mynodes.nodesPhi[0], d0=1.)
c.mysubs.addlink(1, 1, c.mynodes.nodesPhi[1], d0=1.)
c.mysubs.addlink(2, 1, c.mynodes.nodesPhi[2], d0=1.)
c.mysubs.addlink(0, 2, c.mynodes.nodesPhi[0], d0=1.)
c.mysubs.addlink(1, 2, c.mynodes.nodesPhi[1], d0=1.)
c.mysubs.addlink(2, 2, c.mynodes.nodesPhi[2], d0=1.)
c.mysubs.addlink(0, 3, c.mynodes.nodesPhi[0], d0=1.)
c.mysubs.addlink(1, 3, c.mynodes.nodesPhi[1], d0=1.)
c.mysubs.addlink(2, 3, c.mynodes.nodesPhi[2], d0=1.)


configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces = c.oneequil_withsubs()

animateconfigs(configs, links, nodeforces, linkforces, ts, subs, subslinks, subsnodeforces, subslinkforces)
mlab.show()
