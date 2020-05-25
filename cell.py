from __future__ import division

import sys
import os
import shutil
import warnings

import numpy as np
import numpy.random as npr
import scipy.linalg
from scipy.spatial import Delaunay
from scipy.stats import lognorm
import itertools

from math import exp, log, sqrt

from myivp.myivp import solve_ivp

warnings.filterwarnings("ignore", category=DeprecationWarning)

null = np.array([0.0, 0.0, 0.0])
ex = np.array([1.0, 0.0, 0.0])
ey = np.array([0.0, 1.0, 0.0])
ez = np.array([0.0, 0.0, 1.0])


def update_progress(progress):
    """
    Simple progress bar update.
    :param progress: float. Fraction of the work done, to update bar.
    :return:
    """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rProgress: [{0}] {1} % {2}".format("#" * block + "-" * (barLength - block), round(progress * 100, 1),
                                                status)
    sys.stdout.write(text)
    sys.stdout.flush()


def ccw(A, B, C):
    return np.greater((C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]),
                      (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0]))


def getNormvec(v):
    """
    Calculate normalized vector(s).
    :param v: numpy array of shape (n, k).
    :return: numpy array of shape (n, k) normalized along last axis
    """
    d = scipy.linalg.norm(v, axis=-1)
    vecinds = np.where(d > 1e-5)  # filter for division by 0
    v[vecinds] /= d[..., None][vecinds]
    return v


def getNormtoo(v):
    """
    Calculate normalized vector(s) and norms.
    :param v: numpy array of shape (n, k).
    :return: (1) numpy array of shape (n, k) normalized along last axis and
             (2) numpy array of shape (n) containing the norms of the n k-shaped vectors.
    """
    d = scipy.linalg.norm(v, axis=-1)
    vecinds = np.where(d > 1e-5)  # filter for division by 0
    v[vecinds] /= d[..., None][vecinds]
    return v, d


def getRotMatArray(Phis):
    """
    Calculate rotation matrices from vectors indicating the rotation axis
    :param Phis: numpy array of shape (n, 3)
    :return: numpy array of shape (n, 3, 3). Each n underlying 3-d-vector phi in Phis is used as the basis of a
    3x3-rotation matrix, where the direction of phi indicates the axis and direction of rotation, and its length is the
    angle of rotation
    """
    Thetas = scipy.linalg.norm(Phis, axis=1)  # calculate the angle of rotation
    phiinds = np.where(Thetas > 1e-5)  # filter for division by 0
    Phis[phiinds] /= Thetas[..., None][phiinds]
    a = np.cos(Thetas / 2)
    b, c, d = np.transpose(Phis) * np.sin(Thetas / 2)
    RotMat = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                      [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                      [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])
    return np.transpose(RotMat, axes=(2, 0, 1))


def getRotMat(Phis):
    """
    Calculate rotation matrix from vector indicating the rotation axis
    :param Phis: numpy array of shape (3)
    :return: numpy array of shape (3, 3). Phis is used as the basis of a 3x3-rotation matrix, where the direction of
    Phis indicates the axis and direction of rotation, and its length is the angle of rotation
    """
    Axis, Theta = getNormtoo(Phis)  # calculate the angle of rotation
    a = np.cos(Theta / 2)  # filter for division by 0
    b, c, d = Axis * np.sin(Theta / 2)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                    [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                    [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def VoronoiNeighbors(positions, vodims=2):
    """
    Calculate set of neighbors in a Voronoi tessellation form given positions
    :param positions: numpy array of shape (n, 3) indicating the positions of particles
    :param vodims: should be value 2 or 3, indicating the number of dimensions in which the tessellation is performed
    :return: set of tuples (i, j) indicating the indices of neighboring particles in positions with i<j
    """
    if vodims == 2:
        positions = [n[:2] for n in positions]
    # tri: list of interconnected particles: [ (a, b, c), (b, c, d), ... ]
    tri = Delaunay(positions, qhull_options='QJ')
    # neighbors contain pairs of adjacent particles: [ (a,b), (c,d), ... ]
    neighbors = [list(itertools.combinations(v, 2)) for v in tri.simplices]
    n = []
    for (i, j) in itertools.chain.from_iterable(neighbors):
        if i < j:
            n.append((i, j))
        else:
            n.append((j, i))
    neighbors = set(n)

    return neighbors


def relaunch_CellMech(savedir, num_cells, num_subs=0, dt=0.01, nmax=300, qmin=0.001, d0_0=1., p_add=1., p_del=0.2,
                      p_add_subs=None, p_del_subs=None, c1=0.05, c2=0.1, c3=0.2, chkx=False, d0max=2., dims=3, F_contr=1.,
                      isF0=False, isanchor=False, issubs=False, force_contr=True):
    """
    Create an instance of CellMech and set it up so that a simulation can be continued from where it was previously
    ended. Take special care if settings where changed in space or time (e.g. bend, twist or Hookean parameters),
    manual changes might become necessary. Take care to run timeevo with isinit=False. Also not set up for anchors
    and external forces yet.
    :param savedir: string, name of directory where previous data is saved
    :param num_cells: integer, the number of tissue cells
    :param num_subs: integer, the number of substrate cells
    :param dt: float, the time unit for scaling simulation time
    :param nmax: integer, the maximum time (in simulation time) for until cutoff when calculating
        mechanical equilibrium
    :param qmin: float, the square of the maximum force per cell until mechanical equilibration is cut off
    :param d0_0: float, the global equilibrium link length (d_0 in czirok2014cell)
    :param p_add: float, base probability for adding tissue-tissue links
    :param p_del: float, base probability for removing tissue-tissue links
    :param p_add_subs: float, base probability for adding tissue-substrate links
    :param p_del_subs: float, base probability for removing tissue-substrate links
    :param c1: float, constant of contractility/volume exlusion
    :param c2: float, constant of tissue elasticity
    :param c3: float, the variance of the noise in updating the link lengths is 2 * (c2 ** 2) * dt were dt is the
            time used for modifying a link
    :param chkx: bool, whether or not to check for link crossings (only functional for dims==2
    :param d0max: float, maximum cell-cell distance to allow a link to be added
    :param dims: 2 or 3, the number of dimensions of the simulations
    :param F_contr: target force for force-dependent update of equilibrium lengths
    :param isF0: bool, whether or not external forces are a part of the problem
    :param isanchor: bool, whether or not tissue cells are anchored to a x0-position
    :param issubs: True (with substrate), False (without substrate) or "lonesome" (with substrate but only one
        tissue cell)
    :param force_contr: boolean, if False: update done as suggested in czirok2014cell. if True: force-dependent
        component included.
    :return: Initiated instance of CellMech
    """

    c = CellMech(num_cells=num_cells, num_subs=num_subs, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add,
                 p_del=p_del, p_add_subs=p_add_subs, p_del_subs=p_del_subs, c1=c1, c2=c2, c3=c3,
                 chkx=chkx, d0max=d0max, dims=dims, F_contr=F_contr, isF0=isF0, isanchor=isanchor, issubs=issubs,
                 force_contr=force_contr)

    # load data (everything not related to the substrate)
    # load data on time, save the last time step,
    # create sub-folder in savedir for snapshots with all previous data saved as number 0
    sts = np.load(savedir + "/ts.npy")
    c.snaptimes = list(sts)
    c.lastt = c.snaptimes[-1]
    c.snaptimes = c.saveonesnap("ts", savedir, c.snaptimes)
    del sts

    # load data on tissue cell positions, save last positions,
    # create sub-folder in savedir for snapshots with all previous data saved as number 0
    snodesr = np.load(savedir + "/nodesr.npy")
    c.mynodes.nodesnap = list(snodesr)
    c.mynodes.nodesX = c.mynodes.nodesnap[-1]
    c.mynodes.nodesnap = c.saveonesnap("nodesr", savedir, c.mynodes.nodesnap)
    del snodesr

    # load last tissue cell orientations
    c.mynodes.nodesPhi = np.load(savedir + "/phi.npy")

    # load data on existing links, recreate last existing tissue-tissue links,
    # create sub-folder in savedir for snapshots with all previous data saved as number 0
    slinks = list(np.load(savedir + "/links.npy"))
    lastlinks = slinks[-1]
    for link in lastlinks:
        c.mynodes.addlink(link[0], link[1])
    c.mynodes.linksnap = c.saveonesnap("links", savedir, slinks)
    del slinks, lastlinks

    # create sub-folder in savedir for snapshots on node forces with all previous data saved as number 0
    c.mynodes.fnodesnap = c.saveonesnap("nodesf", savedir, np.load(savedir + "/nodesf.npy"))

    # create sub-folder in savedir for snapshots on link forces with all previous data saved as number 0
    c.mynodes.flinksnap = c.saveonesnap("linksf", savedir, np.load(savedir + "/linksf.npy"))

    # load data on t and n vectors and on individual link equilibrium lengths
    nodeinds = np.where(c.mynodes.islink == True)

    c.mynodes.t[nodeinds] = np.load(savedir + "/tang.npy")
    c.mynodes.norm[nodeinds] = np.load(savedir + "/norm.npy")
    c.mynodes.d0[nodeinds] = np.load(savedir + "/d0.npy")

    if c.issubs is not False:
        # do everything for substrate
        # load data on tissue cell positions, save last positions,
        # create sub-folder in savedir for snapshots with all previous data saved as number 0
        c.mysubs.nodesX = np.load(savedir + "/subsnodesr.npy")

        # load last tissue cell orientations
        c.mysubs.nodesPhi = np.load(savedir + "/subsphi.npy")

        # load data on existing links, recreate last existing tissue-tissue links,
        # create sub-folder in savedir for snapshots with all previous data saved as number 0
        ssubslinks = list(np.load(savedir + "/subslinks.npy"))
        lastlinks = ssubslinks[-1]
        for link in lastlinks:
            c.mysubs.addlink(link[0], link[1], c.mynodes.nodesX[link[0]], c.mynodes.nodesPhi[link[0]])
        c.mysubs.linksnap = c.saveonesnap("subslinks", savedir, ssubslinks)

        # create sub-folder in savedir for snapshots on node forces with all previous data saved as number 0
        c.mysubs.fnodesnap = c.saveonesnap("subsnodesf", savedir, np.load(savedir + "/subsnodesf.npy"))

        # create sub-folder in savedir for snapshots on link forces with all previous data saved as number 0
        c.mynodes.flinksnap = c.saveonesnap("subslinksf", savedir, np.load(savedir + "/subslinksf.npy"))

        # load data on t and n vectors and on individual link equilibrium lengths
        nodeinds = np.where(c.mysubs.islink == True)

        c.mysubs.tcell[nodeinds] = np.load(savedir + "/substcell.npy")
        c.mysubs.tsubs[nodeinds] = np.load(savedir + "/substsubs.npy")
        c.mysubs.normcell[nodeinds] = np.load(savedir + "/subsnormcell.npy")
        c.mysubs.normsubs[nodeinds] = np.load(savedir + "/subsnormsubs.npy")
        c.mysubs.d0[nodeinds] = np.load(savedir + "/subsd0.npy")

    c.nsaves += 1

    return c


class NodeConfiguration:
    def __init__(self, num, num_subs, d0_0, p_add, p_del, c1, c2, c3, F_contr, dims, isF0, isanchor, plasticity):
        """
        Class containing data for all tissue nodes and tissue-tissue links. Is automatically initialized by class
        CellMech
        :param num: integer, the number of tissue cells
        :param num_subs: integer, the number of substrate cells
        :param d0_0: float, the global equilibrium link length (d_0 in czirok2014cell)
        :param p_add: float, base probability for adding tissue-tissue links
        :param p_del: float, base probability for removing tissue-tissue links
        :param c1: float, constant of contractility/volume exlusion
        :param c2: float, constant of tissue elasticity
        :param c3: float, the variance of the noise in updating the link lengths is 2 * (c2 ** 2) * dt were dt is the
            time used for modifying a link
        :param F_contr: target value for contractile force
        :param dims: 2 or 3, the number of dimensions of the simulations
        :param isF0: bool, whether or not external forces are a part of the problem
        :param isanchor: bool, whether or not tissue cells are anchored to a x0-position
        :param plasticity: Either None if Hookean, bend and twist constants are set individually per link, or tuple
            containing global values for the three constants in shape (k1, k2, k3) = (bend, twist, Hooke)
        """
        if dims == 2:
            self.updateLinkForces = lambda PHI, T, Norm, NormT, Bend, Twist, K, D0, Nodeinds: \
                self.updateLinkForces2D(PHI, T, Bend, K, D0, Nodeinds)
            self.dims = dims
        elif dims == 3:
            self.updateLinkForces = lambda PHI, T, Norm, NormT, Bend, Twist, K, D0, Nodeinds: \
                self.updateLinkForces3D(PHI, T, Norm, NormT, Bend, Twist, K, D0, Nodeinds)
            self.dims = dims
        else:
            print("Oops! Wrong number of dimensions here.")
            sys.exit()

        self.dims = dims
        # variables to store cell number and cell positions and angles
        self.N = num
        self.N2 = 2 * self.N
        self.N_inv = 1. / self.N

        # description of nodes
        self.nodesX = np.zeros((self.N, 3))             # r of nodes
        self.nodesPhi = np.zeros((self.N, 3))           # phi of nodes
        self.Fnode = np.zeros((self.N, 3))              # total force on node
        self.Mnode = np.zeros((self.N, 3))              # total torsion on node
        self.isF0 = isF0
        self.F0 = np.zeros((self.N, 3))                 # external force on node
        self.isanchor = isanchor
        self.X0 = np.zeros((self.N, 3))                 # node anchor, must be set if needed!
        self.knode = np.zeros((self.N,))                # spring constant of node to anchor point, defaults to 0

        self.gaps = np.zeros((num_subs, 3))

        # description of links
        # islink[i, j] is True if nodes i and j are connected via link
        self.islink = np.full((self.N, self.N), False)  # Describes link at node [0] leading to node [1]

        self.e = np.zeros((self.N, self.N, 3))           # direction connecting nodes (a.k.a. "actual direction")
        self.d = np.zeros((self.N, self.N))              # distance between nodes (a.k.a. "actual distance")
        self.d0_0 = d0_0                                 # global equilibrium link length
        if plasticity is None:
            self.k = np.zeros((self.N, self.N))              # spring constant between nodes
            self.bend = np.zeros((self.N, self.N))           # bending rigidity
            self.twist = np.zeros((self.N, self.N))          # torsion spring constant
            self.saveram = False
        else:
            self.bend = plasticity[0]
            self.twist = plasticity[1]
            self.k = plasticity[2]
            self.saveram = True
        self.d0 = np.zeros((self.N, self.N))             # equilibrium distance between nodes,
        self.t = np.zeros((self.N, self.N, 3))           # tangent vector of link at node (a.k.a. "preferred direction")
        self.norm = np.zeros((self.N, self.N, 3))        # normal vector of link at node
        self.Mlink = np.zeros((self.N, self.N, 3))       # Torsion from link on node
        self.Flink = np.zeros((self.N, self.N, 3))       # Force from link on node
        self.Flink_tens = np.zeros((self.N, self.N))     # Tensile component of Flink
        self.F_contr = F_contr                           # Target value for contractile force

        self.p_add = p_add
        self.p_del = p_del
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        # functions for randoms in default_update_d0
        self.lowers = np.tril_indices(self.N, -1)
        self.randomlength = int(self.N * (self.N - 1) / 2)
        self.randomsummand = np.zeros((self.N, self.N))

        # stuff for documentation
        self.nodesnap = []
        self.linksnap = []
        self.fnodesnap = []
        self.flinksnap = []

        self.nodesum = lambda: 0

        self.reset_nodesum()

    def reset_nodesum(self):
        """
        Reset the different components taken in account when the total forces on nodes are set up, based on setting of
        self.isF0 and self.isanchor
        :return:
        """
        if self.isF0 is False and self.isanchor is False:
            # only forces exerted by tissue-tissue links
            self.nodesum = lambda: np.sum(self.Flink, axis=1)
        elif self.isF0 is True and self.isanchor is False:
            # tissue-tissue link forces and external forces
            self.nodesum = lambda: np.sum(self.Flink, axis=1) + self.F0
        elif self.isF0 is False and self.isanchor is True:
            # tissue-tissue link forces and forces resulting from nodes being anchored to r0
            self.nodesum = lambda: np.sum(self.Flink, axis=1) + \
                                   np.multiply(self.knode[..., None], (self.X0 - self.nodesX))
        elif self.isF0 is True and self.isanchor is True:
            # tissue-tissue link forces, forces resulting from nodes being anchored to r0 and external forces
            self.nodesum = lambda: np.sum(self.Flink, axis=1) + self.F0 + \
                                   np.multiply(self.knode[..., None], (self.X0 - self.nodesX))

    def addlink(self, ni, mi, t1=None, t2=None, d0=None, bend=1., twist=1., k=1.5, n=None, norm1=None, norm2=None):
        """
        Add a new tissue-tissue link
        :param ni: integer, index of one of the cells involved in the link
        :param mi: integer, index of the second cell
        :param t1: numpy array of shape (3), the vector tangential to the link at the surface of cell ni or None
        :param t2: numpy array of shape (3), the vector tangential to the link at the surface of cell mi or None
        :param d0: float, the link's initial equilibrium length. If None: set to current link length
        :param bend: float, the link's bending rigidity
        :param twist: float, the link's twisting rigidity
        :param k: float, the link's Hookean constant.
        :param n: numpy array of shape (3), the chosen normal vector, or None
        :param norm1: numpy array of shape (3), the chosen normal vector at cell ni. If None: set to n
        :param norm2: numpy array of shape (3), the chosen normal vector at cell mi. If None: set to n
        :return:
        """
        self.islink[ni, mi], self.islink[mi, ni] = True, True

        if not self.saveram:
            self.k[ni, mi], self.k[mi, ni] = k, k  # spring parameter
            self.bend[mi, ni], self.bend[ni, mi] = bend, bend
            self.twist[mi, ni], self.twist[ni, mi] = twist, twist

        newdX = self.nodesX[mi] - self.nodesX[ni]
        newd = scipy.linalg.norm(newdX)
        newe = newdX/newd
        self.d[ni, mi], self.d[mi, ni] = newd, newd
        self.e[ni, mi], self.e[mi, ni] = newe, -newe

        if d0 is None:
            d0 = self.d[ni, mi]
        self.d0[ni, mi], self.d0[mi, ni] = d0, d0  # equilibrium distance
        # preferred directions
        RotMat1 = getRotMat(-self.nodesPhi[ni])
        RotMat2 = getRotMat(-self.nodesPhi[mi])
        if t1 is None:
            self.t[ni, mi] = np.dot(RotMat1, self.e[ni, mi])
        else:
            self.t[ni, mi] = np.dot(RotMat1, getNormvec(t1))
        if t2 is None:
            self.t[mi, ni] = np.dot(RotMat2, self.e[mi, ni])
        else:
            self.t[mi, ni] = np.dot(RotMat2, getNormvec(t2))
        if n is None:
            n, q = getNormtoo(np.cross(self.e[ni, mi], ez))  # n is perpendicular to e
            # n is perpendicular to z (l is in the x-y plane)
            if q < 1e-5:
                n = getNormvec(np.cross(self.e[ni, mi], ex))  # e || ez   =>	n is perpendicular to x
        if norm1 is None:
            norm1 = np.dot(RotMat1, n)
        self.norm[ni, mi] = norm1
        if norm2 is None:
            norm2 = np.dot(RotMat2, n)
        self.norm[mi, ni] = norm2

    def removelink(self, ni, mi):
        """
        Remove the link between cells ni and mi
        :param ni: integer, index of one of the cells involved in the link
        :param mi: integer, index of the second cell
        :return:
        """
        self.islink[ni, mi], self.islink[mi, ni] = False, False
        self.d[ni, mi], self.d[mi, ni] = 0, 0
        self.e[ni, mi], self.e[mi, ni] = null, null
        self.Flink[ni, mi], self.Flink[mi, ni], self.Mlink[ni, mi], self.Mlink[mi, ni] = null, null, null, null
        self.Flink_tens[ni, mi], self.Flink_tens[mi, ni] = 0, 0
        self.t[ni, mi], self.t[mi, ni], self.norm[ni, mi], self.norm[mi, ni] = null, null, null, null
        self.d0[ni, mi], self.d0[mi, ni] = 0, 0
        if not self.saveram:
            self.k[ni, mi], self.k[mi, ni] = 0, 0
            self.bend[ni, mi], self.bend[mi, ni], self.twist[ni, mi], self.twist[mi, ni] = 0, 0, 0, 0

    def updateDists(self, X):
        """
        Calculate the distances and directions between nodes which are connected by links and save the information in
        self.d (distances) and self.e (normed directions)
        :param X: numpy array of shape (n), containing the positions for which the calculations should be performed
        :return:
        """
        inds0, inds1 = self.getLinkTuple()
        dX = X[inds1] - X[inds0]
        d = scipy.linalg.norm(dX, axis=1)
        e = dX / d[..., None]
        self.d[inds0, inds1], self.d[inds1, inds0] = d, d
        self.e[inds0, inds1], self.e[inds1, inds0] = e, -e

    def compactStuffINeed(self):
        """
        Extract the relevant information on existing links from large arrays containing information on all
        hypothetically possible links
        :return: compacted numpy arrays for self.t (nl, 3), self.norm (nl, 3), transposed version of self.norm (nl, 3),
        self.bend (nl), self.twist (nl), self.k (nl), self.d0 (nl) and indices of nodes at ends of links (nl, 2).
        Parantheses indicate shapes of arrays, nl is the number of links
        """
        nodeinds = np.where(self.islink == True)
        nodelen = len(nodeinds[0])
        t = self.t[nodeinds]
        norm = self.norm[nodeinds]
        normT = np.transpose(self.norm, axes=(1, 0, 2))[nodeinds]
        if not self.saveram:
            bend = self.bend[nodeinds]
            twist = self.twist[nodeinds]
            k = self.k[nodeinds]
        else:
            bend = self.bend * np.ones((nodelen,))
            twist = self.twist * np.ones((nodelen,))
            k = self.k * np.ones((nodelen,))
        d0 = self.d0[nodeinds]

        return t, norm, normT, bend, twist, k, d0, nodeinds

    def updateLinkForces2D(self, PHI, T, Bend, K, D0, Nodeinds):
        """
        Update the forces exerted on the links for 2-d-simulations. Input is of shape created by compactStuffINeed()
        :param PHI: Orientation of tissue nodes
        :param T: tangent vectors at tissue cell surfaces
        :param Bend: bending rigidities
        :param K: Hookean constants
        :param D0: individual link equilibrium lengths
        :param Nodeinds: link indices
        :return:
        """
        E = self.e[Nodeinds]
        D = self.d[Nodeinds]

        # rotated version of t to fit current setup
        TNow = np.einsum("ijk, ik -> ij", getRotMatArray(PHI[Nodeinds[0]]), T)

        self.Mlink[Nodeinds] = Bend[..., None] * np.cross(TNow, E)  # Eq 3

        M = self.Mlink + np.transpose(self.Mlink, axes=(1, 0, 2))
        M = M[Nodeinds]

        # Eqs. 10, 13, 14, 15
        self.Flink_tens[Nodeinds] = K * (D - D0)
        self.Flink[Nodeinds] = self.Flink_tens[Nodeinds][..., None] * E + np.cross(M, E) / D[:, None]

    def updateLinkForces3D(self, PHI, T, Norm, NormT, Bend, Twist, K, D0, Nodeinds):
        """
        Update the forces exerted on the links for 3-d-simulations. Input is of shape created by compactStuffINeed()
        :param PHI: Orientation of tissue nodes
        :param T: tangent vectors at tissue cell surfaces
        :param Norm: normal vectors at tissue cell surfaces
        :param NormT: re-ordered normal vectors
        :param Bend: bending rigidities
        :param Twist: twist rigidity
        :param K: Hookean constants
        :param D0: individual link equilibrium lengths
        :param Nodeinds: link indices
        :return:
        """
        E = self.e[Nodeinds]
        D = self.d[Nodeinds]
        # NodesPhi = PHI[Nodeinds[0]]
        # NodesPhiT = PHI[Nodeinds[1]]

        rot = getRotMatArray(PHI[Nodeinds[0]])

        # rotated version of Norm and NormT to fit current setup
        NormNow = np.einsum("ijk, ik -> ij", rot, Norm)
        NormTNow = np.einsum("ijk, ik -> ij", getRotMatArray(PHI[Nodeinds[1]]), NormT)

        # rotated version of t to fit current setup
        # TNow = np.einsum("ijk, ik -> ij", getRotMatArray(NodesPhi), T)

        # calculated new vector \bm{\tilde{n}}_{A, l}
        NormTilde = getNormvec(NormNow - np.einsum("ij, ij -> i", NormNow, E)[:, None] * E)
        NormTTilde = getNormvec(NormTNow - np.einsum("ij, ij -> i", NormTNow, E)[:, None] * E)

        self.Mlink[Nodeinds] = Bend[..., None] * np.cross(np.einsum("ijk, ik -> ij", rot, T), E) + \
                               Twist[..., None] * np.cross(NormTilde, NormTTilde)  # Eq 5

        M = self.Mlink + np.transpose(self.Mlink, axes=(1, 0, 2))
        M = M[Nodeinds]

        # Eqs. 10, 13, 14, 15
        self.Flink_tens[Nodeinds] = K * (D - D0)
        self.Flink[Nodeinds] = self.Flink_tens[Nodeinds][..., None] * E + np.cross(M, E) / D[:, None]

    def getForces(self, x, t, norm, normT, bend, twist, k, d0, nodeinds):
        """
        Calculate forces and torques on tissue nodes and tissue-tissue links. Input except for x in shape returned by
        compactStuffINeed()
        :param x: numpy array of shape (3 * 2 * self.N) with positions and orientations of tissue nodes
        for which forces should be calculated
        :param t: tangent vectors at tissue cell surfaces
        :param norm: normal vectors at tissue cell surfaces
        :param normT: re-ordered normal vectors
        :param bend: bending rigidities
        :param twist: twist rigidity
        :param k: Hookean constants
        :param d0: individual link equilibrium lengths
        :param nodeinds: link indices
        :return: numpy array of shape (3 * 2 * self.N) containing forces and torques on tissue nodes in form readable by
        solve_ivp
        """

        # reshape X to form readable by class
        X = x.reshape(-1, 3)
        Phi = X[self.N:self.N2, :]
        X = X[:self.N, :]
        self.updateDists(X)
        self.updateLinkForces(Phi, t, norm, normT, bend, twist, k, d0, nodeinds)
        self.Fnode = self.nodesum()
        self.Mnode = np.sum(self.Mlink, axis=1)
        return np.concatenate((self.Fnode, self.Mnode, self.gaps), axis=0).flatten()

    def getLinkList(self):
        """
        Get an array of the indices of the nodes at each end of each link
        :return: numpy array of shape (nl, 2) where nl is the number of links. All links along axis 0, the indices of
        the two nodes connected by the link along axis 1, where the node with the larger index is the first entry
        """
        allLinks0, allLinks1 = np.where(self.islink == True)
        return np.array([[allLinks0[i], allLinks1[i]] for i in range(len(allLinks0)) if allLinks0[i] > allLinks1[i]])

    def getLinkTuple(self):
        """
        Get a tuple of the indices of the nodes at the end of each link
        :return: tuple (a, b) of numpy arrays of shape (nl), where nl is the number of links. a[i] and b[i] are the
        nodes at the two ends of the i-th link.
        """
        allLinks0, allLinks1 = np.where(self.islink == True)
        inds = np.where(allLinks0 > allLinks1)
        return allLinks0[inds], allLinks1[inds]

    def update_d0(self, dt, force=True):
        """
        Update the equilibrium link length so that each link maintains a constant force
        :param dt: float, the time taken for the last plasticity step
        :param force: boolean, if False: update done as suggested in czirok2014cell. if True: force-dependent component
        included.
        :return:
        """
        nodeinds = np.where(self.islink == True)
        myd0 = self.d0[nodeinds]

        temprandom = npr.random((self.randomlength,))
        self.randomsummand[self.lowers], self.randomsummand.T[self.lowers] = temprandom, temprandom

        if force:
            # lognorm fitted to match behavior for d0min==0.8, d0max==2.0 and d0_0==1.0
            myd0 += self.c1 * ((self.Flink_tens[nodeinds]) - self.F_contr) * dt * \
                    0.69 * lognorm.pdf(self.d[nodeinds], .7, loc=.7, scale=.5)

        myd0 += self.c2 * (self.d0_0 - myd0) * dt + self.c3 * sqrt(dt) * (2 * self.randomsummand[nodeinds] - 1)

        self.d0[nodeinds] = myd0


class SubsConfiguration:
    def __init__(self, num_cells, num_subs, d0_0, p_add, p_del, c1, c2, c3, F_contr, plasticity):
        """
        Class containing data for all substrate nodes and substrate-tissue links. Is automatically initialized by class
        CellMech if CellMech.issubs is not False. Substrate nodes behave like tissue nodes, but can only form links
        with other tissue nodes. They have three rotational but no translational degrees of freedom.

        :param num_cells: integer, the number of tissue cells
        :param num_subs:  integer, the number of substrate cells
        :param d0_0: float, the global equilibrium link length (d_0 in czirok 2014cell)
        :param p_add: float, base probability for adding tissue-tissue links
        :param p_del: float, base probability for removing tissue-tissue links
        :param c1: float, constant of contractility/volume exlusion
        :param c2: float, constant of tissue elasticity
        :param c3: float, the variance of the noise in updating the link lengths is 2 * (c2 ** 2) * dt were dt is the
            time used for modifying a link
        :param F_contr: target value for contractile force
        :param plasticity: Either None if Hookean, bend and twist constants are set individually per link, or tuple
            containing global values for the three constants in shape (k1, k2, k3) = (bend, twist, Hooke)
        """
        # variables to store cell number and cell positions and angles
        self.N = num_cells
        self.N2 = 2 * self.N
        self.Nsubs = num_subs

        # description of nodes
        self.nodesX = np.zeros((self.Nsubs, 3))              # r of subs nodes
        self.nodesPhi = np.zeros((self.Nsubs, 3))            # phi of subs nodes
        self.Fnode = np.zeros((self.Nsubs, 3))               # force exerted on subs nodes
        self.Mnode = np.zeros((self.Nsubs, 3))               # torque exerted on subs nodes

        # description of links
        # islink[i, j] is True if nodes i and j are connected via link
        self.islink = np.full((self.N, self.Nsubs), False)   # Describes link at cell node [0] leading to subs node [1]

        self.e = np.zeros((self.N, self.Nsubs, 3))           # direction from cell node to subs node
        self.d = np.zeros((self.N, self.Nsubs))              # distance between nodes (a.k.a. "actual distance")
        if plasticity is None:
            self.k = np.zeros((self.N, self.Nsubs))              # spring constant between nodes
            self.bend = np.zeros((self.N, self.Nsubs))           # bending rigidity
            self.twist = np.zeros((self.N, self.Nsubs))          # torsion spring constant
            self.saveram = False
        else:
            self.bend = plasticity[0]
            self.twist = plasticity[1]
            self.k = plasticity[2]
            self.saveram = True
        self.d0 = np.zeros((self.N, self.Nsubs))             # equilibrium distance between nodes
        self.d0_0 = d0_0                                     # global target equilibrium link length
        self.tcell = np.zeros((self.N, self.Nsubs, 3))       # tangent vector of link at cell node
        self.tsubs = np.zeros((self.N, self.Nsubs, 3))       # tangent vector of link at subs node
        self.normcell = np.zeros((self.N, self.Nsubs, 3))    # normal vector of link at cell node
        self.normsubs = np.zeros((self.N, self.Nsubs, 3))    # normal vector of link at subs node
        self.Mcelllink = np.zeros((self.N, self.Nsubs, 3))   # Torsion from link on cell node
        self.Msubslink = np.zeros((self.N, self.Nsubs, 3))   # Torsion from link on subs node
        self.Flink = np.zeros((self.N, self.Nsubs, 3))       # Force from link on cell node
        self.Flink_tens = np.zeros((self.N, self.Nsubs))     # Tensile component of Flink
        self.F_contr = F_contr                               # target value for contractile force

        self.p_add = p_add
        self.p_del = p_del
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        # stuff for documentation
        self.linksnap = []
        self.fnodesnap = []
        self.flinksnap = []

    def addlink(self, ni, mi, cellx, cellphi, t1=None, d0=None, bend=1., twist=1., k=1.5,
                n=None, norm1=None, norm2=None):
        """
        Add a new tissue-substrate link
        :param ni: integer, index of the tissue node involved in the link
        :param mi: integer, index of the substrate noed involved in the link
        :param cellx: position of the tissue node
        :param cellphi: orientation of the tissue node
        :param t1: numpy array of shape (3), the vector tangential to the link at the surface of the substrate cell mi
            or None
        :param d0: float, the link's initial equilibrium length. If None: set to current link length
        :param bend: float, the link's bending rigidity
        :param twist: float, the link's twisting rigidity
        :param k: float, the link's Hookean constant.
        :param n: numpy array of shape (3), the chosen normal vector, or None
        :param norm1: numpy array of shape (3), the chosen normal vector at tissue cell ni. If None: set to n
        :param norm2: numpy array of shape (3), the chosen normal vector at substrate cell mi. If None: set to n
        :return:
        """

        self.islink[ni, mi] = True

        if not self.saveram:
            self.k[ni, mi] = k  # spring parameter
            self.bend[ni, mi] = bend
            self.twist[ni, mi] = twist

        newdX = self.nodesX[mi] - cellx
        newd = scipy.linalg.norm(newdX)
        self.d[ni, mi] = newd
        self.e[ni, mi] = newdX / newd

        if d0 is None:
            d0 = self.d[ni, mi]
        self.d0[ni, mi] = d0  # equilibrium distance
        RotMat1 = getRotMat(-cellphi)
        if t1 is None:
            self.tcell[ni, mi] = np.dot(RotMat1, self.e[ni, mi])
            self.tsubs[ni, mi] = -self.e[ni, mi]
        else:
            self.tcell[ni, mi] = np.dot(RotMat1, getNormvec(t1))
            self.tsubs[ni, mi] = getNormvec(t1)
        if n is None:
            n, q = getNormtoo(np.cross(self.e[ni, mi], ez))  # n is perpendicular to e
            # n is perpendicular to z (l is in the x-y plane)
            if q < 1e-5:
                n = getNormvec(np.cross(self.e[ni, mi], ex))  # e || ez   =>	n is perpendicular to x
        if norm1 is None:
            norm1 = np.dot(RotMat1, n)
        if norm2 is None:
            norm2 = n
        self.normcell[ni, mi] = norm1
        self.normsubs[ni, mi] = norm2

    def removelink(self, ni, mi):
        """
        Remove the link between tissue cell ni and substrate cell mi
        :param ni: integer, index of one of the cells involved in the link
        :param mi: integer, index of the second cell
        :return:
        """
        self.islink[ni, mi] = False
        self.Flink[ni, mi], self.Flink_tens[ni, mi] = null, 0
        self.e[ni, mi], self.d[ni, mi] = null, 0
        self.Mcelllink[ni, mi], self.Msubslink[ni, mi] = null, null
        self.tcell[ni, mi], self.tsubs[ni, mi], self.normcell[ni, mi], self.normsubs[ni, mi] = null, null, null, null
        self.d0[ni, mi] = 0
        if not self.saveram:
            self.k[ni, mi] = 0
            self.bend[ni, mi], self.twist[ni, mi] = 0, 0

    def updateDists(self, X):
        """
        Calculate the distances and directions between nodes which are connected by links and save the information in
        self.d (distances) and self.e (normed directions)
        :param X: numpy array of shape (n), containing the positions of the tissue nodes
        for which the calculations should be performed
        :return:
        """
        inds0, inds1 = self.getLinkTuple()
        dX = self.nodesX[inds1] - X[inds0]
        d = scipy.linalg.norm(dX, axis=1)
        self.d[inds0, inds1] = d
        self.e[inds0, inds1] = dX / d[..., None]

    def compactStuffINeed(self):
        """
        Extract the relevant information on existing links from large arrays containing information on all
        hypothetically possible links
        :return: compacted numpy arrays for self.tcell (nl, 3), self.tsubs, self.normcell (nl, 3),
        self.normsubs (nl, 3), self.bend (nl), self.twist (nl), self.k (nl), self.d0 (nl)
        and indices of nodes at ends of links (nl, 2).
        Parantheses indicate shapes of arrays, nl is the number of links
        """
        nodeinds = np.where(self.islink == True)
        nodelen = len(nodeinds[0])
        tcell = self.tcell[nodeinds]
        tsubs = self.tsubs[nodeinds]
        normcell = self.normcell[nodeinds]
        normsubs = self.normsubs[nodeinds]
        if not self.saveram:
            bend = self.bend[nodeinds]
            twist = self.twist[nodeinds]
            k = self.k[nodeinds]
        else:
            bend = self.bend * np.ones((nodelen,))
            twist = self.twist * np.ones((nodelen,))
            k = self.k * np.ones((nodelen,))
        d0 = self.d0[nodeinds]

        return tcell, tsubs, normcell, normsubs, bend, twist, k, d0, nodeinds

    def updateLinkForces(self, PHI, PHIsubs, TCell, TSubs, NormCell, NormSubs, Bend, Twist, K, D0, Nodeinds):
        """
        Update the forces exerted on the links. Input is of shape created by compactStuffINeed()
        :param PHI: Orientation of tissue nodes
        :param PHIsubs: Orientation of substrate nodes
        :param TCell: tangent vectors at tissue cell surfaces
        :param TSubs: tangent vectors at substrate cell surfaces
        :param NormCell: normal vectors at tissue cell surfaces
        :param NormSubs: normal vectors at substrate cell surfaces
        :param Bend: bending rigidities
        :param Twist: twist rigidity
        :param K: Hookean constants
        :param D0: individual link equilibrium lengths
        :param Nodeinds: link indices
        :return:
        """

        E = self.e[Nodeinds]
        D = self.d[Nodeinds]
        # NodesPhiCell = PHI[Nodeinds[0]]
        # NodesPhiSubs = PHIsubs[Nodeinds[1]]

        rotCell = getRotMatArray(PHI[Nodeinds[0]])
        rotSubs = getRotMatArray(PHIsubs[Nodeinds[1]])

        # rotated version of Norm and NormT to fit current setup
        NormCellNow = np.einsum("ijk, ik -> ij", rotCell, NormCell)
        NormSubsNow = np.einsum("ijk, ik -> ij", rotSubs, NormSubs)

        # rotated version of t to fit current setup
        # TCellNow = np.einsum("ijk, ik -> ij", rotCell, TCell)
        # TSubsNow = np.einsum("ijk, ik -> ij", rotSubs, TSubs)

        # calculated new vector \bm{\tilde{n}}_{A, l}
        NormCellTilde = getNormvec(NormCellNow - np.einsum("ij, ij -> i", NormCellNow, E)[:, None] * E)
        NormSubsTilde = getNormvec(NormSubsNow - np.einsum("ij, ij -> i", NormSubsNow, -E)[:, None] * (-E))

        self.Mcelllink[Nodeinds] = Bend[..., None] * np.cross(np.einsum("ijk, ik -> ij", rotCell, TCell), E) + \
                                   Twist[..., None] * np.cross(NormCellTilde, NormSubsTilde)  # Eq 5 for cells

        self.Msubslink[Nodeinds] = Bend[..., None] * np.cross(np.einsum("ijk, ik -> ij", rotSubs, TSubs), -E) + \
                                   Twist[..., None] * np.cross(NormSubsTilde, NormCellTilde)  # Eq 5 for substrate

        M = self.Mcelllink + self.Msubslink
        M = M[Nodeinds]

        # Eqs. 10, 13, 14, 15
        self.Flink_tens[Nodeinds] = (K * (D - D0))
        self.Flink[Nodeinds] = self.Flink_tens[Nodeinds][..., None] * E + np.cross(M, E) / D[:, None]

    def getForces(self, x, tcell, tsubs, normcell, normsubs, bend, twist, k, d0, nodeinds):
        """
        Calculate forces and torques on tissue nodes from tissue-substrate links. Input except for x in shape returned
        by compactStuffINeed()
        :param x: numpy array of shape (3 * 2 * self.N + 3 * self.Nsubs) with positions and orientations of tissue
        nodes and orientations of substrate nodes
        :param tcell: tangent vectors at tissue cell surfaces
        :param tsubs: tangent vectors at substrate cell surfaces
        :param normcell: normal vectors at tissue cell surfaces
        :param normsubs: normal vectors at substrate cell surfaces
        :param bend: bending rigidities
        :param twist: twist rigidity
        :param k: Hookean constants
        :param d0: individual link equilibrium lengths
        :param nodeinds: link indices
        :return: numpy array of shape (3 * 2 * self.N + 3 * self.Nsubs) containing forces and torques on tissue nodes
        and torques on substrate nodes in form readable by solve_ivp
        """
        # reshape X to form readable by class
        X = x.reshape(-1, 3)
        Phi = X[self.N:self.N2, :]
        Phisubs = X[self.N2:, :]
        X = X[:self.N, :]
        self.updateDists(X)
        self.updateLinkForces(Phi, Phisubs, tcell, tsubs, normcell, normsubs, bend, twist, k, d0, nodeinds)
        self.Fnode = np.sum(self.Flink, axis=0)
        self.Mnode = np.sum(self.Msubslink, axis=0)
        return np.concatenate((np.sum(self.Flink, axis=1), np.sum(self.Mcelllink, axis=1), self.Mnode),
                              axis=0).flatten()

    def getLinkList(self):
        """
        Get an array of the indices of the nodes at each end of each link
        :return: numpy array of shape (nl, 2) where nl is the number of links. All links along axis 0, the indices of
        the two nodes connected by the link along axis 1, where the node with the larger index is the first entry
        """
        allLinks0, allLinks1 = np.where(self.islink == True)
        return np.array([[allLinks0[i], allLinks1[i]] for i in range(len(allLinks0))])

    def getLinkTuple(self):
        """
        Get a tuple of the indices of the nodes at the end of each link
        :return: tuple (a, b) of numpy arrays of shape (nl), where nl is the number of links. a[i] and b[i] are the
        nodes at the two ends of the i-th link.
        """
        return np.where(self.islink == True)

    def update_d0(self, dt, force=True):
        """
        Update the equilibrium link length
        :param dt: float, the time taken for the last plasticity step
        :param force: boolean, if False: update done as suggested in czirok2014cell. if True: force-dependent component
        included.
        :return:
        """
        nodeinds = np.where(self.islink == True)
        myd0 = self.d0[nodeinds]

        subsrandom = npr.random(len(nodeinds[0]))

        if force:
            # lognorm fitted to match behavior for d0min==0.8, d0max==2.0 and d0_0==1.0
            myd0 += self.c1 * ((self.Flink_tens[nodeinds]) - self.F_contr) * dt *\
                    0.69 * lognorm.pdf(self.d[nodeinds], .7, loc=.7, scale=.5)

        myd0 += self.c2 * (self.d0_0 - myd0) * dt + self.c3 * sqrt(dt) * (2 * subsrandom - 1)

        self.d0[nodeinds] = myd0


class CellMech:
    def __init__(self, num_cells, num_subs=0, dt=0.01, nmax=300, qmin=0.001, d0_0=1., p_add=1., p_del=0.2, c1=0.05,
                 c2=0.1, c3=0.2, subs_scale=False, p_add_subs=None, p_del_subs=None, chkx=False, d0max=2., dims=3,
                 F_contr=1., isF0=False, isanchor=False, issubs=False, force_contr=True, plasticity=(1., 1., 1.5)):
        """
        Implementation of model for cell-resolved, multiparticle model of plastic tissue deformations and morphogenesis
        first suggested by Czirok et al in 2014 (https://iopscience.iop.org/article/10.1088/1478-3975/12/1/016005/meta,
        henceforward referred to as czirok2014cell) and first implemented in https://github.com/aczirok/cellmech

        Code contains extensions by Moritz Zeidler, Dept. for Innovative Methods of Computing (IMC),
        Centre for Information Services and High Performance Computing (ZIH) at Technische Universitaet Dresden.
        Contact via moritz.zeidler at tu-dresden.de or andreas.deutsch at tu-dresden.de

        Initialize instance of class CellMech, which serves as the overlying class for simulations. Also initializes
        instances of class NodeConfiguration (always) and  SubsConfiguration (when issubs==True or
        issubs=="lonesome")

        :param num_cells: integer, the number of tissue cells
        :param num_subs: integer, the number of substrate cells
        :param dt: float, the time unit for scaling simulation time
        :param nmax: integer, the maximum time (in simulation time) for until cutoff when calculating
            mechanical equilibrium
        :param qmin: float, the square of the maximum force per cell until mechanical equilibration is cut off
        :param d0_0: float, the global equilibrium link length (d_0 in czirok2014cell)
        :param p_add: float, base probability for adding tissue-tissue links
        :param p_del: float, base probability for removing tissue-tissue links
        :param c1: float, constant of contractility/volume exlusion
        :param c2: float, constant of tissue elasticity
        :param c3: float, the variance of the noise in updating the link lengths is 2 * (c2 ** 2) * dt were dt is the
            time used for modifying a link
        :param p_add_subs: float, base probability for adding tissue-substrate links
        :param p_del_subs: float, base probability for removing tissue-substrate links
        :param chkx: bool, whether or not to check for link crossings (only functional for dims==2
        :param d0max: float, maximum cell-cell distance to allow a link to be added
        :param dims: 2 or 3, the number of dimensions of the simulations
        :param F_contr: target force for force-dependent update of equilibrium lengths
        :param isF0: bool, whether or not external forces are a part of the problem
        :param isanchor: bool, whether or not tissue cells are anchored to a x0-position
        :param issubs: True (with substrate), False (without substrate) or "lonesome" (with substrate but only one
            tissue cell)
        :param force_contr: boolean, if False: update done as suggested in czirok2014cell. if True: force-dependent
            component included.
        :param plasticity: Either None if Hookean, bend and twist constants are set individually per link, or tuple
            containing global values for the three constants in shape (k1, k2, k3) = (bend, twist, Hooke)
        """
        self.dims = dims
        self.issubs = issubs
        # variables to store cell number and cell positions and angles
        self.N = num_cells
        self.N2 = 2 * self.N
        self.N_inv = 1. / self.N

        # parameters for mechanical equilibration
        self.dt = dt
        self.nmax = nmax
        self.tmax = nmax * dt
        self.qmin = np.sqrt(qmin)

        # parameters to add/remove links
        self.d0_0 = d0_0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        if self.dims == 2:
            self.chkx = chkx
        elif self.dims == 3:
            self.chkx = False
        self.d0max = d0max

        self.force_contr = force_contr

        # stuff for documentation
        self.snaptimes = []  # stores the simulation timesteps
        self.lastt = 0
        self.nsaves = 0

        # initialize instance of NodeConfiguration containing data on tissue cells
        self.mynodes = NodeConfiguration(num=num_cells, num_subs=num_subs, p_add=p_add, p_del=p_del,
                                         c1=c1, c2=c2, c3=c3, F_contr=F_contr,
                                         dims=dims, d0_0=d0_0, isF0=isF0, isanchor=isanchor, plasticity=plasticity)

        if self.issubs is True:
            # initialize instance of SubsConfiguration containing data on substrate cells, set functions to account for
            # substrate when calculating forces and saving steps
            if p_add_subs is None:
                p_add_subs = p_add
            if p_del_subs is None:
                p_del_subs = p_del
            if subs_scale is False:
                self.mysubs = SubsConfiguration(num_cells=num_cells, num_subs=num_subs, d0_0=d0_0,
                                                c1=c1, c2=c2, c3=c3, F_contr=F_contr,
                                                p_add=p_add_subs, p_del=p_del_subs, plasticity=plasticity)
            else:
                subsplasticity = (plasticity[0] / subs_scale, plasticity[1] / subs_scale, plasticity[2] / subs_scale)
                self.mysubs = SubsConfiguration(num_cells=num_cells, num_subs=num_subs, d0_0=d0_0,
                                                c1=c1, c2=c2, c3=c3, F_contr=F_contr,
                                                p_add=p_add_subs, p_del=p_del_subs*subs_scale,
                                                plasticity=subsplasticity)
            self.mechEquilibrium = lambda: self.mechEquilibrium_withsubs()
            self.makesnap = lambda t: self.makesnap_withsubs(t)
            self.addLinkList = lambda: self.addLinkList_withsubs()

        elif self.issubs is False:
            # set functions to ignore substrate when calculating forces and saving steps
            self.mechEquilibrium = lambda: self.mechEquilibrium_nosubs()
            self.makesnap = lambda t: self.makesnap_nosubs(t)
            self.addLinkList = lambda: self.addLinkList_nosubs()

        elif self.issubs is "lonesome":
            # initialize instance of SubsConfiguration containing data on substrate cells, set functions to account for
            # substrate when calculating forces and saving steps but simplify tissue behavior for one tissue cell only
            if p_add_subs is None:
                p_add_subs = p_add
            if p_del_subs is None:
                p_del_subs = p_del
            self.mysubs = SubsConfiguration(num_cells=num_cells, num_subs=num_subs, d0_0=d0_0,
                                            c1=c1, c2=c2, c3=c3, F_contr=F_contr,
                                            p_add=p_add_subs, p_del=p_del_subs, plasticity=plasticity)
            self.mechEquilibrium = lambda: self.mechEquilibrium_lonesome()
            self.makesnap = lambda t: self.makesnap_lonesome(t)
            self.addLinkList = lambda: self.addLinkList_lonesome()
        else:
            # catch incorrect choice of issubs
            print("I don't know that type of subs")
            sys.exit()

    def mechEquilibrium_nosubs(self):
        """
        Wrapping for calculating mechanical equilibrium in absence of substrate. Uses slightly modified version of
        scipy.integrate.solve_ivp (exact location of event==0 isn't searched) with method 'LSODA'.
        Integration ends if each component of the force on each cell drops lower than self.qmin,
        or when self.tmax is reached.
        :return: Time needed for mechanical equilibration
        """
        # reshape X and Phi for solveivp
        x = np.concatenate((self.mynodes.nodesX, self.mynodes.nodesPhi), axis=0).flatten()
        # extract data not changed by mechanical equilibrium from large arrays
        t, norm, normT, bend, twist, k, d0, nodeinds = self.mynodes.compactStuffINeed()

        # produce fun for solve_ivp as lambda
        def notatallfun(temp, y): return self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds)

        # produce event function to check whether to end solve_ivp
        def event(temp, y):
            k1 = self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds)
            return np.max(np.abs(k1) - self.qmin)
        event.terminal = True
        event.direction = -1

        # perform equilibration
        res = solve_ivp(fun=notatallfun, t_span=[0, self.tmax], y0=x, method='LSODA', events=[event], atol=1e-3)

        # reshape data returned by solve_ivp to data readable by class
        x = res.y.reshape((-1, 3, len(res.t)))
        self.mynodes.nodesX = x[:self.N, :, -1]
        self.mynodes.nodesPhi = x[self.N:, :, -1]

        return res.t[-1]

    def mechEquilibrium_withsubs(self):
        """
        Wrapping for calculating mechanical equilibrium in presence of substrate. Uses slightly modified version of
        scipy.integrate.solve_ivp (exact location of event==0 isn't searched) with method 'LSODA'.
        Integration ends if each component of the force on each cell drops lower than self.qmin,
        or when self.tmax is reached.
        :return: Time needed for mechanical equilibration
        """
        # reshape X and Phi for solveivp
        x = np.concatenate((self.mynodes.nodesX, self.mynodes.nodesPhi, self.mysubs.nodesPhi), axis=0).flatten()
        # extract data not changed by mechanical equilibrium from large arrays
        t, norm, normT, bend, twist, k, d0, nodeinds = self.mynodes.compactStuffINeed()
        tcell, tsubs, normcell, normsubs, bends, twists, ks, d0s, nodeindss = self.mysubs.compactStuffINeed()

        # produce fun for solve_ivp as lambda
        def notatallfun(temp, y): return self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds) + \
                                         self.mysubs.getForces(y, tcell, tsubs, normcell, normsubs,
                                                               bends, twists, ks, d0s, nodeindss)

        # produce event function to check whether to end solve_ivp
        def event(temp, y):
            k1 = self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds) + \
                 self.mysubs.getForces(y, tcell, tsubs, normcell, normsubs, bends, twists, ks, d0s, nodeindss)
            return np.max(np.abs(k1[:self.N2]) - self.qmin)
        event.terminal = True
        event.direction = -1

        # perform equilibration
        res = solve_ivp(fun=notatallfun, t_span=[0, self.tmax], y0=x, method='LSODA', events=[event], atol=1e-3)

        # reshape data returned by solve_ivp to data readable by class
        x = res.y.reshape((-1, 3, len(res.t)))
        self.mynodes.nodesX = x[:self.N, :, -1]
        self.mynodes.nodesPhi = x[self.N:self.N2, :, -1]
        self.mysubs.nodesPhi = x[self.N2:, :, -1]
        return res.t[-1]

    def mechEquilibrium_lonesome(self):
        """
        Wrapping for calculating mechanical equilibrium in presence of substrate with only one tissue cell.
        Uses slightly modified version of  scipy.integrate.solve_ivp (exact location of event==0 isn't searched)
        with method 'LSODA'. Integration ends if each component of the force on each cell drops lower than self.qmin,
        or when self.tmax is reached.
        :return: Time needed for mechanical equilibration
        """
        # reshape X and Phi for solveivp
        x = np.concatenate((self.mynodes.nodesX, self.mynodes.nodesPhi, self.mysubs.nodesPhi), axis=0).flatten()
        # extract data not changed by mechanical equilibrium from large arrays
        tcell, tsubs, normcell, normsubs, bends, twists, ks, d0s, nodeindss = self.mysubs.compactStuffINeed()

        # produce fun for solve_ivp as lambda
        def notatallfun(temp, y): return self.mysubs.getForces(y, tcell, tsubs, normcell, normsubs,
                                                               bends, twists, ks, d0s, nodeindss)

        # produce event function to check whether to end solve_ivp
        def event(temp, y):
            k1 = self.mysubs.getForces(y, tcell, tsubs, normcell, normsubs, bends, twists, ks, d0s, nodeindss)
            return np.max(np.abs(k1[:self.N2]) - self.qmin)
        event.terminal = True
        event.direction = -1

        # perform equilibration
        res = solve_ivp(fun=notatallfun, t_span=[0, self.tmax], y0=x, method='LSODA', events=[event], atol=1e-3)

        # reshape data returned by solve_ivp to data readable by class
        x = res.y.reshape((-1, 3, len(res.t)))
        self.mynodes.nodesX = x[:self.N, :, -1]
        self.mynodes.nodesPhi = x[self.N:self.N2, :, -1]
        self.mysubs.nodesPhi = x[self.N2:, :, -1]
        return res.t[-1]

    def intersect_all(self):
        """
        Find intersections in  current 2-d configuration of cell positions saved in subclass self.mynodes.
        :return: numpy array of shape (n, 2, 2) for n intersections of links. First axis: intersections. Second axis:
        the two intersecting links. Third axis: the indices of the cells connected by the link.
        """
        allLinks0, allLinks1 = self.mynodes.getLinkTuple()

        A = self.mynodes.nodesX[allLinks0][:, None, :]
        B = self.mynodes.nodesX[allLinks1][:, None, :]
        C = self.mynodes.nodesX[allLinks0][None, ...]
        D = self.mynodes.nodesX[allLinks1][None, ...]

        mynrm1 = scipy.linalg.norm(A - C, axis=2)
        mynrm2 = scipy.linalg.norm(A - D, axis=2)
        mynrm3 = scipy.linalg.norm(B - C, axis=2)

        distbool1, distbool2, distbool3 = np.greater(mynrm1, 0.01), np.greater(mynrm2, 0.01), np.greater(mynrm3, 0.01)

        distbool = np.logical_and(distbool1, distbool2)
        distbool = np.logical_and(distbool, distbool3)

        ccw1 = ccw(A, C, D)
        ccw2 = ccw(B, C, D)
        ccw3 = ccw(A, B, C)
        ccw4 = ccw(A, B, D)

        ccwa = np.not_equal(ccw1, ccw2)
        ccwb = np.not_equal(ccw3, ccw4)

        ccwf = np.logical_and(ccwa, ccwb)

        finalbool = np.triu(ccwf)
        finalbool[np.where(distbool == False)] = False

        clashinds0, clashinds1 = np.where(finalbool == True)
        clashlinks = np.array([[[allLinks0[clashinds0[i]], allLinks1[clashinds0[i]]],
                                [allLinks0[clashinds1[i]], allLinks1[clashinds1[i]]]] for i in range(len(clashinds0))])

        return clashlinks

    def intersect_withone(self, n1, n2):
        """
        Check whether a hypothetical new link connecting cells n1 and n2 would intersect with another link of the
        current 2-d-configuration saved in subclass self.mynodes.
        :param n1: The index of one of the cells connected by the hypothetical new link
        :param n2: The index of the second cell
        :return: bool True (the new link would intersect another one) or False (it wouldn't)
        """
        allLinks0, allLinks1 = self.mynodes.getLinkTuple()

        A = self.mynodes.nodesX[n1][None, :]
        B = self.mynodes.nodesX[n2][None, :]
        C = self.mynodes.nodesX[allLinks0]
        D = self.mynodes.nodesX[allLinks1]

        mynrm1 = scipy.linalg.norm(A - C, axis=1)
        mynrm2 = scipy.linalg.norm(A - D, axis=1)
        mynrm3 = scipy.linalg.norm(B - C, axis=1)

        distbool1, distbool2, distbool3 = np.greater(mynrm1, 0.01), np.greater(mynrm2, 0.01), np.greater(mynrm3, 0.01)

        distbool = np.logical_and(distbool1, distbool2)
        distbool = np.logical_and(distbool, distbool3)

        ccw1 = ccw(A, C, D)
        ccw2 = ccw(B, C, D)
        ccw3 = ccw(A, B, C)
        ccw4 = ccw(A, B, D)

        ccwa = np.not_equal(ccw1, ccw2)
        ccwb = np.not_equal(ccw3, ccw4)

        finalbool = np.logical_and(ccwa, ccwb)

        finalbool[np.where(distbool == False)] = False

        if True in finalbool:
            return True
        else:
            return False

    def checkLinkX(self):
        """
        Find and delete intersections in current 2-d-configuration of tissue cells
        :return: Nothing
        """
        delete_list = []
        Xs = self.intersect_all()
        while len(Xs) > 0:
            Xsflat = np.array(Xs).reshape(2 * len(Xs), 2)
            # u: unique elements in Xsflat (a.k.a. links), count: number of occurences in Xsflat
            u, count = np.unique(Xsflat, axis=0, return_counts=True)
            badlink = u[np.argmax(count)]
            delete_list.append(badlink)
            newXs = []
            for linkpair in Xs:
                if (badlink == linkpair[0]).all() or (badlink == linkpair[1]).all():
                    continue
                newXs.append(linkpair)
            Xs = newXs
        for badlink in delete_list:
            self.mynodes.removelink(badlink[0], badlink[1])

    def delLinkList(self):
        """
        Identify possible links for deletion
        :return: numpy array of shape (3, n). Along first axis: (a) array of n possible links for deletion, each entry
        is a numpy array of shape (2) containing the indices of the cells connected by the link. (b) array of n deletion
        probabilities, calculated according to eq. 20 of czirok2014cell. (c) array of n boolean variables, True if
        tissue-substrate link, False if tissue-tissue link. Returns no deletable links if there is exactly one
        tissue-substrate link (case relevant for "lonesome" setting of self.issubs).
        """
        del_links, del_probs, del_bools = [], [], []
        linksum = 0
        if self.issubs:
            linklist = self.mysubs.getLinkList()
            linksum += len(linklist)
            for link in linklist:
                if self.mysubs.d[link[0], link[1]] < self.mysubs.d0[link[0], link[1]]:
                    continue        # compressed links are stable
                f = scipy.linalg.norm(self.mysubs.Flink[link[0], link[1]])
                p = exp(f)
                del_links.append(link)
                del_probs.append(p * self.mysubs.p_del)
                del_bools.append(True)  # is tissue-substrate link
        linklist = self.mynodes.getLinkList()
        linksum += len(linklist)
        if linksum == 1:
            return [[], [], []]  # catch case where there is only one tissue-substrate link ("lonesome" setting)
        for link in linklist:
            if self.mynodes.d[link[0], link[1]] < self.mynodes.d0[link[0], link[1]]:
                continue            # compressed links are stable
            f = scipy.linalg.norm(self.mynodes.Flink[link[0], link[1]])
            p = exp(f)
            del_links.append(link)
            del_probs.append(p * self.mynodes.p_del)
            del_bools.append(False)  # is tissue-tissue link
        return np.array([del_links, del_probs, del_bools])

    def tryLink_notsubs(self, n1, n2):
        """
        Test whether a hypothetical new tissue-tissue link (a) already exists, (b) would intersect another already
        existing link (only in 2-d) or (c) would be longer than self.d0max
        :param n1: The index of one of the cells connected by the hypothetical new link
        :param n2: The index of the second cell
        :return: actual length of hypothetical link
        """
        if self.mynodes.islink[n1, n2]:
            return -1  # link refused
        if self.dims == 2:
            if self.intersect_withone(n1, n2):
                return -1  # link refused
        d = scipy.linalg.norm(self.mynodes.nodesX[n1] - self.mynodes.nodesX[n2])
        if d > self.d0max:
            return -1  # link refused
        return d  # link accepted: d > 0

    def tryLink_issubs(self, n1, n2):
        """
        Test whether a hypothetical new tissue-substrate link (a) already exists, (b) would intersect another already
        existing link (only in 2-d) or (c) would be longer than self.d0max
        :param n1: The index of one of the cells connected by the hypothetical new link
        :param n2: The index of the second cell
        :return: actual length of hypothetical link
        """
        if self.mysubs.islink[n1, n2]:
            return -1  # link refused
        if self.dims == 2:
            if self.intersect_withone(n1, n2):
                return -1  # link refused
        d = scipy.linalg.norm(self.mynodes.nodesX[n1] - self.mysubs.nodesX[n2])
        if d > self.d0max:
            return -1  # link refused
        return d  # link accepted: d > 0

    def addLinkList_nosubs(self):
        """
        Identify possible new links in the case of simulations without a substrate. New links based on Voronoi
        tessellation of the tissue nodes.
        :return: numpy array of shape (3, n). Along first axis: (a) array of n possible new links, each entry
        is a tuple containing the indices of the cells connected by the link. (b) array of n deletion
        probabilities, calculated according to eq. 21 of czirok2014cell. (c) array of n boolean variables set to False,
        as all possible links are tissue-tissue links
        """
        add_links, add_probs, add_bools = [], [], []
        for i, j in VoronoiNeighbors(self.mynodes.nodesX, vodims=self.dims):
            d = self.tryLink_notsubs(i, j)
            if d > 1e-5:  # if d < 0: link rejected by tryLink
                p = (1 - (d / self.d0max))
                add_links.append((i, j))
                add_probs.append(p * self.mynodes.p_add)
                add_bools.append(False)
        return np.array([add_links, add_probs, add_bools])

    def addLinkList_withsubs(self):
        """
        Identify possible new links in the case of simulations with a substrate. New links based on Voronoi
        tessellation of all tissue and substrate nodes.
        :return: numpy array of shape (3, n). Along first axis: (a) array of n possible new links, each entry
        is a tuple containing the indices of the nodes connected by the link. Substrate nodes are numbered as
        continuation of tissue-node-list. (b) array of n deletion probabilities, calculated according to eq. 21 of
        czirok2014cell. (c) array of n boolean variables, True if tissue-substrate link, False if tissue-tissue link.
        """
        add_links, add_probs, add_bools = [], [], []
        allnodes = np.concatenate((self.mynodes.nodesX, self.mysubs.nodesX))
        for i, j in VoronoiNeighbors(allnodes, vodims=self.dims):
            if j >= self.N:  # at least one node is a substrate node
                boo = True
                if i < self.N:
                    d = self.tryLink_issubs(i, j - self.N)
                    boo = True
                else:
                    d = -1  # is substrate-substrate link
            else:  # both
                d = self.tryLink_notsubs(i, j)
                boo = False
            if d > 1e-5:  # if d < 0: link either substrate-substrate link or rejected by tryLink
                p = (1 - (d / self.d0max))
                add_links.append((i, j))
                if boo:
                    add_probs.append(p * self.mysubs.p_add)
                else:
                    add_probs.append(p * self.mynodes.p_add)
                add_bools.append(boo)
        return np.array([add_links, add_probs, add_bools])

    def addLinkList_lonesome(self):
        """
        Identify possible new links in the case of simulations with a substrate but only one tissue cell.
        New links based on Voronoi tessellation of the tissue node and all substrate nodes.
        :return: numpy array of shape (3, n). Along first axis: (a) array of n possible new links, each entry
        is a tuple containing the indices of the nodes connected by the link. Substrate nodes are numbered as
        continuation of tissue-node-list. (b) array of n deletion probabilities, calculated according to eq. 21 of
        czirok2014cell. (c) array of n boolean variables set to True, as all possible links are tissue-substrate links
        """
        add_links, add_probs, add_bools = [], [], []
        for i in range(self.N):
            for j in range(self.mysubs.Nsubs):
                d = self.tryLink_issubs(i, j)
                if d > 1e-5:   # if d < 0: link rejected by tryLink
                    p = 1 - (d / self.d0max)
                    add_links.append((i, j + self.N))
                    add_probs.append(p * self.mysubs.p_add)
                    add_bools.append(True)
        return np.array([add_links, add_probs, add_bools])

    def pickEvent(self, to_del, to_add):
        """
        Decide on next plasticity step, whether to add a link, delete one or do nothing. Decision based on Gillespie
        algorithm
        :param to_del: numpy array returned by self.delLinkList()
        :param to_add: numpy array returned by self.addLinkList()
        :return: time taken up by plasticity step
        """
        l_del, p_del, boo_del = to_del
        s1 = np.sum(p_del)

        l_add, p_add, boo_add = to_add
        s2 = np.sum(p_add)

        S = s1 + s2  # norm for probabilities
        if S < 1e-7:
            print("nothing to do!")
            return 1.
        dt = -log(npr.random()) / S
        if dt > 1:
            # print 'Must adjust d0 variables before the next event!'
            return 1.

        r = S * npr.random()
        if r < s1:  # we will remove a link
            R = r - np.cumsum(p_del)  # find root in s1 - \sum\limits_{i=0}^{n}p_del_n
            ni = np.where(R < 0)[0][0]
            if not boo_del[ni]:  # link to be removed is tissue-tissue link
                self.mynodes.removelink(l_del[ni][0], l_del[ni][1])
                return dt
            else:  # link to be removed is tissue-substrate link
                self.mysubs.removelink(l_del[ni][0], l_del[ni][1])
                return dt

        r = r - s1
        if r < s2:  # we will add a link
            R = r - np.cumsum(p_add)    # find root in s1 - \sum\limits_{i=0}^{n}p_del_n
            ni = np.where(R < 0)[0][0]
            if not boo_add[ni]:  # link to be removed is tissue-tissue link
                self.mynodes.addlink(l_add[ni][0], l_add[ni][1])
                return dt
            else:  # link to be added is tissue-substrate link
                n1 = l_add[ni][0]
                n2 = l_add[ni][1]
                self.mysubs.addlink(n1, n2 - self.N, self.mynodes.nodesX[n1], self.mynodes.nodesPhi[n1])
                return dt

    def modlink(self):
        """
        Perform a plasticity event (add or delete a link)
        :return: Time taken up by the plasticity event
        """
        if self.chkx:
            self.checkLinkX()
        to_del = self.delLinkList()
        to_add = self.addLinkList()
        dt = self.pickEvent(to_del, to_add)
        self.mynodes.update_d0(dt, force=self.force_contr)
        if self.issubs is not False:
            self.mysubs.update_d0(dt, force=self.force_contr)
        return dt

    def makesnap_nosubs(self, t):
        """
        Save important data of current configuration (tissue node positions, Forces on tissue nodes, List of
        tissue-tissue links, forces on those links, current time in simulation run) for simulations without substrate
        :param t: float, current time in simulation run
        :return: Nothing
        """
        self.mynodes.nodesnap.append(self.mynodes.nodesX.copy())
        self.mynodes.fnodesnap.append(self.mynodes.Fnode.copy())
        linkList = self.mynodes.getLinkList()
        self.mynodes.linksnap.append(linkList)
        self.mynodes.flinksnap.append(self.mynodes.Flink[linkList[..., 0], linkList[..., 1]])
        self.snaptimes.append(t)

    def makesnap_withsubs(self, t):
        """
        Save important configuration data (node positions, Forces on nodes, List of links, forces on those links,
        current time in simulation run) for simulations with substrate
        :param t: float, current time in simulation run
        :return: Nothing
        """
        self.mynodes.nodesnap.append(self.mynodes.nodesX.copy())
        self.mynodes.fnodesnap.append(self.mynodes.Fnode.copy())
        self.mysubs.fnodesnap.append(self.mysubs.Fnode.copy())
        linkList = self.mynodes.getLinkList()
        self.mynodes.linksnap.append(linkList)
        self.mynodes.flinksnap.append(self.mynodes.Flink[linkList[..., 0], linkList[..., 1]])
        linkList = self.mysubs.getLinkList()
        self.mysubs.linksnap.append(linkList)
        self.mysubs.flinksnap.append(-self.mysubs.Flink[linkList[..., 0], linkList[..., 1]])
        self.snaptimes.append(t)

    def makesnap_lonesome(self, t):
        """
        Save important configuration data (node positions, Forces on nodes, List of tissue-substrate links,
        forces on those links, current time in simulation run) for simulations with substrate
        :param t: float, current time in simulation run
        :return: Nothing
        """
        self.mynodes.nodesnap.append(self.mynodes.nodesX.copy())
        self.mynodes.fnodesnap.append(self.mynodes.Fnode.copy())
        self.mysubs.fnodesnap.append(self.mysubs.Fnode.copy())
        linkList = self.mysubs.getLinkList()
        self.mysubs.linksnap.append(linkList)
        self.mysubs.flinksnap.append(-self.mysubs.Flink[linkList[..., 0], linkList[..., 1]])
        self.snaptimes.append(t)

    def saveonesnap(self, savewhat, savedir, savelist):
        """
        Save current snapshot information of one type to disk
        :param savewhat: str, what to save, used as name for directory to save snapshots
        :param savedir: str, name of directory holding all simulation data
        :param savelist: list containing snapshots
        :return: empty list
        """
        nstr = str(self.nsaves).zfill(3)
        if not os.path.isdir("./" + savedir + "/" + savewhat):
            os.mkdir("./" + savedir + "/" + savewhat)
        np.save(savedir + "/" + savewhat + "/" + nstr, savelist)
        return []

    def savedata(self, savedir="res", savenodes_r=True, savelinks=True, savenodes_f=True, savelinks_f=True, savet=True,
                 savephi=True, savetang=True, savenorm=True, saved0=True):
        """
        Save important configuration data to disk and clear snapshots in memory
        :param savedir: string, name of directory to save in (based on current working directory)
        :param savenodes_r: boolean, whether to save node positions
        :param savelinks: boolean, whether to save links
        :param savenodes_f: boolean, whether to save forces on nodes
        :param savelinks_f: boolean, whether to save forces on links
        :param savet: boolean, whether to save timesteps
        :param savephi: boolean, whether to save last phi configuration (only needed in case of later relaunch)
        :param savetang: boolean, whether to save last t (only needed in case of later relaunch)
        :param savenorm: boolean, whether to save last norm (only needed in case of later relaunch)
        :param saved0: boolean, whether to save last d0 values (only needed in case of later relaunch)
        :return:
        """
        linklist = np.where(self.mynodes.islink == True)
        if not os.path.isdir("./" + savedir):
            os.mkdir("./" + savedir)
        if savenodes_r:
            self.mynodes.nodesnap = self.saveonesnap("nodesr", savedir, self.mynodes.nodesnap)
        if savenodes_f:
            self.mynodes.fnodesnap = self.saveonesnap("nodesf", savedir, self.mynodes.fnodesnap)
        if savelinks:
            self.mynodes.linksnap = self.saveonesnap("links", savedir, self.mynodes.linksnap)
        if savelinks_f:
            self.mynodes.flinksnap = self.saveonesnap("linksf", savedir, self.mynodes.flinksnap)
        if savet:
            self.lastt = self.snaptimes[-1]
            self.snaptimes = self.saveonesnap("ts", savedir, self.snaptimes)
        if savephi:
            np.save(savedir + "/phi", self.mynodes.nodesPhi)
        if savetang:
            np.save(savedir + "/tang", self.mynodes.t[linklist])
        if savenorm:
            np.save(savedir + "/norm", self.mynodes.norm[linklist])
        if saved0:
            np.save(savedir + "/d0", self.mynodes.d0[linklist])

        if self.issubs:
            linklist = np.where(self.mysubs.islink == True)
            if savenodes_r:
                np.save(savedir + "/subsnodesr", self.mysubs.nodesX)
            if savenodes_f:
                self.mysubs.fnodesnap = self.saveonesnap("subsnodesf", savedir, self.mysubs.fnodesnap)
            if savelinks:
                self.mysubs.linksnap = self.saveonesnap("subslinks", savedir, self.mysubs.linksnap)
            if savelinks_f:
                self.mysubs.flinksnap = self.saveonesnap("subslinksf", savedir, self.mysubs.flinksnap)
            if savephi:
                np.save(savedir + "/subsphi", self.mysubs.nodesPhi)
            if savetang:
                np.save(savedir + "/substcell", self.mysubs.tcell[linklist])
                np.save(savedir + "/substsubs", self.mysubs.tsubs[linklist])
            if savenorm:
                np.save(savedir + "/subsnormcell", self.mysubs.normcell[linklist])
                np.save(savedir + "/subsnormsubs", self.mysubs.normsubs[linklist])
            if saved0:
                np.save(savedir + "/subsd0", self.mysubs.d0[linklist])

        self.nsaves += 1

    def cleanonesave(self, savewhat, savedir):
        """
        Delete a directory holding temporary snapshot files and combine them into one .npy file
        :param savewhat: str, type of data to be saved, is also the name of the directory in savedir containing the data
            and the name of the .npy file which will hold the combined data
        :param savedir: str, directory containing all simulation results
        :return:
        """
        templist = []
        savestr = savedir + "/" + savewhat
        for i in range(self.nsaves):
            nstr = savestr + "/" + str(i).zfill(3) + ".npy"
            templist += list(np.load(nstr,allow_pickle=True))
        np.save(savestr, templist)
        del templist
        shutil.rmtree(savestr)

    def cleansaves(self, savedir="res", savenodes_r=True, savelinks=True, savenodes_f=True, savelinks_f=True,
                   savet=True):
        """
        Clean up temporary directories, combine temporary .npy files into one .npy file each
        :param savedir: string, name of directory to save in (based on current working directory)
        :param savenodes_r: boolean, whether node positions where saved
        :param savelinks: boolean, whether links where saved
        :param savenodes_f: boolean, whether forces on nodes where saved
        :param savelinks_f: boolean, whether forces on links where saved
        :param savet: boolean, whether timesteps where saved
        :return:
        """
        if savenodes_r:
            self.cleanonesave("nodesr", savedir)
        if savenodes_f:
            self.cleanonesave("nodesf", savedir)
        if savelinks:
            self.cleanonesave("links", savedir)
        if savelinks_f:
            self.cleanonesave("linksf", savedir)
        if savet:
            self.cleanonesave("ts", savedir)

        if self.issubs:
            if savenodes_f:
                self.cleanonesave("subsnodesf", savedir)
            if savelinks:
                self.cleanonesave("subslinks", savedir)
            if savelinks_f:
                self.cleanonesave("subslinksf", savedir)

    def timeevo(self, tmax, isinit=True, isfinis=True, record=True, progress=True, dtrec=0,
                savedata=True, savedir="res", dtsave=None):
        """
        Perform simulation run with alternating steps of mechanical equilibration and plasticity
        :param tmax: Maximum time for simulation run
        :param isinit: boolean, whether this is the first segment of a simulation run
        :param isfinis: boolean, whether this is the final segment of a simulation run (the simulation can still
            be continued otherwise, but a new class instance should then be initiated with relaunch_CellMech())
        :param record: boolean, whether to save simulation data for after code has finished
        :param progress: show progress bar
        :param dtrec: float, snapshot will be made of config after every tissue plasticity step if dtsave==0, otherwise
            each time t has crossed a new n*dtrec line
        :param savedata: boolean, whether to write the data to the drive (make sure that record==True)
        :param savedir: string, name of the directory for saving the data
        :param dtsave: float, snapshot will be written to drive after each time t has crossed a new n*dtsave line;
            or None, in that case data will only be written after tmax
        :return:
        """

        # pre-production

        if isinit:
            myrandom = npr.random((self.mynodes.randomlength,))
            self.mynodes.randomsummand[self.mynodes.lowers] = myrandom
            self.mynodes.randomsummand.T[self.mynodes.lowers] = myrandom
            self.mynodes.d0 += 0.04 * self.mynodes.randomsummand
            t = 0
            if record:
                self.makesnap(t)
        else:
            if record:
                t = self.lastt
                tmax += t
            else:
                t = 0
        tlast_rec = t
        tlast_save = t
        if dtsave is None:
            dtsave = tmax

        # main loop

        while t < tmax:
            dt = self.mechEquilibrium()
            t += dt
            dt = self.modlink()
            t += dt
            if record and (t - tlast_rec > dtrec or t > tmax):
                self.makesnap(t)
                if dtrec != 0:
                    tlast_rec = t - t % dtrec
            if record and savedata and (t - tlast_save > dtsave or t > tmax):
                self.savedata(savedir)
                tlast_save = t - t % dtsave
            if progress:
                update_progress(t / tmax)

        # post-production

        if record and savedata and isfinis:
            self.cleansaves(savedir)

    def oneequil(self):
        """
        Perform simulation run with only one step of mechanical equilibration and no tissue plasticity for a setup
        without a substrate
        :return: Tuple of Node positions for all timesteps (numpy array), link configurations for each timestep
        (unchanged, numpy array with identical entries along first axis), None, None, timesteps (numpy array)
        """
        linkList = self.mynodes.getLinkList()
        # reshape X and Phi for solveivp
        x = np.concatenate((self.mynodes.nodesX, self.mynodes.nodesPhi), axis=0).flatten()
        t, norm, normT, bend, twist, k, d0, nodeinds = self.mynodes.compactStuffINeed()

        # produce fun for solve_ivp as lambda
        def notatallfun(temp, y): return self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds)

        # produce event function to check whether to end solve_ivp
        def event(temp, y):
            k1 = self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds)
            return np.max(np.abs(k1) - self.qmin)
        event.terminal = True
        event.direction = -1

        # perform equilibration
        res = solve_ivp(fun=notatallfun, t_span=[0, self.tmax], y0=x, events=[event], method="LSODA", atol=1e-3)

        # reshape data returned by solve_ivp to date readable by class and save it in appropriate places
        x = res.y.reshape((-1, 3, len(res.t)))
        self.snaptimes = res.t
        self.mynodes.nodesnap = np.transpose(x[:self.N, :, :], axes=(2, 0, 1))
        self.mynodes.nodesX = self.mynodes.nodesnap[-1]
        self.mynodes.linksnap = np.tile(linkList, (len(self.snaptimes), 1, 1))
        return self.mynodes.nodesnap, self.mynodes.linksnap, None, None, self.snaptimes

    def oneequil_withsubs(self):
        """
        Perform simulation run with only one step of mechanical equilibration and no tissue plasticity for a setup
        with a substrate
        :return: Tuple Node positions for all timesteps (numpy array), link configurations for each timestep (unchanged,
        numpy array with identical entries along first axis), None, None, timesteps (numpy array)
        """
        linkList = self.mynodes.getLinkList()
        # reshape X and Phi for solveivp
        x = np.concatenate((self.mynodes.nodesX, self.mynodes.nodesPhi, self.mysubs.nodesPhi), axis=0).flatten()
        t, norm, normT, bend, twist, k, d0, nodeinds = self.mynodes.compactStuffINeed()
        tcell, tsubs, normcell, normsubs, bends, twists, ks, d0s, nodeindss = self.mysubs.compactStuffINeed()

        # produce fun for solve_ivp as lambda
        def notatallfun(temp, y): return self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds) + \
                                         self.mysubs.getForces(y, tcell, tsubs, normcell, normsubs,
                                                               bends, twists, ks, d0s, nodeindss)

        # produce event function to check wether to end solve_ivp
        def event(temp, y):
            k1 = self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds) + \
                 self.mysubs.getForces(y, tcell, tsubs, normcell, normsubs, bends, twists, ks, d0s, nodeindss)
            return np.max(np.abs(k1) - self.qmin)
        event.terminal = True
        event.direction = -1

        res = solve_ivp(fun=notatallfun, t_span=[0, self.tmax], y0=x, events=[event], method="LSODA", atol=1e-3)
        # reshape data returned by solve_ivp to date readable by class and save it in appropriate places
        x = res.y.reshape((-1, 3, len(res.t)))
        self.snaptimes = res.t
        self.mynodes.nodesnap = np.transpose(x[:self.N, :, :], axes=(2, 0, 1))
        return self.mynodes.nodesnap, np.tile(linkList, (len(self.snaptimes), 1, 1)), None, None, self.snaptimes
