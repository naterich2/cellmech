#!/usr/bin/python  -u

from __future__ import division

import sys
import warnings
from math import exp, log, sqrt

import numpy as np
import numpy.random as npr
import numpy.ma as ma
import scipy.linalg
from myivp.myivp import solve_ivp
import itertools
from scipy.spatial import Delaunay

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
    # returns normalized v
    d = scipy.linalg.norm(v, axis=-1)
    v_masked = ma.array(v) / ma.array(d[..., None])
    v = ma.getdata(v_masked.filled(0))
    return v


def getNormtoo(v):
    # returns norm of v and normalized v
    d = scipy.linalg.norm(v, axis=-1)
    v_masked = ma.array(v) / ma.array(d[..., None])
    v = ma.getdata(v_masked.filled(0))
    return v, d


def getRotMatArray(Phis):
    Thetas = scipy.linalg.norm(Phis, axis=1)
    Axes_masked = ma.array(Phis) / ma.array(Thetas[..., None])
    Axes = ma.getdata(Axes_masked.filled(0))
    a = np.cos(Thetas / 2)
    b, c, d = np.transpose(Axes) * np.sin(Thetas / 2)
    RotMat = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                      [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                      [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])
    return np.transpose(RotMat, axes=(2, 0, 1))


def getRotMat(Phis):
    Axis, Theta = getNormtoo(Phis)
    a = np.cos(Theta / 2)
    b, c, d = Axis * np.sin(Theta / 2)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                    [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                    [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def VoronoiNeighbors(positions, vodims=2):
    if vodims == 3:
        # p = [n for (n, f) in positions]  # component [0] is a 3d vector
        p = positions
    else:
        p = [n[:2] for n in positions]
    # tri: list of interconnectedparticles: [ (a, b, c), (b, c, d), ... ]
    tri = Delaunay(p, qhull_options='QJ')
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


class NodeConfiguration:
    def __init__(self, num, dims=3, isF0=False, isanchor=False):
        if dims == 2:
            self.updateLinkForces = lambda PHI, T, Norm, NormT, Bend, Twist, K, D0, Nodeinds: \
                self.updateLinkForces2D(PHI, T, Bend, K, D0, Nodeinds)
            self.dims = dims
        elif dims == 3:
            self.updateLinkForces = lambda PHI, T, Norm, NormT, Bend, Twist, K, D0, Nodeinds: \
                self.updateLinkForces3D(PHI, T, Norm, NormT, Bend, Twist, K, D0, Nodeinds)
            self.dims = dims
        else:
            print "Oops! Wrong number of dimensions here."
            sys.exit()

        self.dims = dims
        # variables to store cell number and cell positions and angles
        self.N = num
        self.N_inv = 1. / self.N

        """description of nodes"""
        self.nodesX = np.zeros((self.N, 3))             # r of nodes
        self.nodesPhi = np.zeros((self.N, 3))           # phi of nodes
        self.Fnode = np.zeros((self.N, 3))              # total force on node
        self.Mnode = np.zeros((self.N, 3))              # total torsion on node
        self.isF0 = isF0
        self.F0 = np.zeros((self.N, 3))                 # external force on node
        self.isanchor = isanchor
        self.X0 = np.zeros((self.N, 3))                 # node anchor, must be set if needed!
        self.knode = np.zeros((self.N,))                 # spring constant of node to anchor point, defaults to 0

        """description of links"""
        # islink[i, j] is True if nodes i and j are connected via link
        self.islink = np.full((self.N, self.N), False)  # Describes link at node [0] leading to node [1]

        self.e = np.zeros((self.N, self.N, 3))          # direction connecting nodes (a.k.a. "actual direction")
        self.d = np.zeros((self.N, self.N))             # distance between nodes (a.k.a. "actual distance")
        self.k = np.zeros((self.N, self.N))             # spring constant between nodes
        self.bend = 10. * np.ones((self.N, self.N))     # bending rigidity
        self.twist = 1. * np.ones((self.N, self.N))     # torsion spring constant
        self.d0 = np.zeros((self.N, self.N))            # equilibrium distance between nodes,
        self.t = np.zeros((self.N, self.N, 3))          # tangent vector of link at node (a.k.a. "preferred direction")
        self.norm = np.zeros((self.N, self.N, 3))       # normal vector of link at node
        self.Mlink = np.zeros((self.N, self.N, 3))      # Torsion from link on node
        self.Flink = np.zeros((self.N, self.N, 3))      # Force from link on node

        # event for terminating numerical integration
        self.Q = [1]

        """stuff for documentation"""
        self.nodesnap = []
        self.linksnap = []
        self.fnodesnap = []
        self.flinksnap = []

        self.nodesum = lambda: 0

        self.reset_nodesum()

    def reset_nodesum(self):
        # add forces (external or anchor based) to nodes
        if self.isF0 is False and self.isanchor is False:
            self.nodesum = lambda: np.sum(self.Flink, axis=1)
        elif self.isF0 is True and self.isanchor is False:
            self.nodesum = lambda: np.sum(self.Flink, axis=1) + self.F0
        elif self.isF0 is False and self.isanchor is True:
            self.nodesum = lambda: np.sum(self.Flink, axis=1) + self.knode * (self.X0 - self.nodesX)
        elif self.isF0 is True and self.isanchor is True:
            self.nodesum = lambda: np.sum(self.Flink, axis=1) + self.F0 + self.knode * (self.X0 - self.nodesX)

    def addlink(self, ni, mi, t1=None, t2=None, d0=None, k=None, bend=None, twist=None, n=None, norm1=None, norm2=None):

        self.islink[ni, mi], self.islink[mi, ni] = True, True

        if k is None:
            k = 15.
        self.k[ni, mi], self.k[mi, ni] = k, k  # spring parameter
        if bend is not None:
            self.bend[mi, ni], self.bend[ni, mi] = bend, bend
        if twist is not None:
            self.twist[mi, ni], self.bend[ni, mi] = twist, twist

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
        self.islink[ni, mi], self.islink[mi, ni] = False, False
        self.Flink[ni, mi], self.Flink[mi, ni], self.Mlink[ni, mi], self.Mlink[mi, ni] = null, null, null, null
        self.t[ni, mi], self.t[mi, ni], self.norm[ni, mi], self.norm[mi, ni] = null, null, null, null
        self.k[ni, mi], self.k[mi, ni], self.d0[ni, mi], self.d0[mi, ni] = 0, 0, 0, 0

    def updateDists(self, X):
        dX = X - X[:, None, :]
        self.d = scipy.linalg.norm(dX, axis=2)
        e_masked = ma.array(dX) / ma.array(self.d[..., None])
        self.e = ma.getdata(e_masked.filled(0))

    def compactStuffINeed(self):
        # get only those parts of the big arrays that are actually needed
        nodeinds = np.where(self.islink == True)
        t = self.t[nodeinds]
        norm = self.norm[nodeinds]
        normT = np.transpose(self.norm, axes=(1, 0, 2))[nodeinds]
        bend = self.bend[nodeinds]
        twist = self.twist[nodeinds]
        k = self.k[nodeinds]
        d0 = self.d0[nodeinds]

        return t, norm, normT, bend, twist, k, d0, nodeinds

    def updateLinkForces2D(self, PHI, T, Bend, K, D0, Nodeinds):
        E = self.e[Nodeinds]
        D = self.d[Nodeinds]

        # rotated version of t to fit current setup
        TNow = np.einsum("ijk, ik -> ij", getRotMatArray(PHI[Nodeinds[0]]), T)

        self.Mlink[Nodeinds] = Bend[..., None] * np.cross(TNow, E)  # Eq 3

        M = self.Mlink + np.transpose(self.Mlink, axes=(1, 0, 2))
        M = M[Nodeinds]

        self.Flink[Nodeinds] = (K * (D - D0))[..., None] * E + np.cross(M, E) / D[:, None]  # Eqs. 10, 13, 14, 15

    def updateLinkForces3D(self, PHI, T, Norm, NormT, Bend, Twist, K, D0, Nodeinds):
        E = self.e[Nodeinds]
        D = self.d[Nodeinds]
        NodesPhi = PHI[Nodeinds[0]]
        NodesPhiT = PHI[Nodeinds[1]]

        # rotated version of Norm and NormT to fit current setup
        NormNow = np.einsum("ijk, ik -> ij", getRotMatArray(NodesPhi), Norm)
        NormTNow = np.einsum("ijk, ik -> ij", getRotMatArray(NodesPhiT), NormT)

        # rotated version of t to fit current setup
        TNow = np.einsum("ijk, ik -> ij", getRotMatArray(NodesPhi), T)

        # calculated new vector \bm{\tilde{n}}_{A, l}
        NormTilde = getNormvec(NormNow - np.einsum("ij, ij -> i", NormNow, E)[:, None] * E)
        NormTTilde = getNormvec(NormTNow - np.einsum("ij, ij -> i", NormTNow, E)[:, None] * E)

        self.Mlink[Nodeinds] = Bend[..., None] * np.cross(TNow, E) + \
                               Twist[..., None] * np.cross(NormTilde, NormTTilde)  # Eq 5

        M = self.Mlink + np.transpose(self.Mlink, axes=(1, 0, 2))
        M = M[Nodeinds]

        self.Flink[Nodeinds] = (K * (D - D0))[..., None] * E + np.cross(M, E) / D[:, None]  # Eqs. 10, 13, 14, 15

    def getForces(self, x, t, norm, normT, bend, twist, k, d0, nodeinds):
        X = x.reshape(self.N, 6)
        Phi = X[:, 3:]
        X = X[:, :3]
        self.updateDists(X)
        self.updateLinkForces(Phi, t, norm, normT, bend, twist, k, d0, nodeinds)
        self.Fnode = self.nodesum()
        self.Mnode = np.sum(self.Mlink, axis=1)
        return np.concatenate((self.Fnode, self.Mnode), axis=1).flatten()

    def getLinkList(self):
        allLinks0, allLinks1 = np.where(self.islink == True)
        return np.array([[allLinks0[i], allLinks1[i]] for i in range(len(allLinks0)) if allLinks0[i] > allLinks1[i]])

    def getLinkTuple(self):
        allLinks0, allLinks1 = np.where(self.islink == True)
        inds = np.where(allLinks0 > allLinks1)
        return allLinks0[inds], allLinks1[inds]


class SubsConfiguration:
    def __init__(self, num_cells, num_subs, dims=3):
        self.dims = dims
        # variables to store cell number and cell positions and angles
        self.N = num_cells
        self.Nsubs = num_subs

        """description of nodes"""
        self.nodesX = np.zeros((self.Nsubs, 3))  # r of subs nodes
        self.Fnode = np.zeros((self.Nsubs, 3))  # force exerted on subs nodes
        # self.Mnode = np.zeros((self.Nsubs, 3))  # torque exerted on subs nodes

        """description of links"""
        # islink[i, j] is True if nodes i and j are connected via link
        self.islink = np.full((self.N, self.Nsubs), False)  # Describes link at cell node [0] leading to subs node [1]

        self.e = np.zeros((self.N, self.Nsubs, 3))  # direction from cell node to subs node
        self.d = np.zeros((self.N, self.Nsubs))  # distance between nodes (a.k.a. "actual distance")
        self.k = np.zeros((self.N, self.Nsubs))  # spring constant between nodes
        self.bend = 10. * np.ones((self.N, self.Nsubs))  # bending rigidity
        self.twist = 1. * np.ones((self.N, self.Nsubs))  # torsion spring constant
        self.d0 = np.zeros((self.N, self.Nsubs))  # equilibrium distance between nodes,
        self.tcell = np.zeros((self.N, self.Nsubs, 3))  # tangent vector of link at cell node
        self.tsubs = np.zeros((self.N, self.Nsubs, 3))  # tangent vector of link at subs node
        self.normcell = np.zeros((self.N, self.Nsubs, 3))  # normal vector of link at cell node
        self.normsubs = np.zeros((self.N, self.Nsubs, 3))  # normal vector of link at subs node
        self.Mcelllink = np.zeros((self.N, self.Nsubs, 3))  # Torsion from link on cell node
        self.Msubslink = np.zeros((self.N, self.Nsubs, 3))  # Torsion from link on subs node
        self.Flink = np.zeros((self.N, self.Nsubs, 3))  # Force from link on cell node

        """stuff for documentation"""
        self.linksnap = []
        self.fnodesnap = []
        self.flinksnap = []

    def addlink(self, ni, mi, cellphi, t1=None, d0=None, k=None, bend=None, twist=None, n=None, norm1=None, norm2=None):
        self.islink[ni, mi] = True

        if k is None:
            k = 15.
        self.k[ni, mi] = k  # spring parameter
        if bend is not None:
            self.bend[ni, mi] = bend
        if twist is not None:
            self.twist[ni, mi] = twist

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
        self.islink[ni, mi] = False
        self.Flink[ni, mi], self.Mcelllink[ni, mi], self.Msubslink[ni, mi] = null, null, null
        self.tcell[ni, mi], self.tsubs[ni, mi], self.normcell[ni, mi], self.normsubs[ni, mi] = null, null, null, null
        self.k[ni, mi], self.d0[ni, mi] = 0, 0

    def updateDists(self, X):
        dX = self.nodesX - X[:, None, :]
        self.d = scipy.linalg.norm(dX, axis=2)
        e_masked = ma.array(dX) / ma.array(self.d[..., None])
        self.e = ma.getdata(e_masked.filled(0))

    def compactStuffINeed(self):
        # get only those parts of the big arrays that are actually needed
        nodeinds = np.where(self.islink == True)
        tcell = self.tcell[nodeinds]
        tsubs = self.tsubs[nodeinds]
        normcell = self.normcell[nodeinds]
        normsubs = self.normsubs[nodeinds]
        bend = self.bend[nodeinds]
        twist = self.twist[nodeinds]
        k = self.k[nodeinds]
        d0 = self.d0[nodeinds]

        return tcell, tsubs, normcell, normsubs, bend, twist, k, d0, nodeinds

    def updateLinkForces(self, PHI, TCell, TSubs, NormCell, NormSubs, Bend, Twist, K, D0, Nodeinds):
        E = self.e[Nodeinds]
        D = self.d[Nodeinds]
        NodesPhiCell = PHI[Nodeinds[0]]

        # rotated version of Norm and NormT to fit current setup
        NormCellNow = np.einsum("ijk, ik -> ij", getRotMatArray(NodesPhiCell), NormCell)

        # rotated version of t to fit current setup
        TCellNow = np.einsum("ijk, ik -> ij", getRotMatArray(NodesPhiCell), TCell)

        # calculated new vector \bm{\tilde{n}}_{A, l}
        NormCellTilde = getNormvec(NormCellNow - np.einsum("ij, ij -> i", NormCellNow, E)[:, None] * E)
        NormSubsTilde = getNormvec(NormSubs - np.einsum("ij, ij -> i", NormSubs, -E)[:, None] * (-E))

        self.Mcelllink[Nodeinds] = Bend[..., None] * np.cross(TCellNow, E) + \
                                   Twist[..., None] * np.cross(NormCellTilde, NormSubsTilde)  # Eq 5 for cells

        self.Msubslink[Nodeinds] = Bend[..., None] * np.cross(TSubs, -E) + \
                                   Twist[..., None] * np.cross(NormSubsTilde, NormCellTilde)  # Eq 5 for substrate

        M = self.Mcelllink + self.Msubslink
        M = M[Nodeinds]

        self.Flink[Nodeinds] = (K * (D - D0))[..., None] * E + np.cross(M, E) / D[:, None]  # Eqs. 10, 13, 14, 15

    def getForces(self, x, tcell, tsubs, normcell, normsubs, bend, twist, k, d0, nodeinds):
        X = x.reshape(self.N, 6)
        Phi = X[:, 3:]
        X = X[:, :3]
        self.updateDists(X)
        self.updateLinkForces(Phi, tcell, tsubs, normcell, normsubs, bend, twist, k, d0, nodeinds)
        self.Fnode = np.sum(self.Flink, axis=0)
        # self.Mnode = np.sum(self.Mcelllink, axis=0)
        return np.concatenate((np.sum(self.Flink, axis=1), np.sum(self.Mcelllink, axis=1)), axis=1).flatten()

    def getLinkList(self):
        allLinks0, allLinks1 = np.where(self.islink == True)
        return np.array([[allLinks0[i], allLinks1[i]] for i in range(len(allLinks0))])

    def getLinkTuple(self):
        return np.where(self.islink == True)


class CellMech:
    def __init__(self, num_cells, num_subs=None, dt=0.01, nmax=3000, qmin=0.001, d0_0=1, force_limit=15., p_add=1.,
                 p_del=0.2, chkx=False, d0max=2., dims=3, isF0=False, isanchor=False, issubs=False):

        self.dims = dims
        self.issubs = issubs
        # variables to store cell number and cell positions and angles
        self.N = num_cells
        self.N_inv = 1. / self.N
        # parameters for mechanical equilibration
        self.dt = dt  # scaling factor for
        self.nmax = nmax
        self.tmax = nmax * dt
        self.qmin = np.sqrt(qmin)
        # parameters to add/remove links
        self.d0_0 = d0_0
        self.force_limit = force_limit
        self.p_add = p_add
        self.p_del = p_del
        self.chkx = chkx
        self.d0max = d0max
        # functions for randoms in default_update_d0
        self.lowers = np.tril_indices(self.N, -1)
        self.randomsummand = np.zeros((self.N, self.N))
        self.randomlength = int(self.N * (self.N - 1) / 2)

        """stuff for documentation"""
        self.snaptimes = []

        self.mynodes = NodeConfiguration(num=num_cells, dims=dims, isF0=isF0, isanchor=isanchor)

        if self.issubs is True:
            self.mysubs = SubsConfiguration(num_cells=num_cells, num_subs=num_subs)
            self.mechEquilibrium = lambda: self.mechEquilibrium_withsubs()
            self.makesnap = lambda t: self.makesnap_withsubs(t)
            self.addLinkList = lambda: self.addLinkList_withsubs()
        elif self.issubs is False:
            self.mechEquilibrium = lambda: self.mechEquilibrium_nosubs()
            self.makesnap = lambda t: self.makesnap_nosubs(t)
            self.addLinkList = lambda: self.addLinkList_nosubs()
        elif self.issubs is "lonesome":
            self.mysubs = SubsConfiguration(num_cells=num_cells, num_subs=num_subs)
            self.mechEquilibrium = lambda: self.mechEquilibrium_lonesome()
            self.makesnap = lambda t: self.makesnap_withsubs(t)
            self.addLinkList = lambda: self.addLinkList_lonesome()

    def mechEquilibrium_nosubs(self):
        # reshape X and Phi for solveivp
        x = np.concatenate((self.mynodes.nodesX.copy(), self.mynodes.nodesPhi.copy()), axis=1).flatten()
        t, norm, normT, bend, twist, k, d0, nodeinds = self.mynodes.compactStuffINeed()

        # produce fun for solve_ivp as lambda
        def notatallfun(temp, y): return self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds)

        # produce event function to check whether to end solve_ivp
        def event(temp, y):
            k1 = self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds)
            return np.max(np.abs(k1) - self.qmin)

        event.terminal = True
        event.direction = -1

        res = solve_ivp(fun=notatallfun, t_span=[0, self.tmax], y0=x, method='LSODA', events=[event], atol=1e-3)
        x = res.y.reshape((self.N, 6, len(res.t)))
        self.mynodes.nodesX = x[:, :3, -1]
        self.mynodes.nodesPhi = x[:, 3:, -1]
        if res.status != 1:
            print res.status
        return res.t[-1]

    def mechEquilibrium_withsubs(self):
        x = np.concatenate((self.mynodes.nodesX.copy(), self.mynodes.nodesPhi.copy()), axis=1).flatten()
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
            return np.max(np.abs(k1) - self.qmin)
        event.terminal = True
        event.direction = -1

        res = solve_ivp(fun=notatallfun, t_span=[0, self.tmax], y0=x, method='LSODA', events=[event], atol=1e-3)
        x = res.y.reshape((self.N, 6, len(res.t)))
        self.mynodes.nodesX = x[:, :3, -1]
        self.mynodes.nodesPhi = x[:, 3:, -1]
        return res.t[-1]

    def mechEquilibrium_lonesome(self):
        x = np.concatenate((self.mynodes.nodesX.copy(), self.mynodes.nodesPhi.copy()), axis=1).flatten()
        tcell, tsubs, normcell, normsubs, bends, twists, ks, d0s, nodeindss = self.mysubs.compactStuffINeed()

        # produce fun for solve_ivp as lambda
        def notatallfun(temp, y): return self.mysubs.getForces(y, tcell, tsubs, normcell, normsubs,
                                                               bends, twists, ks, d0s, nodeindss)

        # produce event function to check whether to end solve_ivp
        def event(temp, y):
            k1 = self.mysubs.getForces(y, tcell, tsubs, normcell, normsubs, bends, twists, ks, d0s, nodeindss)
            return np.max(np.abs(k1) - self.qmin)
        event.terminal = True
        event.direction = -1

        res = solve_ivp(fun=notatallfun, t_span=[0, self.tmax], y0=x, method='LSODA', events=[event], atol=1e-3)
        x = res.y.reshape((self.N, 6, len(res.t)))
        self.mynodes.nodesX = x[:, :3, -1]
        self.mynodes.nodesPhi = x[:, 3:, -1]
        return res.t[-1]

    def intersect_all(self):
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
        linklist = self.mynodes.getLinkList()
        to_del = []
        for link in linklist:
            if self.mynodes.d[link[0], link[1]] < self.mynodes.d0[link[0], link[1]]:
                continue            # compressed links are stable
            f = scipy.linalg.norm(self.mynodes.Flink[link[0], link[1]])
            p = exp(f / self.force_limit)
            to_del.append((link, p))
        if self.issubs:
            linklist = self.mysubs.getLinkList()
            for link in linklist:
                if self.mysubs.d[link[0], link[1]] < self.mysubs.d0[link[0], link[1]]:
                    continue
                f = scipy.linalg.norm(self.mysubs.Flink[link[0], link[1]])
                p = exp(f / self.force_limit)
                to_del.append((link, p))
        return to_del

    def tryLink_notsubs(self, n1, n2):
        if self.mynodes.islink[n1, n2]:
            return -1
        if self.dims == 2:
            if self.intersect_withone(n1, n2):
                return -1  # false
        d = scipy.linalg.norm(self.mynodes.nodesX[n1] - self.mynodes.nodesX[n2])
        if d > self.d0max:
            return -1  # false
        return d  # true: d>0

    def tryLink_issubs(self, n1, n2):
        if self.mysubs.islink[n1, n2]:
            return -1
        if self.dims == 2:
            if self.intersect_withone(n1, n2):
                return -1  # false
        d = scipy.linalg.norm(self.mynodes.nodesX[n1] - self.mysubs.nodesX[n2])
        if d > self.d0max:
            return -1  # false
        return d  # true: d>0

    def addLinkList_nosubs(self):
        to_add = []
        for i, j in VoronoiNeighbors(self.mynodes.nodesX, vodims=self.dims):
            d = self.tryLink_notsubs(i, j)
            if d > 1e-5:
                p = (1 - (d / self.d0max))
                to_add.append(((i, j), p))
        return to_add

    def addLinkList_withsubs(self):
        to_add = []
        allnodes = np.concatenate((self.mynodes.nodesX, self.mysubs.nodesX))
        for i, j in VoronoiNeighbors(allnodes, vodims=self.dims):
            if j >= self.N:
                if i < self.N:
                    d = self.tryLink_issubs(i, j - self.N)
                else:
                    d = -1
            else:
                d = self.tryLink_notsubs(i, j)
            if d > 1e-5:
                p = (1 - (d / self.d0max))
                to_add.append(((i, j), p))
        return to_add

    def addLinkList_lonesome(self):
        to_add = []
        for i in range(self.N):
            for j in range(self.mysubs.Nsubs):
                d = self.tryLink_issubs(i, j)
                if d > 1e-5:
                    p = (1 - (d / self.d0max))
                    to_add.append(((i, j), p))
        return to_add

    def pickEvent(self, to_del, to_add):
        s1 = 0.
        for (l, p) in to_del:
            s1 += p * self.p_del
        s2 = 0.
        for (q, p) in to_add:
            s2 += p * self.p_add

        S = s1 + s2
        if S < 1e-7:
            print "nothing to do!"
            return 1.
        dt = -log(npr.random()) / S
        if dt > 1:
            # print 'Must adjust d0 variables before the next event!'
            return 1.

        r = S * npr.random()
        if r < s1:  # we will remove a link
            for (l, p) in to_del:
                r = r - p * self.p_del
                if r < 0:
                    if l[1] < self.N:
                        self.mynodes.removelink(l[0], l[1])
                        return dt
                    else:
                        self.mysubs.removelink(l[0], l[1] - self.N)
                        return dt
        r = r - s1
        if r < s2:  # we will add a link
            for ((n1, n2), p) in to_add:
                r = r - p * self.p_add
                if r < 0:
                    if n2 < self.N:
                        self.mynodes.addlink(n1, n2)
                        return dt
                    else:
                        self.mysubs.addlink(n1, n2 - self.N, self.mynodes.nodesPhi[n1])
                        return dt

    def default_update_d0(self, dt):
        myrandom = npr.random((self.randomlength, ))
        self.randomsummand[self.lowers], self.randomsummand.T[self.lowers] = myrandom, myrandom
        self.mynodes.d0 += 0.2 * (self.d0_0 - self.mynodes.d0) * dt + 0.2 * (
                   2 * sqrt(dt) * self.randomsummand - sqrt(dt))              # magic number 0.2 and 0.05??
        if self.issubs is not False:
            self.mysubs.d0 += 0.2 * (self.d0_0 - self.mysubs.d0) * dt + 0.2 * (
                    2 * sqrt(dt) * self.randomsummand - sqrt(dt))  # magic number 0.2 and 0.05??

    def modlink(self):
        if self.chkx:
            self.checkLinkX()
        to_del = self.delLinkList()
        to_add = self.addLinkList()
        dt = self.pickEvent(to_del, to_add)
        self.default_update_d0(dt)
        return dt

    def makesnap_nosubs(self, t):
        self.mynodes.nodesnap.append(self.mynodes.nodesX.copy())
        self.mynodes.fnodesnap.append(self.mynodes.Fnode.copy())
        linkList = self.mynodes.getLinkList()
        self.mynodes.linksnap.append(linkList)
        self.mynodes.flinksnap.append(scipy.linalg.norm(self.mynodes.Flink[linkList[..., 0], linkList[..., 1]], axis=1))
        self.snaptimes.append(t)

    def makesnap_withsubs(self, t):
        self.mynodes.nodesnap.append(self.mynodes.nodesX.copy())
        self.mynodes.fnodesnap.append(self.mynodes.Fnode.copy())
        self.mysubs.fnodesnap.append(self.mysubs.Fnode.copy())
        linkList = self.mynodes.getLinkList()
        self.mynodes.linksnap.append(linkList)
        self.mynodes.flinksnap.append(scipy.linalg.norm(self.mynodes.Flink[linkList[..., 0], linkList[..., 1]], axis=1))
        linkList = self.mysubs.getLinkList()
        self.mysubs.linksnap.append(linkList)
        self.mysubs.flinksnap.append(scipy.linalg.norm(-self.mysubs.Flink[linkList[..., 0], linkList[..., 1]], axis=1))
        self.snaptimes.append(t)

    def savedata(self, savenodes_r=True, savelinks=True, savenodes_f=True, savelinks_f=True, savet=True):
        if savenodes_r:
            np.save("nodesr", self.mynodes.nodesnap)
        if savenodes_f:
            np.save("nodesf", self.mynodes.fnodesnap)
        if savelinks:
            np.save("links", self.mynodes.linksnap)
        if savelinks_f:
            np.save("linksf", self.mynodes.flinksnap)
        if savet:
            np.save("ts", self.snaptimes)

        if self.issubs:
            if savenodes_r:
                np.save("subsnodesr", self.mysubs.nodesX)
            if savenodes_f:
                np.save("subsnodesf", self.mysubs.fnodesnap)
            if savelinks:
                np.save("subslinks", self.mysubs.linksnap)
            if savelinks_f:
                np.save("subslinksf", self.mysubs.flinksnap)

    def timeevo(self, tmax, isinit=True, isfinis=True, record=False):
        t = 0.
        self.default_update_d0(self.dt)
        if record and isinit:
            self.makesnap(0)
        while t < tmax:
            dt = self.mechEquilibrium()
            t += dt
            if record:
                self.makesnap(t)
            update_progress(t / tmax)
            dt = self.modlink()
            t += dt
            if record:
                self.makesnap(t)
            update_progress(t / tmax)
        if record and isfinis:
            self.mynodes.nodesnap = np.array(self.mynodes.nodesnap)
            self.mynodes.fnodesnap = np.array(self.mynodes.fnodesnap)
            self.mynodes.snaptimes = np.array(self.snaptimes)
        if self.issubs:
            return self.mynodes.nodesnap, self.mynodes.linksnap, self.mynodes.fnodesnap, self.mynodes.flinksnap, \
                   self.snaptimes, \
                   self.mysubs.nodesX, self.mysubs.linksnap, self.mysubs.fnodesnap, self.mysubs.flinksnap

        else:
            return self.mynodes.nodesnap, self.mynodes.linksnap, self.mynodes.fnodesnap, self.mynodes.flinksnap, \
                   self.snaptimes

    def oneequil(self):
        linkList = self.mynodes.getLinkList()
        # reshape X and Phi for solveivp
        x = np.concatenate((self.mynodes.nodesX.copy(), self.mynodes.nodesPhi.copy()), axis=1).flatten()
        t, norm, normT, bend, twist, k, d0, nodeinds = self.mynodes.compactStuffINeed()

        # produce fun for solve_ivp as lambda
        def notatallfun(temp, y): return self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds)

        # produce event function to check wether to end solve_ivp
        def event(temp, y):
            k1 = self.mynodes.getForces(y, t, norm, normT, bend, twist, k, d0, nodeinds)
            print np.max(np.abs(k1))
            return np.max(np.abs(k1) - self.qmin)
        event.terminal = True
        event.direction = -1

        res = solve_ivp(fun=notatallfun, t_span=[0, self.tmax], y0=x, events=[event], method="LSODA", atol=1e-3)
        self.snaptimes = res.t
        self.mynodes.nodesnap = np.transpose(res.y.reshape((self.N, 6, len(self.snaptimes))), axes=(2, 0, 1))[..., :3]
        return self.mynodes.nodesnap, np.tile(linkList, (len(self.snaptimes), 1, 1)), None, None, self.snaptimes

    def oneequil_withsubs(self):
        linkList = self.mynodes.getLinkList()
        # reshape X and Phi for solveivp
        x = np.concatenate((self.mynodes.nodesX.copy(), self.mynodes.nodesPhi.copy()), axis=1).flatten()
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
        self.snaptimes = res.t
        self.mynodes.nodesnap = np.transpose(res.y.reshape((self.N, 6, len(self.snaptimes))), axes=(2, 0, 1))[..., :3]
        return self.mynodes.nodesnap, np.tile(linkList, (len(self.snaptimes), 1, 1)), None, None, self.snaptimes
