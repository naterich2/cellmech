from cell import *


def generatePoint(L):
    X0 = (npr.rand() - .5) * L
    Y0 = (npr.rand() - .5) * L
    Z0 = 0.
    return np.array([X0, Y0, Z0])


def rand3d(L):
    return (np.random.random(3) - 0.5) * L


def square(L, N, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0max, d0min, dims):
    if N is None:
        N = int(L ** 2)

    c = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del,
                 chkx=chkx, d0max=d0max, dims=dims, issubs=False)

    for ni in range(N):
        while True:
            R1 = generatePoint(L)
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(c.mynodes.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        c.mynodes.nodesX[ni] = R1
    return c


def cube(L, N, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0max, stretch=1.):
    if N is not None:
        print "N decission overriden because of possible conflict with Lmax"
    N = int(L ** 3)

    c = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, chkx=chkx, d0max=d0max, dims=3,
                 issubs=False)

    x0 = - stretch * L / 2.
    for ni in range(L):
        for nj in range(L):
            for nk in range(L):
                c.mynodes.nodesX[L * L * ni + L * nj + nk] = \
                    np.array([x0 + stretch * ni, x0 + stretch * nj, x0 + stretch * nk])
    return c


def square_wsubs(L, Lsubs, N, Nsubs, dt, nmax, qmin, d0_0, p_add, p_del, p_add_subs, p_del_subs,
                 chkx, d0max, d0min, dims):
    if N is None:
        N = int(L ** 2)
    if Nsubs is None:
        Nsubs = int(Lsubs ** 2)
    
    c = CellMech(N, num_subs=Nsubs, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del,
                 p_add_subs=p_add_subs, p_del_subs=p_del_subs, chkx=chkx, d0max=d0max, dims=dims, issubs=True)

    for ni in range(N):
        while True:
            R1 = generatePoint(L)
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(c.mynodes.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        c.mynodes.nodesX[ni] = R1

    for ni in range(Nsubs):
        while True:
            R1 = generatePoint(Lsubs)
            R1[2] = -d0_0
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(c.mysubs.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        c.mysubs.nodesX[ni] = R1

    return c


def onelonesome(L, dense, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0max, dims):
    Nsubs = int(L * L * dense)
    N = 1
    d0min = round(0.8 / sqrt(dense), 3)

    c = CellMech(N, num_subs=Nsubs, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, p_add_subs=p_add, 
                 p_del_subs=p_del, chkx=chkx, d0max=d0max, dims=dims, issubs="lonesome")

    c.mynodes.nodesX[0] = [0, 0, 0]

    for ni in range(Nsubs):
        while True:
            R1 = generatePoint(L)
            R1[2] = -d0_0
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(c.mysubs.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        c.mysubs.nodesX[ni] = R1

    return c


def bilayer(L, N, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0min, d0max, dims):
    if N is None:
        N = int(2 * (L ** 2))

    c = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, p_add_subs=p_add, 
                 p_del_subs=p_del, chkx=chkx, d0max=d0max, dims=dims, issubs=False)

    for ni in range(N):
        while True:
            R1 = generatePoint(L)
            if ni >= N / 2:
                R1[2] = d0_0
            OK = True
            for nj in range(ni):
                d = np.linalg.norm(c.mynodes.nodesX[nj] - R1)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        c.mynodes.nodesX[ni] = R1
    return c


def savesquare(d0min, L=10, N=None):
    if N is None:
        N = int(L ** 2)
    R = []
    for ni in range(N):
        while True:
            R1 = generatePoint(L)
            OK = True
            for r in R:
                d = np.linalg.norm(R1 - r)
                if d < d0min:
                    OK = False
                    break
            if OK:
                break
        R.append(R1)
    R = np.array(R)
    np.save("Rinit", R)

    return R


def fromdefault(R, dt, nmax, qmin, d0_0, p_add, p_del, chkx, d0max, dims):
    N = len(R)
    c = CellMech(N, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del, p_add_subs=p_add, 
                 p_del_subs=p_del, chkx=chkx, d0max=d0max, dims=dims, issubs=False)
    for ni in range(N):
        c.mynodes.nodesX[ni] = R[ni]

    return c
