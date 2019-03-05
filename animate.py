from mayavi import mlab
import scipy.linalg
import numpy as np


def showconfig(c, l, nF, fl, figure, figureindex=0, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
               figsize=(1000, 1000), cmap='viridis', vmaxlinks=5, vmaxcells=5, cbar=False, upto=-1):
    """

    :param c: numpy array of shape (ncells, 3) containing positions of cells
    :param l: numpy array of shapes (nlinks, 2) containing indices of cells connected by links
    :param nF: numpy array of shape (ncells, 3) containing forces on cells
    :param fl: numpy array of shapes (nlinks) containing absolute value of forces on links connecting  cells
    :param figure: figure passed on
    :param figureindex: n of figure
    :param bgcolor: tuple of shape (3,) indicating foreground color
    :param fgcolor: tuple of shape (3,) indicating background color
    :param figsize: tuple of shape (2,) indicating figure size
    :param cmap: color map visualization
    :param cbar: color bar
    :param vmaxlinks: maximum value of link forces for tuning of color-scale
    :param vmaxcells: maximum value of node forces for tuning of color-scale
    :param upto: index of last tissue cell when concatenating tissue and substrate cells
    :return: animation data
    """

    if figure is None:
        fig = mlab.figure(figureindex, bgcolor=bgcolor, fgcolor=fgcolor, size=figsize)
    else:
        fig = figure
    x, y, z = c.T  # extract all x, y and z positions in individual arrays
    xl, yl, zl = c[l[..., 0]].T  # extract x, y, and z positions of one end for each link
    rxl, ryl, rzl = (c[l[..., 1]] - c[l[..., 0]]).T  # extract x, y and z components of vectors describing links
    fc = scipy.linalg.norm(nF, axis=1)  # get absolute value of force on nodes

    # initialize cell visualization
    cells = mlab.points3d(x[:upto], y[:upto], z[:upto], fc[:upto], scale_factor=1, opacity=0.5, resolution=16,
                          scale_mode='none', vmin=0., colormap=cmap, vmax=vmaxcells)

    # initialize link visualization
    links = mlab.quiver3d(xl, yl, zl, rxl, ryl, rzl, scalars=fl, mode='2ddash', line_width=4., scale_mode='vector',
                          scale_factor=1, colormap=cmap, vmin=0., vmax=vmaxlinks)
    links.glyph.color_mode = "color_by_scalar"
    if cbar:
        mlab.scalarbar(links, nb_labels=2, title='Force on link')
    return cells, links


def pack(A, B):
    """
    Concatenates two lists of numpy arrays. shape(A[i]) must be equal to shape(B[i]) for all i
    :param A: list containing numpy arrays
    :param B: list containing numpy arrays
    :return: list of concatenated numpy arrays
    """
    if len(A) == 0:
        return B
    try:
        C = []
        for ind in range(len(A)):
            C.append(np.concatenate((A[ind], B[ind])))
        return C
    except TypeError:
        return A


@mlab.animate(delay=70)
def animateconfigs(Simdata, SubsSimdata=None,
                   figureindex=0, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), figsize=(1000, 1000),
                   cmap='viridis', cbar=False, showsubs=False):
    """
    Create animation of simulation results previously created with cell.py
    :param Simdata: Tuple containing items:
        Configs: numpy array of shape (timesteps, ncells, 3) containing positions of tissue cells
        Links: list of length (timesteps,) containing numpy arrays of shapes (nlinks, 2) containing indices of tissue
            cells connected by links
        nodeForces: numpy array of shape (timesteps, ncells, 3) containing forces on tissue cells
        linkForces: list of length (timesteps,) containing numpy arrays of shapes (nlinks, 3) containing forces on
            links connecting tissue cells
        ts: numpy array of shape (timesteps,) containing times of the system snapshots
    :param SubsSimdata: None if simulations didn't include substrate, or Tuple containing items:
        Subs: numpy array of shape (timesteps, ncells, 3) containing positions of substrate cells
        SubsLinks: list of length (timesteps,) containing numpy arrays of shape (nlinks, 2) containing indices of
            tissue and substrate cells connected by links
        subsnodeForces: numpy array of shape (timesteps, ncells, 3) containing forces on substrate cells
        subslinkForces: list of length (timesteps,) containing numpy arrays of shapes (nlinks, 3) containing forces
            on links connecting tissue with substrate cells
    :param figureindex: n of figure
    :param bgcolor: tuple of shape (3,) indicating foreground color
    :param fgcolor: tuple of shape (3,) indicating background color
    :param figsize: tuple of shape (2,) indicating figure size
    :param cmap: color map visualization
    :param cbar: color bar
    :param showsubs: boolean, whether or not to explicitly visualize substrate cells
    :return:
    """
    # unpack Simdata and SubsSimdata
    Configs, Links, nodeForces, linkForces, ts = Simdata

    if SubsSimdata is None:
        Subs = None
        SubsLinks = None
        subsnodeForces = None
        subslinkForces = None
    else:
        Subs, SubsLinks, subsnodeForces, subslinkForces = SubsSimdata

    fig = mlab.figure(figureindex, bgcolor=bgcolor, fgcolor=fgcolor, size=figsize)

    if showsubs:
        upto = None
    else:
        upto = len(Configs[0])  # index of last tissue cell when concatenating tissue and substrate cells

    # prepare data for animation in different cases of CellMech.issubs

    if nodeForces is None:
        nodeForces = np.zeros(Configs.shape)
    if linkForces is None and Links is not None:
        linkForces = np.zeros((len(Links), len(Links[0]), 3))

    if Subs is not None:
        Subs = np.tile(Subs, (len(Configs), 1, 1))
        for t in range(len(SubsLinks)):
            try:
                SubsLinks[t][:, 1] += len(Configs[t])
            except IndexError:
                pass

    if Links is None:
        Links = SubsLinks
        linkForces = subslinkForces
        SubsLinks = None
        subslinkForces = None

    Configs = pack(Configs, Subs)
    Links = pack(Links, SubsLinks)
    nodeForces = pack(nodeForces, subsnodeForces)
    linkForces = pack(linkForces, subslinkForces)

    # get absolute value of force on links
    linkForces = [scipy.linalg.norm(lFstep, axis=1) for lFstep in linkForces]

    # get maximum value of node and link forces to tune the color scale
    vmaxcells = np.max(scipy.linalg.norm(nodeForces, axis=2))
    vmaxlinks = max([np.max(timestep) for timestep in linkForces])

    # show first timestep of animation
    cells, links = showconfig(Configs[0], Links[0], nodeForces[0], linkForces[0], fig, cmap=cmap, cbar=cbar,
                              vmaxcells=vmaxcells, vmaxlinks=vmaxlinks, upto=upto)

    text = mlab.title('0.0', height=.9)  # show current time

    while True:
        for (c, l, nF, fl, t) in zip(Configs, Links, nodeForces, linkForces, ts):
            x, y, z = c.T  # extract all x, y and z positions in individual arrays
            xl, yl, zl = c[l[..., 0]].T  # extract x, y, and z positions of one end for each link
            rxl, ryl, rzl = (c[l[..., 1]] - c[l[..., 0]]).T  # extract x, y and z components of vectors describing links
            fc = scipy.linalg.norm(nF, axis=1)  # get absolute value of force on nodes

            # update data
            cells.mlab_source.set(x=x[:upto], y=y[:upto], z=z[:upto], scalars=fc[:upto])
            links.mlab_source.reset(x=xl, y=yl, z=zl, u=rxl, v=ryl, w=rzl, scalars=fl)
            text.set(text='{}'.format(round(t, 2)))
            yield


if __name__ == '__main__':

    # produce animation from previously saved simulation results

    ####################

    skip = 1            # only use every skip-th simulation step for animation
    dir = "resy"         # location of simulation results
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

        animateconfigs((configs, links, nodeforces, linkforces, ts), (subs, subslinks, subsnodeforces, subslinkforces),
                       showsubs=False)

    except IOError: # if no substrate results exist
        animateconfigs((configs, links, nodeforces, linkforces, ts))

    mlab.show()