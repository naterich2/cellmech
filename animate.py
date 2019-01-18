#!/usr/bin/python  -u

from mayavi import mlab
import scipy.linalg
import numpy as np


def showconfig(c, l, nF, fl, figure=None, figureindex=0, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
               figsize=(1000, 1000), cmap='viridis', vmaxlinks=5, vmaxcells=5, cbar=False, upto=-1):
    if figure is None:
        fig = mlab.figure(figureindex, bgcolor=bgcolor, fgcolor=fgcolor, size=figsize)
    else:
        fig = figure
    x, y, z = c.T
    xl, yl, zl = c[l[..., 0]].T
    rxl, ryl, rzl = (c[l[..., 1]] - c[l[..., 0]]).T
    fc = scipy.linalg.norm(nF, axis=1)

    cells = mlab.points3d(x[:upto], y[:upto], z[:upto], fc[:upto], scale_factor=1, opacity=0.5, resolution=16,
                          scale_mode='none', vmin=0., colormap=cmap, vmax=vmaxcells)

    links = mlab.quiver3d(xl, yl, zl, rxl, ryl, rzl, scalars=fl, mode='2ddash', line_width=4., scale_mode='vector',
                          scale_factor=1, colormap=cmap, vmin=0., vmax=vmaxlinks)
    links.glyph.color_mode = "color_by_scalar"
    if cbar:
        mlab.scalarbar(links, nb_labels=2, title='Force on link')
    return cells, links


def pack(A, B):
    # concatenates lists of numpy arrays. watch out for shapes!
    try:
        C = []
        for ind in range(len(A)):
            C.append(np.concatenate((A[ind], B[ind])))
        return C
    except TypeError:
        return A


@mlab.animate(delay=70)
def animateconfigs(Configs, Links, nodeForces, linkForces, ts,
                   Subs=None, SubsLinks=None, subsnodeForces=None, subslinkForces=None,
                   figureindex=0, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), figsize=(1000, 1000),
                   cmap='viridis', cbar=False, showsubs=True):
    fig = mlab.figure(figureindex, bgcolor=bgcolor, fgcolor=fgcolor, size=figsize)


    if showsubs:
        upto = -1
    else:
        upto = len(Configs[0])

    if nodeForces is None:
        nodeForces = np.zeros(Configs.shape)
    if linkForces is None:
        linkForces = np.zeros((len(Links), len(Links[0])))

    if Subs is not None:
        Subs = np.tile(Subs, (len(Configs), 1, 1))
        for t in range(len(SubsLinks)):
            SubsLinks[t][:, 1] += len(Configs[t])

    Configs = pack(Configs, Subs)
    Links = pack(Links, SubsLinks)
    nodeForces = pack(nodeForces, subsnodeForces)
    linkForces = pack(linkForces, subslinkForces)

    vmaxcells = np.max(scipy.linalg.norm(nodeForces, axis=2))
    vmaxlinks = max([np.max(timestep) for timestep in linkForces])

    cells, links = showconfig(Configs[0], Links[0], nodeForces[0], linkForces[0], fig,
                              vmaxcells=vmaxcells, vmaxlinks=vmaxlinks, upto=upto)
    text = mlab.title('0.0', height=.9)

    while True:
        for (c, l, nF, fl, t) in zip(Configs, Links, nodeForces, linkForces, ts):
            x, y, z = c.T
            xl, yl, zl = c[l[..., 0]].T
            rxl, ryl, rzl = (c[l[..., 1]] - c[l[..., 0]]).T
            fc = scipy.linalg.norm(nF, axis=1)

            cells.mlab_source.set(x=x[:upto], y=y[:upto], z=z[:upto], scalars=fc[:upto])
            links.mlab_source.reset(x=xl, y=yl, z=zl, u=rxl, v=ryl, w=rzl, scalars=fl)
            text.set(text='{}'.format(round(t, 2)))
            yield
