from mayavi import mlab
import scipy.linalg
import numpy as np
import subprocess, os, sys


def initconfig(c, l, nF, fl, figure, figureindex=0, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
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
def animateconfigs(Simdata, SubsSimdata=None, record=False, recorddir="./movie/", recordname="ani",
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
    :param record: boolean, whether or not to create and save a movie in oppose to only showing the animation.
        If set to true: call record_cleanup() after completing mlab.show()
    :param recorddir: string, directory where the images are saved and the movie should be save
    :param recordname: string, prefix of the images and the movie
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
    cells, links = initconfig(Configs[0], Links[0], nodeForces[0], linkForces[0], fig, cmap=cmap, cbar=cbar,
                              vmaxcells=vmaxcells, vmaxlinks=vmaxlinks, upto=upto)

    text = mlab.title('0.0', height=.9)  # show current time

    # Output path for saving animation images as intermediate step to producing recording
    out_path = recorddir
    if record and not os.path.isdir(out_path):
        try:
            os.mkdir(out_path)
        except OSError:
            print("Too many levels in recorddir missing. Sorry!")
            sys.exit()
    out_path = os.path.abspath(out_path)
    prefix = recordname
    ext = '.png'
    padding = 5
    i = 0

    # create animation
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

            if record:
                tstring = str(i).zfill(padding)
                filename = os.path.join(out_path, '{}_{}{}'.format(prefix, tstring, ext))
                mlab.savefig(filename=filename)
                i += 1
                if i == len(Configs):
                    i = 0

            yield


def record_cleanup(out_path="./movie", prefix="ani", fps=10):
    """
    Transform intermediate image files into a movie
    :param out_path: string, directory where the images are saved and the video should be save
    :param prefix: string, prefix of the images and the video
    :param fps: int, frames per second of the resulting video
    :return:
    """
    out_path = os.path.abspath(out_path)
    ext = '.png'
    padding = 5

    ffmpeg_fname = os.path.join(out_path, '{}_%0{}d{}'.format(prefix, padding, ext))
    cmd = 'ffmpeg -f image2 -r {} -i {} -q:v 1 -vcodec mpeg4 -y {}/{}.mp4'.format(fps, ffmpeg_fname, out_path, prefix)
    print(cmd)
    subprocess.check_output(['bash', '-c', cmd])

    # Remove temp image files with extension
    [os.remove(out_path + "/" + f) for f in os.listdir(out_path) if f.endswith(ext)]


def fetchdata(fetchdir, toskip=1):
    """
    Loads data for simulation from files in directory "dir"
    :param fetchdir: string, name of directory holding data
    :param toskip: int, only use every skip-th simulation step for animation
    :return: a) boolean value indicating if simulation data contains substrate information, b) tuple with data
        on node positions, links, forces on nodes, forces on links and time steps and c) if substrate exists:
        positions of substrate nodes, substrate links, forces on substrate nodes, forces on substrate links
    """
    configs = np.load(fetchdir + "/nodesr.npy",allow_pickle=True,encoding='bytes')[::toskip]
    links = np.load(fetchdir + "/links.npy",allow_pickle=True,encoding='bytes')[::toskip]
    nodeforces = np.load(fetchdir + "/nodesf.npy",allow_pickle=True,encoding='bytes')[::toskip]
    linkforces = np.load(fetchdir + "/linksf.npy",allow_pickle=True,encoding='bytes')[::toskip]
    ts = np.load(fetchdir + "/ts.npy",allow_pickle=True,encoding='bytes')[::toskip]

    try:     # try to include substrate details if they exists
        subs = np.load(fetchdir + "/subsnodesr.npy",allow_pickle=True,encoding='bytes')[::toskip]
        subslinks = np.load(fetchdir + "/subslinks.npy",allow_pickle=True,encoding='bytes')[::toskip]
        subsnodeforces = np.load(fetchdir + "/subsnodesf.npy",allow_pickle=True,encoding='bytes')[::toskip]
        subslinkforces = np.load(fetchdir + "/subslinksf.npy",allow_pickle=True,encoding='bytes')[::toskip]

        return True, (configs, links, nodeforces, linkforces, ts), (subs, subslinks, subsnodeforces, subslinkforces)

    except IOError:  # if no substrate results exist
        return False, (configs, links, nodeforces, linkforces, ts)


if __name__ == '__main__':

    # produce animation from previously saved simulation results

    ####################

    skip = 1                # only use every skip-th simulation step for animation
    datadir = "res"         # location of simulation results
    showsubs = False        # whether or not to visualize substrate nodes

    record = False          # whether or not to create and save a movie in oppose to only showing the animation
    recorddir = "./movie"   # directory for saving the movie
    recordname = "ani"      # name of the movie
    fps = 10                # frames per second of the movie

    ####################

    simdata = fetchdata(datadir, skip)

    if simdata[0]:
        animateconfigs(simdata[1], simdata[2],
                       showsubs=showsubs, record=record, recorddir=recorddir, recordname=recordname)
    else:
        animateconfigs(simdata[1], record=record, recorddir=recorddir, recordname=recordname)

    mlab.show(stop=True)

    if record:  # create movie from intermediate files
        record_cleanup(recorddir=recorddir, recordname=recordname, fps=fps)
