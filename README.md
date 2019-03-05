# cellmech
<pre>

requires numpy, scipy for running simulations
requires mayavi for visualizing simulation results

Implementation of model for cell-resolved, multiparticle model of plastic tissue deformations and morphogenesis
first suggested by Czirok et al in 2014 (https://iopscience.iop.org/article/10.1088/1478-3975/12/1/016005/meta)
and first implemented in https://github.com/aczirok/cellmech.

Code contains extensions by Moritz Zeidler, Dept. for Innovative Methods of Computing (IMC), Centre for Information 
Services and High Performance Computing (ZIH) at Technische Universitaet Dresden. Code introduces substrate nodes. 
Substrate nodes behave like tissue nodes, but can only form links with other tissue nodes. They have three rotational,
but no translational degrees of freedom.

Contact via moritz.zeidler at tu-dresden.de or andreas.deutsch at tu-dresden.de.
        
************************************************************************************************************************

View files begining with "test_" for simple examples on how to use the code.

cell.py:
    the algorithm for running the model
    
myivp:
    contains a modified version of scipy.integrate.solve_ivp
    
animate.py:
    functions for 3D-animation of simulation results
    
makeanimation.py:
    code for animation of previously generated simulation results
    
test_minimal.py:
    run one mechanical equilibration for cells with stretched links

test_plane.py:
    run simulation of cells in 2d initialized in square
    
test_rod.py:
    run simulation in 2d of cells initalized in the shape of double-rod
    
test_bilayer.py:
    run simulation of bilayer of cells
    
test_onelonesome.py:
    run simulation of one cell diffusing across a substrate
    
test_substrate.py:
    run simulation of cells initialized in square above substrate
    
rest_relaunch.py:
    run simulation of cells in 2d initialized in square with interruption and relaunch after half-time

************************************************************************************************************************

Main functions from package "cell.py" relevant for using code:


Initiate system by calling: 

CellMech(self, num_cells, num_subs=0, dt=0.01, nmax=300, qmin=0.001, d0_0=1., force_limit=15., p_add=1.,
                 p_del=0.2, p_add_subs=None, p_del_subs=None, chkx=False, d0max=2., dims=3, F_contr=1.,
                 isF0=False, isanchor=False, issubs=False, force_contr=True)
        
        :param num_cells: integer, the number of tissue cells
        :param num_subs: integer, the number of substrate cells
        :param dt: float, the time unit for scaling simulation time
        :param nmax: integer, the maximum time (in simulation time) for until cutoff when calculating
            mechanical equilibrium
        :param qmin: float, the square of the maximum force per cell until mechanical equilibration is cut off
        :param d0_0: float, the global equilibrium link length (d_0 in czirok2014cell)
        :param force_limit: float, F* from czirok2014cell
        :param p_add: float, base probability for adding tissue-tissue links
        :param p_del: float, base probability for removing tissue-tissue links
        :param p_add_subs: float, base probability for adding tissue-substrate links
        :param p_del_subs: float, base probability for removing tissue-substrate links
        :param chkx: bool, whether or not to check for link crossings (only functional for dims==2
        :param d0max: float, maximum cell-cell distance to allow a link to be added
        :param dims: 2 or 3, the number of dimensions of the simulations
        :param isF0: bool, whether or not external forces are a part of the problem
        :param isanchor: bool, whether or not tissue cells are anchored to a x0-position
        :param issubs: True (with substrate), False (without substrate) or "lonesome" (with substrate but only one
            tissue cell)
        :param force_contr: boolean, if False: update done as suggested in czirok2014cell. if True: force-dependent
            componentincluded.
        :return: instance of class CellMech
   
        
Add links between tissue cells by calling (on instance of class CellMech):

mynodes.addlink(ni, mi, t1=None, t2=None, d0=None, k=15., bend=10., twist=1., n=None, norm1=None, norm2=None)

        :param ni: integer, index of one of the cells involved in the link
        :param mi: integer, index of the second cell
        :param t1: numpy array of shape (3), the vector tangential to the link at the surface of cell ni or None
        :param t2: numpy array of shape (3), the vector tangential to the link at the surface of cell mi or None
        :param d0: float, the link's initial equilibrium length. If None: set to current link length
        :param k: float, the link's Hookean constant.
        :param bend: float, the link's bending rigidity
        :param twist: float, the link's twisting rigidity
        :param n: numpy array of shape (3), the chosen normal vector, or None
        :param norm1: numpy array of shape (3), the chosen normal vector at cell ni. If None: set to n
        :param norm2: numpy array of shape (3), the chosen normal vector at cell mi. If None: set to n
            

Add links between tissue and substrate cells by calling (on instance of class CellMech):

mysubs.addlink(ni, mi, cellx, cellphi, t1=None, d0=None, k=15., bend=10., twist=1., n=None, norm1=None, norm2=None)
            
        :param ni: integer, index of the tissue node involved in the link
        :param mi: integer, index of the substrate noed involved in the link
        :param cellx: position of the tissue node
        :param cellphi: orientation of the tissue node
        :param t1: numpy array of shape (3), the vector tangential to the link at the surface of the substrate cell mi
            or None
        :param d0: float, the link's initial equilibrium length. If None: set to current link length
        :param k: float, the link's Hookean constant.
        :param bend: float, the link's bending rigidity
        :param twist: float, the link's twisting rigidity
        :param n: numpy array of shape (3), the chosen normal vector, or None
        :param norm1: numpy array of shape (3), the chosen normal vector at tissue cell ni. If None: set to n
        :param norm2: numpy array of shape (3), the chosen normal vector at substrate cell mi. If None: set to n
            
            
Run simulation by calling (on instance of class CellMech): 

timeevo(tmax, isinit=True, isfinis=True, record=False, progress=True, dtsave=0)

        :param tmax: Maximum time for simulation run
        :param isinit: boolean, whether this is the first segment of a simulation run
        :param isfinis: boolean, whether this is the last segment of a simulation run
        :param record: boolean, whether to save simulation data for after code has finished
        :param progress: show progress bar
        :param dtsave: float, snapshot will be made of config after every tissue plasticity step if dtsave==0, otherwise
            each time t has crossed a new n*dtsave line
        :return: if self.issubs is False: Tuple of a) numpy array containing x-positions of tissue nodes at the end of 
            eachtime step, b) list containing numpy arrays of link index tuples, c) numpy array containing forces on 
            nodes for each time step, d) list containing numpy array for each timestep with forces on links, e) numpy 
            array of time at time steps
            else: additional tuple of f) numpy array containing fixed positions of substrate nodes, g) list of index 
            tuples for substrate-tissue links at time steps, h) numpy array of forces on substrate nodes at time steps, 
            i) list of forces on substrate-tissue links at time steps
            
            
Relaunch simulation by calling:

relaunch_CellMech(savedir, num_cells, num_subs=0, dt=0.01, nmax=300, qmin=0.001, d0_0=1., force_limit=15., p_add=1.,
                 p_del=0.2, p_add_subs=None, p_del_subs=None, chkx=False, d0max=2., dims=3, F_contr=1.,
                 isF0=False, isanchor=False, issubs=False, force_contr=True)
                 
        :param savedir: string, name of directory where previous data is saved
        :param num_cells: integer, the number of tissue cells
        :param num_subs: integer, the number of substrate cells
        :param dt: float, the time unit for scaling simulation time
        :param nmax: integer, the maximum time (in simulation time) for until cutoff when calculating
            mechanical equilibrium
        :param qmin: float, the square of the maximum force per cell until mechanical equilibration is cut off
        :param d0_0: float, the global equilibrium link length (d_0 in czirok2014cell)
        :param force_limit: float, F* from czirok2014cell
        :param p_add: float, base probability for adding tissue-tissue links
        :param p_del: float, base probability for removing tissue-tissue links
        :param p_add_subs: float, base probability for adding tissue-substrate links
        :param p_del_subs: float, base probability for removing tissue-substrate links
        :param chkx: bool, whether or not to check for link crossings (only functional for dims==2
        :param d0max: float, maximum cell-cell distance to allow a link to be added
        :param dims: 2 or 3, the number of dimensions of the simulations
        :param isF0: bool, whether or not external forces are a part of the problem
        :param isanchor: bool, whether or not tissue cells are anchored to a x0-position
        :param issubs: True (with substrate), False (without substrate) or "lonesome" (with substrate but only one
            tissue cell)
        :param force_contr: boolean, if False: update done as suggested in czirok2014cell. if True: force-dependent
            componentincluded.
        :return: Initiated instance of CellMech
        
        
************************************************************************************************************************

Main function from package "animate.py" relevant for using code:


Animate simulation results by calling: 

animateconfigs(Simdata, SubsSimdata=None, figureindex=0, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), figsize=(1000, 1000),
               cmap='viridis', cbar=False, showsubs=False)
               
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

</pre>
