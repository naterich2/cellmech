# cellmech
<pre>

requires numpy, scipy for running simulations
requires mayavi for visualizing simulation results

Implementation of model for cell-resolved, multiparticle model of plastic tissue deformations and morphogenesis
first suggested by Czirok et al in 2014 (https://iopscience.iop.org/article/10.1088/1478-3975/12/1/016005/meta)
and first implemented in https://github.com/aczirok/cellmech.

Code contains extensions by Moritz Zeidler, Dept. for Innovative Methods of Computing (IMC),
Centre for Information Services and High Performance Computing (ZIH) at Technische Universitaet Dresden.
Contact via moritz.zeidler at tu-dresden.de or andreas.deutsch at tu-dresden.de

View files begining with "test_" for simple examples on how to use the code.

cell.py:
    the algorithm for running the model
    
animate.py:
    functions for 3D-animation of simulation results
    
makeanimation.py:
    code for animation of previously generated simulation results

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

</pre>
