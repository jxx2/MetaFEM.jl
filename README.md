# MetaFEM
A GPU-accelerated generic FEM solver by meta-expressions
## Current Status
The package is usable as a single GPU-accelerated generic FEM solver. No active developement.

There are some issues with name export. In the example scripts, "using .MetaFEM" instead of "using MetaFEM" is a workaround.

The package is under [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html).  

## Documentation

[![][docs-dev-img]][docs-dev-url] [![][docs-paper-img]][docs-paper-url]

## Overview

MetaFEM is initially a research project in the [Advanced Manufacturing Processes Laboratory](http://ampl.mech.northwestern.edu/index.html) of Northwestern University, attempting to provide fast, highly customizable evaluations for practical manufacturing processes.

Function-wise, MetaFEM is similar to a skeleton version of FEniCS or FreeFEM, i.e., MetaFEM takes in some high-level mathematical expressions (PDE weak-forms) and a mesh, e.g., thermal conduction in a pikachu, then outputs the corresponding simulation, e.g., to VTK files, resulting in something similar to the logo through common softwares like Paraview. Different from the classical approaches, however, MetaFEM uses only the most fundamental concepts, i.e., tensor components added, multiplied or algebraically operated in the variational form(s). There is no RHS/LHS, no Dirichlet boundary, etc., but only the variational residue in the component form, and the script (describing physics) looks just like the mathematical expressions on a mechanics class.

Quick examples can be found in [documents](https://jxx2.github.io/MetaFEM.jl/dev/examples/md/pikachu/pikachu/) and [sources](https://github.com/jxx2/MetaFEM/tree/main/examples), where "*.jl" files are the main scripts. 

## Related works (in Julia)
* [Gridap.jl](https://github.com/gridap/Gridap.jl), is a more classical approach on the generic FEM solver, similar to (a Julia version of) FEniCS/FreeFEM.
* [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl), is a mesh system/finite element toolbox. 
* [JuliaFEM.jl](https://github.com/JuliaFEM/JuliaFEM.jl), is a solid mechanics solver.



[docs-dev-img]: https://img.shields.io/badge/docs-latest%20release-blue
[docs-dev-url]: https://jxx2.github.io/MetaFEM.jl/dev/

[docs-paper-img]: https://img.shields.io/badge/paper-arxiv-blue
[docs-paper-url]: https://arxiv.org/abs/2111.03541
