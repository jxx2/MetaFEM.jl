# MetaFEM
A GPU-accelerated generic FEM solver by meta-expressions

## Documentation

[![][docs-dev-img]][docs-dev-url] [![][docs-paper-img]][docs-paper-url]

## Overview

MetaFEM is initially a research project in the [Advanced Manufacturing Processes Laboratory](http://ampl.mech.northwestern.edu/index.html) of Northwestern University, attempting to provide fast, highly customizable evaluations for practical manufacturing processes.

Function-wise, MetaFEM is similar to a skeleton version of FEniCS or FreeFEM, i.e., MetaFEM takes in some high-level mathematical expressions (PDE weak-forms) and a mesh, e.g., thermal conduction in a pikachu, then outputs the corresponding simulation, e.g., to VTK files, resulting in something similar to the logo through common softwares like Paraview. Different from the classical approaches, however, MetaFEM uses only the most fundamental concepts, i.e., tensor components added, multiplied or algebraically operated in the variational form(s). There is no RHS/LHS, no Dirichlet boundary, etc., but only the variational residue in the component form, and the script (describing physics) looks just like the mathematical expressions on a mechanics class.

Quick examples can be found in [documents](https://jxx2.github.io/MetaFEM.jl/dev/examples/md/pikachu/pikachu/) and [sources](https://github.com/jxx2/MetaFEM/tree/main/examples), where "*.jl" files are the main scripts. 

## Known Issue(s):
1. We observed: when multiple examples are run in sequence, there can be a specific repeatable crash with the message "an illegal memory access was encountered (code 700, ERROR_ILLEGAL_ADDRESS)" for each specific machine (or maybe each different GPU memory size). Manually reclaim memory, e.g.,
```julia
MetaFEM.GC.gc(true)
MetaFEM.CUDA.reclaim()
```
doesn't work either. Using Makie.jl will result an earlier crash.
Currently we restart julia when we encounter this and plot with more lightweight Plots.jl.

## Related works (in Julia)
* [Gridap.jl](https://github.com/gridap/Gridap.jl), is a more classical approach on the generic FEM solver, similar to (a Julia version of) FEniCS/FreeFEM.
* [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl), is a mesh system/finite element toolbox. 
* [JuliaFEM.jl](https://github.com/JuliaFEM/JuliaFEM.jl), is a solid mechanics solver.

Please feel free to massage me in any form on any other package and I will be very happy to cite your work or collaborate. 
(The developer/author has very limited knowledge on this matter.)

## Novelty (from other generic FEM solvers)
MetaFEM is formulated in its own original theory:
* From the symbolics perspective, MetaFEM is rewriting-based, by [rewriting rules](https://github.com/jxx2/MetaFEM.jl/blob/main/src/symbolics/101_Simplify_Rule.jl), for customization.
* From the FEM perspective, MetaFEM is based on meta-expressions, which results in a compact codebase and a small API function number, e.g., there is no helper function like grad/∇, div, etc. 
* From the software perspective, MetaFEM is fully vectorized and GPU-accelerated by design. Core datastructures:
  * [GPU_Table](https://github.com/jxx2/MetaFEM.jl/blob/main/src/misc/05_GPU_Table.jl), the struct of arrays.
  * [GPU_Dict](https://github.com/jxx2/MetaFEM.jl/blob/main/src/misc/06_GPU_Dict.jl), the hash table.

## Current Status
The package is usable as a single GPU-accelerated generic FEM solver.  

The first version of the document is considered completed, although it will be further extended in the following weeks.
  
More features are in developement, e.g., more examples, distributed computing, cutcell mesh, etc.. 

The immediate next step is to make the simulation distributed, denoted by the version v0.2.0, which will be the first formal release allowing a simulation at the practical scale. Before that, any minor bug-fix will be denoted as v0.1.x.

[docs-dev-img]: https://img.shields.io/badge/docs-latest%20release-blue
[docs-dev-url]: https://jxx2.github.io/MetaFEM.jl/dev/

[docs-paper-img]: https://img.shields.io/badge/paper-arxiv-blue
[docs-paper-url]: https://arxiv.org/abs/2111.03541
