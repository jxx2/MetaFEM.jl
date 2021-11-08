# MetaFEM
A GPU-accelerated generic FEM solver by meta-expressions

## Documentation

[![][docs-dev-img]][docs-dev-url] [![][docs-paper-img]][docs-paper-url]

## Overview

MetaFEM is initially a research project in the [Advanced Manufacturing Processes Laboratory](http://ampl.mech.northwestern.edu/index.html) of Northwestern University, attempting to provide fast, highly customizable evaluations for practical manufacturing processes.

Function-wise, MetaFEM is similar to a skeleton version of FEniCS or FreeFEM, i.e., MetaFEM takes in some high-level mathematical expressions (PDE weak-forms) and a mesh, e.g., thermal conduction in a pikachu, then outputs the corresponding simulation, e.g., to VTK files, resulting in something similar to the logo through common softwares like Paraview. Different from the classical approaches, however, MetaFEM uses only the most fundamental concepts, i.e., tensor components added, multiplied or algebraically operated in the variational form(s). There is no RHS/LHS, no Dirichlet boundary, no helper functions like grad/∇, etc., but only the variational residue in the component form, and the script (describing physics) looks just like the mathematical expressions on a mechanics class.

Quick examples can be found in [documents](https://jxx2.github.io/MetaFEM.jl/dev/examples/md/pikachu/pikachu/) and [sources](https://github.com/jxx2/MetaFEM/tree/main/examples), where "*.jl" files are the main scripts. 

## Current Status
The package is usable as a single GPU "classical" generic FEM solver.  
The project is in its initial release, and the documents are to be extended in the following days.  
More features are in developement, e.g., distributed computing, etc.. 

[docs-dev-img]: https://img.shields.io/badge/docs-latest%20release-blue
[docs-dev-url]: https://jxx2.github.io/MetaFEM.jl/dev/

[docs-paper-img]: https://img.shields.io/badge/paper-arxiv-blue
[docs-paper-url]: https://arxiv.org/abs/2111.03541
