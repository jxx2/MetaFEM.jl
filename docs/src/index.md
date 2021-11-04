# MetaFEM.jl
Welcome to MetaFEM, a GPU-accelerated generic finite element solver by meta-expressions.

Function-wise, MetaFEM is similar to a skeleton version of FEniCS or FreeFEM, i.e., MetaFEM takes in some high-level mathematical expressions (PDE weak-forms) and a mesh, e.g., thermal conduction in a pikachu, then outputs the corresponding simulation, e.g., to VTK files, resulting in something similar to the logo through common softwares like Paraview. Different from the classical approaches, MetaFEM uses only the most fundamental concepts,
i.e., tensor components added, multiplied or algebraically operated together. There is no RHS/LHS, no Dirichlet boundary, etc.

Software-wise, MetaFEM is an original implementation from nearly scratch, i.e., MetaFEM is natively coded in Julia directly based on CUDA.jl, with the indirect adaptations from the other two external libraries: MacroTools.jl and IterativeSolvers.jl. MetaFEM V0.1.0 contains ~5000 lines of code.

MetaFEM contains:
1. A rule-based Computer Algebra System (CAS), i.e., symbolic differentiation and simplification;
2. A 2D/3D mesh system, with simplex/cube Lagrangian elements at arbitrary order and cubic serendipity elements at either order 2 or 3;
3. A FEM kernel that assembles everything and generates the code; and
4. The infrastructure to put the simulation on GPU.

Examples of 2D/3D thermal conduction, linear elasticity and incompressible flow are readily usable [here](https://github.com/jxx2/MetaFEM.jl/tree/main/examples), which can be a good starting point if you want a quick try and don't want to read the rest of the document. In each subfolder, the *.jl file is the main script that actually works. Paraview state file(s) *.pvsm may be also helpful in quick visualization, by open Paraview-File-Load State.

!!! note

    MetaFEM is in a primitive stage and under development. Any input, e.g., questions of usage, bugs, feature requests, or any ideas to make it better, will be very welcomed. Please feel free to open an issue on [Github](https://github.com/jxx2/MetaFEM) or directly email me at jiaxixie2022@u.northwestern.edu.

## Installation
* Through REPL-Pkg, i.e., Press `]` in Julia REPL to enter `pkg>`, then:
```
pkg> add MetaFEM
```
* Or directly:
```julia
julia> import Pkg; Pkg.add("MetaFEM")
```
