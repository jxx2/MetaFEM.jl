## Core
MetaFEM has three basic data structures:
* FEM_Domain, the overall system, defines $Kx=d$.
* WorkPiece, a mesh assigned with physics, defines both domain physics and other numerical modifications like stabilization.
* Boundary, an array of segment/face IDs assigned with physics, defines boundary condition.
```@docs
    FEM_Domain
    MetaFEM.WorkPiece
    add_Boundary!
```
## Mesh
The input to define a mesh is (vert, connections), very similar to the mesh loading process in other fields, e.g., for plotting
* vert is a $dim \times N_n$ CuArray of coordinates, where $N_n$ is the number of nodes.
* connections is a $N_v \times N_e$ CuArray of sequenced node IDs where $N_v$ is the number of nodes in each element and $N_e$ is the total element number.

To define (vert, connections), we have the following helper functions:
```@docs
    make_Square
    read_Mesh
```
With (vert, connections), the first order mesh can be defined by:
```@docs
    construct_TotalMesh
```
We also have the helper function to define boundaries:
```@docs
    get_BoundaryMesh
```

## Symbolics
The APIs to define the physics are pretty straightforward:
```@docs
    @Sym
    @Def
    assign_Boundary_WeakForm!
    visualize
```

## Assembly
There are some necessary preprocesses:
```@docs
    initialize_LocalAssembly!
    mesh_Classical
    compile_Updater_GPU
```

## Update simulation
```@docs
    update_Mesh
    assemble_Global_Variables!
    update_OneStep!
    assemble_X!
    dessemble_X!
```

## Linear solvers
In update\_OneStep, the linear solver, fem\_domain.linear\_solver is called, which is supposed to take a globalfield and return the dx.
We provide the following linear solvers:
```@docs
    solver_LU_CPU
    solver_QR_GPU
    solver_LU_GPU
    iterative_Solve!
    bicgstabl!
    bicgstabl_GS!
    idrs!
    idrs_original!
    gmres!
    lsqr!
    cgs!
```
with the following preconditioners:
```@docs
    Identity
    Pr_Jacobi!
    Pl_Jacobi
    Pl_ILU
```
where the prefix `Pl` denotes left preconditioners while the prefix `Pr`denotes right preconditioners. 

## Other helper functions
```@docs
    write_VTK
    @Takeout
    MEM_UNIT
```
