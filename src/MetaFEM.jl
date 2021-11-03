module MetaFEM

using LinearAlgebra
using SparseArrays

using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER
CUDA.allowscalar(false)

# Framework
export FEM_Domain
export add_WorkPiece, add_Boundary

# Mesh
export make_Square, make_Brick
export read_Mesh, construct_TotalMesh, get_BoundaryMesh

# Symbolics
export @Sym, @External_Sym, @Def, @Takeout, VARIABLE_ATTRIBUTES
export assign_WorkPiece_WeakForm, assign_Boundary_WeakForm, initialize_LocalAssembly

# Assembly
export mesh_Classical, update_Mesh, assemble_Global_Variables, compile_Updater_GPU
export update_OneStep, dessemble_X

# Linear Solvers
export solver_LU_CPU, solver_QR, solver_LU, solver_BiCG, solver_IDRs

# Preconditioners
export precondition_CUDA_Jacobi, precondition_CUDA_ILU

#Visualizations
export write_VTK

filename_match(x) = match(r"(?<main_name>.*)\.(?<tail_name>[a-z]*$)", x)
function include_all_file_in_dir(this_dir::String)
    for file_name in readdir(this_dir)
        this_file_address = string(this_dir, "/", file_name)
        isdir(this_file_address) || include(this_file_address)
    end
end

function include_dir_from_base(this_name::String)
    this_dir = string(BASE_DIR, "/", this_name)
    include_all_file_in_dir(this_dir)
end

BASE_DIR = @__DIR__
include_dir_from_base("misc")
include_dir_from_base("symbolics")
include_dir_from_base("solver")
include_dir_from_base("solver/linear_solver")
include_dir_from_base("mesh/ref_geometry")
include_dir_from_base("mesh/spatial_discretization")
include_dir_from_base("mesh/unstructured_mesh")

# one-liner counts size including this file
count_Lines(path::String) = isdir(path) ? sum(count_Lines.(joinpath.(path, readdir(path)))) : length(readlines(path))
println("Loaded ", count_Lines(BASE_DIR), " lines of code in MetaFEM") 

end
##
