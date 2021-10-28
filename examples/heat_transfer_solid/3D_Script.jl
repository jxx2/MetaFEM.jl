using MetaFEM
# Load mesh
dim = 3
fem_domain = FEM_Domain(; dim = dim)
element_shape = :SIMPLEX
src_fname = joinpath(@__DIR__, "3D_COMSOL_Mesh.mphtxt")
vert, connections = read_Mesh(src_fname)
ref_mesh = construct_TotalMesh(vert ./ 100, connections)
# Define Boundary
@Takeout (vertices, segments) FROM ref_mesh
sIDs = get_Boundary(ref_mesh)
wp_ID = add_WorkPiece(ref_mesh; fem_domain = fem_domain)
flux_bg_ID = add_BoundaryGroup(wp_ID, sIDs; fem_domain = fem_domain)
# Physics
T₀ = 273.15 + 20
k = 0.6 
h = 25. 
C = 1.
α = 0.
Tₑₙᵥ = T₀

@External_Sym (s, CONTROLPOINT_VAR)
@Def begin
    heat_dissipation = - k * Bilinear(T{;i}, T{;i}) + Bilinear(T, s + α * (Tₑₙᵥ - T))
    conv_boundary = h * Bilinear(T, Tₑₙᵥ - T) 
end
assign_WorkPiece_WeakForm(wp_ID, heat_dissipation; fem_domain = fem_domain)
assign_BoundaryGroup_WeakForm(wp_ID, flux_bg_ID, conv_boundary; fem_domain = fem_domain)
initialize_LocalAssembly(fem_domain.dim, fem_domain.workpieces; explicit_max_sd_order = 1)
## Assembly
mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)

@time begin
    for wp in fem_domain.workpieces
        update_Mesh(fem_domain.dim, wp, wp.element_space)
    end
    assemble_Global_Variables(fem_domain = fem_domain)
    compile_Updater_GPU(domain_ID = 1, fem_domain = fem_domain)
end
## Run
# fem_domain.linear_solver = x -> solver_IDRs(x; Pl_func = precondition_CUDA_Jacobi, max_iter = 5000, max_pass = 10, s = 8)
fem_domain.linear_solver = solver_LU_CPU 
fem_domain.globalfield.converge_tol = 1e-5

cpts = fem_domain.workpieces[1].mesh.controlpoints
cp_IDs = findall(cpts.is_occupied)
cpts.T[cp_IDs] .= Tₑₙᵥ 
cpts.s[cp_IDs] .= 1600. 

update_OneStep(fem_domain.time_discretization; fem_domain = fem_domain)
dessemble_X(fem_domain.workpieces, fem_domain.globalfield)
##
wp = fem_domain.workpieces[1]
write_VTK(string(@__DIR__, "\\", "3D_MetaFEM_Result.vtk"), wp; scale = 100)
##
# count element/node number
fem_domain.workpieces[1].mesh.elements.is_occupied |> sum |> println
fem_domain.workpieces[1].mesh.controlpoints.is_occupied |> sum |> println


