using MetaFEM

fem_domain = FEM_Domain(; dim = 3)

element_shape = :SIMPLEX
src_fname = joinpath(@__DIR__, "3D_COMSOL_Mesh.mphtxt")
vert, connections = read_Mesh(src_fname)
ref_mesh = construct_TotalMesh(vert ./ 100, connections)

wp_ID = add_WorkPiece(ref_mesh; fem_domain = fem_domain)

fIDs = get_BoundaryMesh(ref_mesh)
flux_bg_ID = add_Boundary(wp_ID, fIDs; fem_domain = fem_domain)

C = 1.
k = 0.6
h = 25.
Tₑₙᵥ = 273.15 + 20

@External_Sym (s, CONTROLPOINT_VAR)
@Def begin
    heat_dissipation = - k * Bilinear(T{;i}, T{;i}) + Bilinear(T, s)
    conv_boundary = h * Bilinear(T, Tₑₙᵥ - T)
end
assign_WorkPiece_WeakForm(wp_ID, heat_dissipation; fem_domain = fem_domain)
assign_Boundary_WeakForm(wp_ID, flux_bg_ID, conv_boundary; fem_domain = fem_domain)

initialize_LocalAssembly(fem_domain.dim, fem_domain.workpieces; explicit_max_sd_order = 1)

mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)

compile_Updater_GPU(domain_ID = 1, fem_domain = fem_domain)

for wp in fem_domain.workpieces
    update_Mesh(fem_domain.dim, wp, wp.element_space)
end

assemble_Global_Variables(fem_domain = fem_domain)

fem_domain.linear_solver = solver_LU_CPU
fem_domain.globalfield.converge_tol = 1e-5

cpts = fem_domain.workpieces[1].mesh.controlpoints
cp_IDs = findall(cpts.is_occupied)
cpts.T[cp_IDs] .= Tₑₙᵥ # Static problem doesn't need initial values, but just for completeness. The default initial values are zeros.
cpts.s[cp_IDs] .= 1600.

update_OneStep(fem_domain.time_discretization; fem_domain = fem_domain)

dessemble_X(fem_domain.workpieces, fem_domain.globalfield)

write_VTK(joinpath(@__DIR__, "3D_MetaFEM_Result.vtk"), fem_domain.workpieces[1]; scale = 100)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

