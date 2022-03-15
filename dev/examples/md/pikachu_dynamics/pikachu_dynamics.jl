using MetaFEM
initialize_Definitions!()
fem_domain = FEM_Domain(; dim = 3)
element_shape = :SIMPLEX
src_fname = joinpath(@__DIR__, "3D_COMSOL_Mesh.mphtxt")
vert, connections = read_Mesh(src_fname)
ref_mesh = construct_TotalMesh(vert ./ 100, connections)

wp_ID = add_WorkPiece!(ref_mesh; fem_domain = fem_domain)
fIDs = get_BoundaryMesh(ref_mesh)
flux_bg_ID = add_Boundary!(wp_ID, fIDs; fem_domain = fem_domain)

C = 4.184 * 1e3 # C is simply chosen arbitrarily for convenience
k = 0.6
h = 25.
Tₑₙᵥ = 273.15 + 20
α = 0.
@Sym T
@External_Sym (s, CONTROLPOINT_VAR)
@Def begin
    heat_dissipation = - C * Bilinear(T, T{;t}) - k * Bilinear(T{;i}, T{;i}) + Bilinear(T, s + α * (Tₑₙᵥ - T))
    conv_boundary = h * Bilinear(T, Tₑₙᵥ - T)
end
assign_WorkPiece_WeakForm!(wp_ID, heat_dissipation; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, flux_bg_ID, conv_boundary; fem_domain = fem_domain)

initialize_LocalAssembly!(fem_domain; explicit_max_sd_order = 1)
mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)
compile_Updater_GPU(domain_ID = 1, fem_domain = fem_domain)

for wp in fem_domain.workpieces
    update_Mesh(fem_domain.dim, wp, wp.element_space)
end
assemble_Global_Variables!(fem_domain = fem_domain)

fem_domain.linear_solver = x -> iterative_Solve!(x; Sv_func! = idrs!, maxiter = 2000, max_pass = 10, s = 8)
fem_domain.globalfield.converge_tol = 1e-6
fem_domain.globalfield.dt = 1

cpts = fem_domain.workpieces[1].mesh.controlpoints
cp_IDs = findall(cpts.is_occupied)
cpts.T[cp_IDs] .= Tₑₙᵥ
cpts.s[cp_IDs] .= 1600.

assemble_X!(fem_domain.workpieces, fem_domain.globalfield)

for i = 1:10
    update_OneStep!(fem_domain.time_discretization; fem_domain = fem_domain)
    dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)

    wp = fem_domain.workpieces[1]
    write_VTK(joinpath(@__DIR__, "history", "3D_MetaFEM_Result_$i.vtk"), wp; scale = 100)
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

