using MetaFEM
#------------------------------
# Load mesh
#------------------------------
dim = 3
fem_domain = FEM_Domain(; dim = dim)
element_shape = :SIMPLEX
src_fname = joinpath(@__DIR__, "3D_COMSOL_Mesh.mphtxt")
vert, connections = read_Mesh(src_fname)
ref_mesh = construct_TotalMesh(vert ./ 100, connections)
#------------------------------
# Define Boundary
#------------------------------
@Takeout (vertices, segments) FROM ref_mesh
sIDs = get_BoundaryMesh(ref_mesh)
wp_ID = add_WorkPiece(ref_mesh; fem_domain = fem_domain)
flux_bg_ID = add_Boundary(wp_ID, sIDs; fem_domain = fem_domain)
#------------------------------
# Physics
#------------------------------
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
assign_Boundary_WeakForm(wp_ID, flux_bg_ID, conv_boundary; fem_domain = fem_domain)
initialize_LocalAssembly(fem_domain.dim, fem_domain.workpieces; explicit_max_sd_order = 1)
#------------------------------
## Assembly
#------------------------------
mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)

@time begin
    for wp in fem_domain.workpieces
        update_Mesh(fem_domain.dim, wp, wp.element_space)
    end
    assemble_Global_Variables(fem_domain = fem_domain)
    compile_Updater_GPU(domain_ID = 1, fem_domain = fem_domain)
end
#------------------------------
## Run
#------------------------------
# fem_domain.linear_solver = x -> solver_IDRs(x; Pl_func = precondition_CUDA_Jacobi, max_iter = 5000, max_pass = 10, s = 8)
fem_domain.linear_solver = solver_LU_CPU 
fem_domain.globalfield.converge_tol = 1e-5

cpts = fem_domain.workpieces[1].mesh.controlpoints
cp_IDs = findall(cpts.is_occupied)
cpts.T[cp_IDs] .= Tₑₙᵥ 
cpts.s[cp_IDs] .= 1600. 

update_OneStep(fem_domain.time_discretization; fem_domain = fem_domain)
dessemble_X(fem_domain.workpieces, fem_domain.globalfield)
#------------------------------
## save
#------------------------------
wp = fem_domain.workpieces[1]
write_VTK(string(@__DIR__, "\\", "3D_MetaFEM_Result.vtk"), wp; scale = 100)
#------------------------------
## native paraview line plot is not easy to unify format, so we 
#  re-plot the data (sampled in paraview and extracted as CSV) in Julia
#------------------------------
using CSV, DataFrames
COMSOL_a = CSV.read(joinpath(@__DIR__, "COMSOL_a.csv"), DataFrame)
COMSOL_b = CSV.read(joinpath(@__DIR__, "COMSOL_b.csv"), DataFrame)
MetaFEM_a = CSV.read(joinpath(@__DIR__, "MetaFEM_a.csv"), DataFrame)
MetaFEM_b = CSV.read(joinpath(@__DIR__, "MetaFEM_b.csv"), DataFrame)

fig = Figure(resolution = (1000, 600))
ax1 = fig[1, 1] = Axis(fig)
fontsize = 25

ax1.title = "Temperature at sample points along two vertical lines"
ax1.titlesize = fontsize
ax1.xticks = 0:5:50
ax1.yticks = 295:2:303
ax1.xlabel = "y(cm)"
ax1.ylabel = "T(K)" 
ax1.xlabelsize = fontsize
ax1.ylabelsize = fontsize

plot_comsol_a = scatter!(ax1, COMSOL_a.arc_length, COMSOL_a.Temperature, markersize = 10px, color = :blue)
plot_meta_a = scatter!(ax1, MetaFEM_a.arc_length, MetaFEM_a.T, markersize = 6px, color =:red)

plot_comsol_b = scatter!(ax1, COMSOL_b.arc_length, COMSOL_b.Temperature, markersize = 10px, color = :dodgerblue4)
plot_meta_b = scatter!(ax1, MetaFEM_b.arc_length, MetaFEM_b.T, markersize = 6px, color =:salmon)

Legend(fig, [plot_comsol_a, plot_comsol_b, plot_meta_a, plot_meta_b], 
["COMSOL, line a", "COMSOL, line b", "MetaFEM, line a", "MetaFEM, line b"], bbox = (400, 500, 100, 300), labelsize = fontsize)
fig
##
save(string(@__DIR__, "\\", "3D_Thermal_Lines.png"), fig)