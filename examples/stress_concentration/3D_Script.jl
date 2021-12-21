using MetaFEM

fem_domain = FEM_Domain(dim = 3)
element_shape = :CUBE #this is because the original inp was made by CUBE
src_fname = joinpath(@__DIR__, "3D_Mesh.inp")
vert, connections = read_Mesh(src_fname)
ref_mesh = construct_TotalMesh(vert, connections)
# To define the boundary (facets), there should be an more elegant interface later
@Takeout (vertices, faces) FROM ref_mesh
facet_IDs = get_BoundaryMesh(ref_mesh)
vIDs = faces.vertex_IDs[:, facet_IDs] 
x1_mean = vec(sum(vertices.x1[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x2_mean = vec(sum(vertices.x2[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x3_mean = vec(sum(vertices.x3[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)

L, err_scale = 5., 0.05

facet_IDs_left = facet_IDs[(x1_mean .< err_scale) .& (x1_mean .> (.- err_scale))]
facet_IDs_right = facet_IDs[(x1_mean .< (L .+ err_scale)) .& (x1_mean .> (L .- err_scale))]
facet_IDs_front = facet_IDs[(x2_mean .< err_scale) .& (x2_mean .> (.- err_scale))]
facet_IDs_back = facet_IDs[(x2_mean .< (L .+ err_scale)) .& (x2_mean .> (L .- err_scale))]
facet_IDs_bottom = facet_IDs[(x3_mean .< err_scale) .& (x3_mean .> (.- err_scale))]
facet_IDs_top = facet_IDs[(x3_mean .< (L .+ err_scale)) .& (x3_mean .> (L .- err_scale))]

wp_ID = add_WorkPiece!(ref_mesh; fem_domain = fem_domain)
d1_fix_bg_ID = add_Boundary!(wp_ID, facet_IDs_left; fem_domain = fem_domain) #left fixed
d2_fix_bg_ID = add_Boundary!(wp_ID, facet_IDs_front; fem_domain = fem_domain) #left fixed
d3_fix_bg_ID = add_Boundary!(wp_ID, facet_IDs_bottom; fem_domain = fem_domain) #left fixed

free_bg_ID = add_Boundary!(wp_ID, vcat(facet_IDs_right, facet_IDs_top); fem_domain = fem_domain) #bot & right free
loaded_bg_ID = add_Boundary!(wp_ID, facet_IDs_back; fem_domain = fem_domain) #top loadeds

# To define the Physics
E = 210e9 #young's modulus
ν = 0.3
λ = E * ν / ((1 + ν) * (1 - 2 * ν))
μ = E / (2 * (1 + ν))
τᵇ = 10000 * E / L ^ 2

@Sym d
@External_Sym (dʷ, CONTROLPOINT_VAR) (σˡ, CONTROLPOINT_VAR, SYMMETRIC_TENSOR)
@Def ε{i,j} = (d{i;j} + d{j;i}) / 2.
@Def σ{i,j} = λ * δ{i,j} * ε{m,m} + 2. * μ * ε{i,j}
@Def Elastrostatic_Domain = - Bilinear(ε{i,j}, σ{i,j})

@Def begin
    WF_domain = Elastrostatic_Domain
    WF_d1_fixed_bdy = τᵇ * Bilinear(d{1}, (dʷ{1} - d{1}))
    WF_d2_fixed_bdy = τᵇ * Bilinear(d{2}, (dʷ{2} - d{2}))
    WF_d3_fixed_bdy = τᵇ * Bilinear(d{3}, (dʷ{3} - d{3}))
    WF_loaded_bdy = Bilinear(d{2}, σˡ{2,2} * n{2})
end

assign_WorkPiece_WeakForm!(wp_ID, WF_domain; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, d1_fix_bg_ID, WF_d1_fixed_bdy; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, d2_fix_bg_ID, WF_d2_fixed_bdy; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, d3_fix_bg_ID, WF_d3_fixed_bdy; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, loaded_bg_ID, WF_loaded_bdy; fem_domain = fem_domain)
initialize_LocalAssembly!(fem_domain.dim, fem_domain.workpieces)
## Assembly
mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)
for wp in fem_domain.workpieces
    update_Mesh(fem_domain.dim, wp, wp.element_space)
end
assemble_Global_Variables!(; fem_domain = fem_domain)
compile_Updater_GPU(; domain_ID = 1, fem_domain = fem_domain)
##
# fem_domain.linear_solver = x -> iterative_Solve!(x; Sv_func! = gmres!, maxiter = 2000, max_pass = 20, s = 20)
fem_domain.linear_solver = x -> iterative_Solve!(x; Sv_func! = idrs!, maxiter = 2000, max_pass = 20, s = 20)

fem_domain.globalfield.converge_tol = 1e-8
σ_external = 1

cp = fem_domain.workpieces[1].mesh.controlpoints
cp.σˡ2 .= σ_external

update_OneStep!(fem_domain.time_discretization; fem_domain = fem_domain)
dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)
## Visualization
wp = fem_domain.workpieces[1]
write_VTK(string(@__DIR__, "\\", "3D_MetaFEM.vtk"), wp)
#------------------------------
## native paraview line plot is not easy to unify format, so we 
#  re-plot the data (sampled in paraview and extracted as CSV) in Julia
#------------------------------
using CSV, DataFrames
using CairoMakie, Colors
Abaqus_2x = CSV.read(joinpath(@__DIR__, "2D_Abaqus_x.csv"), DataFrame)
Abaqus_2y = CSV.read(joinpath(@__DIR__, "2D_Abaqus_y.csv"), DataFrame)
Abaqus_3x = CSV.read(joinpath(@__DIR__, "3D_Abaqus_x.csv"), DataFrame)
Abaqus_3y = CSV.read(joinpath(@__DIR__, "3D_Abaqus_y.csv"), DataFrame)

MetaFEM_2x = CSV.read(joinpath(@__DIR__, "2D_MetaFEM_x.csv"), DataFrame)
MetaFEM_2y = CSV.read(joinpath(@__DIR__, "2D_MetaFEM_y.csv"), DataFrame)
MetaFEM_3x = CSV.read(joinpath(@__DIR__, "3D_MetaFEM_x.csv"), DataFrame)
MetaFEM_3y = CSV.read(joinpath(@__DIR__, "3D_MetaFEM_y.csv"), DataFrame)

fig = Figure(resolution = (1000, 600))
ax1 = fig[1, 1] = Axis(fig)
fontsize = 25

ax1.limits = (1.0, 5.0, -0.5, 3.5)
ax1.xticks = 1:0.5:5
ax1.yticks = -0.5:0.5:3.5
ax1.xlabel = "Normalized σ₂₂"
ax1.ylabel = "Normalized distance to hole center" 
ax1.xlabelsize = fontsize
ax1.ylabelsize = fontsize

plot_Abaqus_2x = scatter!(ax1, getfield(Abaqus_2x, :columns)[5], getfield(Abaqus_2x, :columns)[4], markersize = 10px, color = :skyblue1)
plot_Abaqus_2y = scatter!(ax1, getfield(Abaqus_2y, :columns)[2], getfield(Abaqus_2y, :columns)[4], markersize = 10px, color = :cadetblue3)
plot_Abaqus_3x = scatter!(ax1, getfield(Abaqus_3x, :columns)[3], getfield(Abaqus_3x, :columns)[2], markersize = 10px, color = :deepskyblue2)
plot_Abaqus_3y = scatter!(ax1, getfield(Abaqus_3y, :columns)[4], getfield(Abaqus_3y, :columns)[2], markersize = 10px, color = :blue)

plot_MetaFEM_2x = scatter!(ax1, getfield(MetaFEM_2x, :columns)[1], getfield(MetaFEM_2x, :columns)[10], markersize = 10px, color = :lightsalmon)
plot_MetaFEM_2y = scatter!(ax1, getfield(MetaFEM_2y, :columns)[end - 1], getfield(MetaFEM_2y, :columns)[10], markersize = 10px, color = :salmon1)
plot_MetaFEM_3x = scatter!(ax1, getfield(MetaFEM_3x, :columns)[end - 3], getfield(MetaFEM_3x, :columns)[1], markersize = 10px, color = :coral2)
plot_MetaFEM_3y = scatter!(ax1, getfield(MetaFEM_3y, :columns)[end - 1], getfield(MetaFEM_3y, :columns)[1], markersize = 10px, color = :red)

Legend(fig, [plot_Abaqus_2x, plot_Abaqus_2y, plot_Abaqus_3x, plot_Abaqus_3y, 
plot_MetaFEM_2x, plot_MetaFEM_2y, plot_MetaFEM_3x, plot_MetaFEM_3y], 
["Abaqus, 2D, x", "Abaqus, 2D, y", "Abaqus, 3D, x", "Abaqus, 3D, y", 
"MetaFEM, 2D, x", "MetaFEM, 2D, y", "MetaFEM, 3D, x", "MetaFEM, 3D, y"], bbox = (700, 900, 330, 530), labelsize = fontsize)
fig
##
save(string(@__DIR__, "\\", "3D_Thermal_Lines.png"), fig)