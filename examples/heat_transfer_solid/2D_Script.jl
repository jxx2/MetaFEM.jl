using MetaFEM
# Mesh
dim = 2
fem_domain = FEM_Domain(dim = dim)

L1, L2 = domain_size = (0.02, 0.01) 
Δx = 5e-4
element_number = Int.(domain_size ./ Δx)
element_shape = :CUBE

vertices, connections = make_Square(domain_size, element_number, element_shape)
ref_mesh = construct_TotalMesh(vertices, connections)
# Define Boundary
@Takeout (vertices, segments) FROM ref_mesh
sIDs = get_Boundary(ref_mesh)
v1IDs = segments.vertex_IDs[1, sIDs] 
v2IDs = segments.vertex_IDs[2, sIDs] 

x1_mean = (vertices.x1[v1IDs] .+ vertices.x1[v2IDs]) ./ 2
x2_mean = (vertices.x2[v1IDs] .+ vertices.x2[v2IDs]) ./ 2

err_scale = Δx * 0.01

sIDs_left = sIDs[(x1_mean .< err_scale) .& (x1_mean .> (.- err_scale))]
sIDs_right = sIDs[(x1_mean .< (L1 .+ err_scale)) .& (x1_mean .> (L1 .- err_scale))]
sIDs_bottom = sIDs[(x2_mean .< err_scale) .& (x2_mean .> (.- err_scale))]
sIDs_top = sIDs[(x2_mean .< (L2 .+ err_scale)) .& (x2_mean .> (L2 .- err_scale))]

wp_ID = add_WorkPiece(ref_mesh; fem_domain = fem_domain)
fixed_bg_ID = add_BoundaryGroup(wp_ID, vcat(sIDs_left, sIDs_right); fem_domain = fem_domain)
top_bg_ID = add_BoundaryGroup(wp_ID, sIDs_top; fem_domain = fem_domain)
# Physics
T₀ = 273.15
k = 3
h = 50
C = 1.
α = 0.
Tw = 900. + T₀
h_penalty = 100000.
Tₑₙᵥ = 50. + T₀
em = 0.7
σᵇ = 5.669e-8

@External_Sym (s, CONTROLPOINT_VAR)
@Def begin
    heat_dissipation = - k * Bilinear(T{;i}, T{;i}) + Bilinear(T, s + α * (Tₑₙᵥ - T))
    conv_rad_boundary = h * Bilinear(T, Tₑₙᵥ - T) + em * σᵇ * Bilinear(T, Tₑₙᵥ^4 - T^4)
    fix_boundary = h_penalty * Bilinear(T, Tw - T) + k * Bilinear(T, n{i} * T{;i}) 
end

assign_WorkPiece_WeakForm(wp_ID, heat_dissipation; fem_domain = fem_domain)
assign_BoundaryGroup_WeakForm(wp_ID, fixed_bg_ID, fix_boundary; fem_domain = fem_domain)
assign_BoundaryGroup_WeakForm(wp_ID, top_bg_ID, conv_rad_boundary; fem_domain = fem_domain)
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
fem_domain.linear_solver = solver_LU_CPU
fem_domain.globalfield.converge_tol = 1e-4

cpts = fem_domain.workpieces[1].mesh.controlpoints
cp_IDs = findall(cpts.is_occupied)
cpts.T[cp_IDs] .= Tₑₙᵥ 
cpts.s[cp_IDs] .= 0.

update_OneStep(fem_domain.time_discretization; fem_domain = fem_domain)
dessemble_X(fem_domain.workpieces, fem_domain.globalfield)
## Visualization
mid_cp_IDs = (cpts.x1 .> L1/2 - 0.1 * Δx) .& (cpts.x1 .< L1/2 + 0.1 * Δx)
num_ys = cpts.x2[mid_cp_IDs] |> collect
num_Ts = cpts.T[mid_cp_IDs] |> collect
# Directly copy and pasted from FEATool result
y_sample = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.0099]
T_sample = [1086.84,  1086,  1082.73,  1077.63,  1070.24,  1060.78,  1048.83,  1034.63,  1017.81,  998.843,  979.249]

using GLMakie
using Colors

fig = Figure(resolution = (1000, 600))
ax1 = fig[1, 1] = Axis(fig)

plot_sam = scatter!(ax1, y_sample, T_sample, markersize = 6px)
plot_num = scatter!(ax1, num_ys, num_Ts, markersize = 3px, color =:salmon)

ax1.title = "Temperature along the middle line x = 0.01"
ax1.xticks = 0.:0.002:0.01
ax1.xlabel = "y(m)"
ax1.ylabel = "T(K)" 
ax1.xlabelsize = 30
ax1.ylabelsize = 30
leg = fig[1, 2] = Legend(fig, [plot_sam, plot_num], ["FEATool", "This work"])
display(fig)
##
save(string(@__DIR__, "\\", "2D_Middle_Line.png"), fig)
## VTK
wp = fem_domain.workpieces[1]
write_VTK(string(@__DIR__, "\\", "2D_Ceramic_Strip.vtk"), wp)
