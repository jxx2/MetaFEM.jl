# # Linear elastic cantilever bending
# In this example we simulate a cantilever beam under different loads, as a straightforward check on the elasticity formulation by comparison to the analytical solutions.
# The source with data/visualization can also be found [here](https://github.com/jxx2/MetaFEM.jl/tree/main/examples/cantilever).
#
# First, we load the package and declare the domain:
using MetaFEM
fem_domain = FEM_Domain(dim = 3)
# ## Geometry
# For cuboid geometry we have helper functions "make\_Square"/"make\_Brick" for 2D/3D:
L_box, e_number, LW_ratio = 1., 4, 10
domain_size = (L_box * LW_ratio, L_box, L_box) 
element_number = (Int(e_number * LW_ratio / 4), e_number, e_number)
element_shape = :CUBE

vert, connections = make_Brick(domain_size, element_number, element_shape)
ref_mesh = construct_TotalMesh(vert, connections)
# Defining boundary can be a little lengthy because we need to find those IDs. 
# Any idea for improvement will be very welcomed.
@Takeout (vertices, faces) FROM ref_mesh
facet_IDs = get_BoundaryMesh(ref_mesh)
vIDs = faces.vertex_IDs[:, facet_IDs] 

x1_mean = vec(sum(vertices.x1[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x2_mean = vec(sum(vertices.x2[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x3_mean = vec(sum(vertices.x3[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)

err_scale = L_box / e_number * 0.01

facet_IDs_left = facet_IDs[(x1_mean .< err_scale) .& (x1_mean .> (.- err_scale))]
facet_IDs_right = facet_IDs[(x1_mean .< (L_box * LW_ratio .+ err_scale)) .& (x1_mean .> (L_box * LW_ratio.- err_scale))]
facet_IDs_front = facet_IDs[(x2_mean .< err_scale) .& (x2_mean .> (.- err_scale))]
facet_IDs_back = facet_IDs[(x2_mean .< (L_box .+ err_scale)) .& (x2_mean .> (L_box .- err_scale))]
facet_IDs_bottom = facet_IDs[(x3_mean .< err_scale) .& (x3_mean .> (.- err_scale))]
facet_IDs_top = facet_IDs[(x3_mean .< (L_box .+ err_scale)) .& (x3_mean .> (L_box .- err_scale))]

wp_ID = add_WorkPiece!(ref_mesh; fem_domain = fem_domain)
fix_bg_ID = add_Boundary!(wp_ID, facet_IDs_left; fem_domain = fem_domain) #left fixed
free_bg_ID = add_Boundary!(wp_ID, vcat(facet_IDs_front, facet_IDs_bottom, facet_IDs_top); fem_domain = fem_domain) #bottom & right & front free
back_bg_ID = add_Boundary!(wp_ID, facet_IDs_back; fem_domain = fem_domain) #back will be loaded
right_bg_ID = add_Boundary!(wp_ID, facet_IDs_right; fem_domain = fem_domain) #top will be loaded
# ## Physics
# The mathematical formulation is:
# ```math
# \text{variable}\quad d_i,\qquad\text{parameters}\quad E,\nu,d^w_i,\sigma^l_{ij},\tau
# ```
# ```math
# \lambda=\frac{E\nu}{(1+\nu)(1-2\nu)},\qquad\mu=\frac{E}{2(1+\nu)}
# ```
# ```math
# \epsilon_{ij}=\frac{d_{i,j}+d_{j,i}}{2},\qquad\sigma_{ij}=\lambda\delta_{ij}\epsilon_{mm}+2\mu\epsilon_{ij}
# ```
# ```math
# -(d_{i,j},\sigma_{ij})=0,\qquad in\quad\Omega
# ```
# ```math
# (d_i,\sigma^l_{ij}n_j)=0,\qquad on\quad(\partial\Omega)_{load}
# ```
# ```math
# (d_i,\tau(d^w_i-d_i))=0,\qquad on\quad(\partial\Omega)_{fix}
# ```
# where $d_i,E,\nu,\lambda,\mu,\epsilon_{ij},\sigma_{ij}$ are the displacement, Young’s modulus, Poisson ratio, Lame's first parameter, shear modulus, strain and stress, respectively.
# 
# The code is: (Note, if we set the Poisson ratio ν = 0.0, the spherical part, i.e., λ related terms, will be actaully removed by the simplification.)
Δx = L_box / e_number
E = 1
ν = 0.001
λ = E * ν / ((1 + ν) * (1 - 2 * ν))
μ = E / (2 * (1 + ν))
τᵇ = 1000 * E / L_box ^ 2

@Sym d
@External_Sym (dʷ, CONTROLPOINT_VAR) (σˡ, CONTROLPOINT_VAR, SYMMETRIC_TENSOR) (σ², CONTROLPOINT_VAR, SYMMETRIC_TENSOR)
@Def ε{i,j} = (d{i;j} + d{j;i}) / 2.
@Def σ{i,j} = λ * δ{i,j} * ε{m,m} + 2. * μ * ε{i,j}
@Def Elastrostatic_Domain = - Bilinear(ε{i,j}, σ{i,j})

@Def begin
    WF_domain = Elastrostatic_Domain
    WF_fixed_bdy = τᵇ * Bilinear(d{i}, (dʷ{i} - d{i}))

    WF_right_bdy = Bilinear(d{i}, σˡ{i,j} * n{j})
    WF_back_bdy = Bilinear(d{i}, σ²{i,j} * n{j})
end

assign_WorkPiece_WeakForm!(wp_ID, WF_domain; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, fix_bg_ID, WF_fixed_bdy; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, right_bg_ID, WF_right_bdy; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, back_bg_ID, WF_back_bdy; fem_domain = fem_domain)

# ## Assembly
initialize_LocalAssembly!(fem_domain)
mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)
compile_Updater_GPU(; domain_ID = 1, fem_domain = fem_domain)

# ## Run
# We run for three different loading conditions with the direct CPU LU solver for simplicity. For elasticity, most iterative solvers (with the default right Jacobi preconditioner) are stable:
for wp in fem_domain.workpieces
    update_Mesh(fem_domain.dim, wp, wp.element_space)
end
assemble_Global_Variables!(; fem_domain = fem_domain)
fem_domain.linear_solver = solver_LU_CPU
fem_domain.globalfield.converge_tol = 1e-5

dx = L_box/e_number
h, l = L_box, (L_box * LW_ratio)
I = 1/12 * h ^ 3
err_scale = 0.25

cpts = fem_domain.workpieces[1].mesh.controlpoints
horizontal_mid_cp_IDs = findall((cpts.x2 .> L_box / 2 - err_scale * dx) .& (cpts.x2 .< L_box / 2 + err_scale * dx) .& 
                                (cpts.x3 .> L_box / 2 - err_scale * dx) .& (cpts.x3 .< L_box / 2 + err_scale * dx))

x_plot = cpts.x1[horizontal_mid_cp_IDs] |> collect
sorted_ids = sortperm(x_plot)

horizontal_mid_cp_IDs = horizontal_mid_cp_IDs[MetaFEM.cu(sorted_ids)]
x_plot = x_plot[sorted_ids] 
y_plots = [[] for i = 1:2]
plot_labels = [String[] for i = 1:2]

σ_external = 1e6
cpts.σˡ6 .= σ_external
cpts.σ²2 .= 0

update_OneStep!(fem_domain.time_discretization; fem_domain = fem_domain)
dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)

y_plot_ana = σ_external * L_box/(6 * E * I) * (3 * l .-  x_plot) .* x_plot .^ 2
y_plot_num = cpts.d2[horizontal_mid_cp_IDs] |> collect
y_max = maximum(y_plot_ana)

push!(y_plots[1], y_plot_ana ./ y_max)
push!(y_plots[2], y_plot_num ./ y_max)
push!(plot_labels[1], "Concentrated load, analytical")
push!(plot_labels[2], "Concentrated load, MetaFEM")

cpts.σˡ6 .= 0
cpts.σ²2 .= σ_external

update_OneStep!(fem_domain.time_discretization; fem_domain = fem_domain)
dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)

y_plot_ana = σ_external / (24 * E * I) * (x_plot.^2 .+ 6 * l ^ 2 .- 4 * l .* x_plot) .* x_plot .^ 2
y_plot_num = cpts.d2[horizontal_mid_cp_IDs] |> collect
y_max = maximum(y_plot_ana)

push!(y_plots[1], y_plot_ana ./ y_max)
push!(y_plots[2], y_plot_num ./ y_max)
push!(plot_labels[1], "Uniform pressure, analytical")
push!(plot_labels[2], "Uniform pressure, MetaFEM")

cpts.σ²2 .= σ_external .* (1. .- cpts.x1 ./ (L_box * LW_ratio))

update_OneStep!(fem_domain.time_discretization; fem_domain = fem_domain)
dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)

y_plot_ana = σ_external / (120 * l * E * I) * (10 * l ^ 3 .- 10 * l^2 .* x_plot .+ 5 * l * x_plot .^ 2 .- x_plot .^ 3) .* x_plot .^ 2
y_plot_num = cpts.d2[horizontal_mid_cp_IDs] |> collect
y_max = maximum(y_plot_ana)

push!(y_plots[1], y_plot_ana ./ y_max)
push!(y_plots[2], y_plot_num ./ y_max)
push!(plot_labels[1], "Linearly distributed pressure, analytical")
push!(plot_labels[2], "Linearly distributed pressure, MetaFEM")
# ## Plot
# With Plots.jl:
using Plots
fig = plot(; size=(800,800), title = "Normalized deflection on the line y = z = 0.5", xlims = (-1., 11.), xticks = 0:2:10, ylims = (-0.1, 1.1), yticks = 0:0.2:1, xlabel = "x", ylabel = "Normalized d₂")
for i = 1:3
    color_val = (i / 3 + 1) / 2
    scatter!(fig, x_plot, y_plots[1][i], markershape = :rect, markersize = 6, color = RGBA(color_val, 0, 0, 1), label = plot_labels[1][i])
    plot!(fig, x_plot, y_plots[2][i], markershape = :circle, markersize = 3, color = RGBA(0, 0.5, color_val, 1), markercolor = RGBA(0, 0.5, color_val, 1), label = plot_labels[2][i])
end
fig.subplots[1].attr[:legend_position] = (0.2, 0.8)
png(fig, joinpath(@__DIR__, "3D_Cantilever_Plots.png"))
# ![cantilever](3D_Cantilever_Plots.png)