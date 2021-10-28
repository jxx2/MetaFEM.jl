using MetaFEM
# Mesh
dim = 2
fem_domain = FEM_Domain(dim = dim)
L_box, e_number = 1., 40
domain_size = (L_box, L_box) 
element_number = (e_number, e_number)
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

err_scale = L_box / e_number * 0.01

sIDs_left = sIDs[(x1_mean .< err_scale) .& (x1_mean .> (.- err_scale))]
sIDs_right = sIDs[(x1_mean .< (L_box .+ err_scale)) .& (x1_mean .> (L_box .- err_scale))]
sIDs_bottom = sIDs[(x2_mean .< err_scale) .& (x2_mean .> (.- err_scale))]
sIDs_top = sIDs[(x2_mean .< (L_box .+ err_scale)) .& (x2_mean .> (L_box .- err_scale))]

wp_ID = add_WorkPiece(ref_mesh; fem_domain = fem_domain)
fixed_bg_ID = add_BoundaryGroup(wp_ID, vcat(sIDs_left, sIDs_bottom, sIDs_right); fem_domain = fem_domain)
top_bg_ID = add_BoundaryGroup(wp_ID, sIDs_top; fem_domain = fem_domain)
# Physics
Δx = L_box / e_number
ρ = 1e3
μ = 1.
ν = μ / ρ
Cᵇ = 128 # 8, 16, 32
τᵇ = μ / ρ * Cᵇ / Δx

@Sym u p
@External_Sym (uʷ, CONTROLPOINT_VAR) (τᵐ, CONTROLPOINT_VAR) (τᶜ, CONTROLPOINT_VAR)

@Def begin
    Rc = u{m;m}
    Rm{i} = u{m} * u{i;m} + p{;i} / ρ - μ / ρ * u{i;m,m}
end

@Def begin
    NS_domain_BASE = - ρ * Bilinear(u{i;j}, u{i} * u{j}) - Bilinear(u{i;i}, p) + Bilinear(p, u{i;i}) + μ * Bilinear(u{i;j}, u{i;j})

    NS_domain_SUPG = τᵐ * ρ * Bilinear(u{i;j}, Rm{i} * u{j}) + τᵐ * Bilinear(p{;i}, Rm{i}) + τᶜ * ρ * Bilinear(u{i;i}, Rc)
    NS_boundary_BASE = ρ * Bilinear(u{i}, u{i} * u{j} * n{j}) + Bilinear(u{i}, p * n{i}) - μ * Bilinear(u{i}, u{i;j} * n{j})

    NS_boundary_DISP = ρ * Bilinear(u{i}, (uʷ{i} * uʷ{j} - u{i} * u{j}) * n{j}) + Bilinear(p, (uʷ{i} - u{i}) * n{i}) +
                        μ * Bilinear(u{i;j}, (uʷ{i} - u{i}) * n{j}) + τᵇ * ρ * Bilinear(u{i}, u{i} - uʷ{i})

    NS_boundary_FIX = ρ * Bilinear(u{i}, - u{i} * u{j} * n{j}) + Bilinear(p, - u{i} * n{i}) +
                      μ * Bilinear(u{i;j}, - u{i} * n{j}) + τᵇ * ρ * Bilinear(u{i}, u{i})
end

@Def begin
    WF_domain = NS_domain_BASE + NS_domain_SUPG
    WF_boundary_top = NS_boundary_BASE + NS_boundary_DISP
    WF_boundary_fix = NS_boundary_BASE + NS_boundary_FIX
end

@time begin
    assign_WorkPiece_WeakForm(wp_ID, WF_domain; fem_domain = fem_domain)
    assign_BoundaryGroup_WeakForm(wp_ID, fixed_bg_ID, WF_boundary_fix; fem_domain = fem_domain)
    assign_BoundaryGroup_WeakForm(wp_ID, top_bg_ID, WF_boundary_top; fem_domain = fem_domain)
    initialize_LocalAssembly(fem_domain.dim, fem_domain.workpieces; explicit_max_sd_order = 1)
end
## Assembly
mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)

@time begin
    for wp in fem_domain.workpieces
        update_Mesh(fem_domain.dim, wp, wp.element_space)
    end
    assemble_Global_Variables(fem_domain = fem_domain)
    compile_Updater_GPU(domain_ID = 1, fem_domain = fem_domain)
end
## Run & Visualization
fem_domain.linear_solver = solver_LU_CPU # CPU LU is practically faster
# fem_domain.linear_solver = x -> solver_LU(x; reorder = 1, singular_tol = 1e-20) 
# fem_domain.linear_solver = x -> solver_QR(x; reorder = 1, singular_tol = 1e-20)
fem_domain.globalfield.converge_tol = 1e-4

using GLMakie, Colors
using CSV, DataFrames

fig = Figure(resolution = (1400, 900), backgroundcolor = RGBf0(0.98, 0.98, 0.98))
ax1 = fig[1, 1] = Axis(fig, title = "Horizontal Velocity on Line x = 0.5",
                    xticks = (-0.4:0.2:1, string.(-0.4:0.2:1)), xlims = (-0.4, 1.0),
                    ylims = (0,1), xlabel = "Normalized U₁", ylabel = "y")
exp_plots, num_plots = [], []

Re_arr = [100, 400, 1000, 3200, 5000]
# Re_arr = [100]
for (color_id, Re) in pairs(Re_arr)

    u_st = Re / L_box * μ / ρ
    fem_domain.globalfield.x .= 0.
    fem_domain.globalfield.t = 0.
    dessemble_X(fem_domain.workpieces, fem_domain.globalfield)

    tmax = Re > 1000 ? 10 : ceil(Re / 100) |> Int
    for i = 1:tmax
        cpts = fem_domain.workpieces[1].mesh.controlpoints
        cp_IDs = findall(cpts.is_occupied)

        u_top = u_st * (i / tmax)
        dt = fem_domain.time_discretization.dt = 0.2 * Δx / u_top

        cpts.uʷ1[cp_IDs] .= u_top
        cpts.τᵐ[cp_IDs] .= (4 / dt ^ 2 + 9 * 16 * ν ^ 2 * dim * Δx ^ (-4) .+ Δx ^ (-2) * (cpts.u1[cp_IDs] .^ 2. + cpts.u2[cp_IDs] .^ 2.)) .^ (-0.5)
        cpts.τᶜ[cp_IDs] .= (cpts.τᵐ[cp_IDs] .* (dim * Δx ^ (-2))) .^ (-1.)

        println("Timestep ", i, " velocity = ", u_top, " tol = ", fem_domain.globalfield.converge_tol)
        update_OneStep(fem_domain.time_discretization; max_iter = 6, fem_domain = fem_domain)
        dessemble_X(fem_domain.workpieces, fem_domain.globalfield)
    end

    filename = string("Ghia_Re", Re, ".csv")
    folder = @__DIR__
    total_path = string(folder, "\\", filename)
    file_data = CSV.read(total_path, DataFrame)

    cpts = fem_domain.workpieces[1].mesh.controlpoints
    @Takeout x1, x2, u1 FROM cpts
    is_occupied = cpts.is_occupied
    dx = L_box/e_number 

    mid_cp_IDs = (x1 .> L_box/2 - 0.25 * dx) .& (x1 .< L_box/2 + 0.25 * dx)
    u_plot = u1[mid_cp_IDs]./u_st |> collect
    y_plot = x2[mid_cp_IDs]./L_box |> collect

    color_val = (color_id / length(Re_arr) + 1) / 2
    ids = sortperm(y_plot)
    exp_plot = scatter!(ax1, file_data.u, file_data.y, marker = '■', markersize = 10px, color = RGBf0(color_val, 0, 0))
    num_plot = scatterlines!(u_plot[ids], y_plot[ids], marker = :circle, markersize = 5px, color = RGBf0(0, 0.5, color_val), markercolor = RGBf0(0, 0.5, color_val))

    push!(exp_plots, exp_plot)
    push!(num_plots, num_plot)
end
exp_plot_names = (x -> string("Re", x, ", Ghia")).(Re_arr)
num_plot_names = (x -> string("Re", x, ", This_work")).(Re_arr)

leg = fig[1, end + 1] = Legend(fig, [exp_plots..., num_plots...], [exp_plot_names..., num_plot_names...])
display(fig)
##
save(string(folder, "\\", "2D_Ux.png"), fig)
## VTK
fem_domain.linear_solver = solver_LU_CPU
fem_domain.globalfield.converge_tol = 1e-5

Re = 1000
u_st = Re / L_box * μ / ρ
fem_domain.globalfield.x .= 0.
fem_domain.globalfield.t = 0
dessemble_X(fem_domain.workpieces, fem_domain.globalfield)

tmax = Re > 1000 ? 10 : ceil(Re / 100) |> Int
for i = 1:tmax
    cpts = fem_domain.workpieces[1].mesh.controlpoints
    cp_IDs = findall(cpts.is_occupied)

    u_top = u_st * (i / tmax)
    dt = fem_domain.time_discretization.dt = 0.2 * Δx / u_top

    cpts.uʷ1[cp_IDs] .= u_top
    cpts.τᵐ[cp_IDs] .= (4 / dt ^ 2 + 9 * 16 * ν ^ 2 * dim * Δx ^ (-4) .+ Δx ^ (-2) * (cpts.u1[cp_IDs] .^ 2. + cpts.u2[cp_IDs] .^ 2.)) .^ (-0.5)
    cpts.τᶜ[cp_IDs] .= (cpts.τᵐ[cp_IDs] .* (dim * Δx ^ (-2))) .^ (-1.)

    println("Timestep ", i, " velocity = ", u_top, " tol = ", fem_domain.globalfield.converge_tol)
    update_OneStep(fem_domain.time_discretization; max_iter = 6, fem_domain = fem_domain)
    dessemble_X(fem_domain.workpieces, fem_domain.globalfield)
end
##
wp = fem_domain.workpieces[1]
write_VTK(string(folder, "\\", "2D_Cavity_Flow.vtk"), wp)