using MetaFEM
#------------------------------
# Load mesh
#------------------------------
dim = 3
fem_domain = FEM_Domain(; dim = dim)
L = 2.5
H = 0.41
element_shape = :SIMPLEX
src_fname = joinpath(@__DIR__, "3D_COMSOL_Mesh.mphtxt")
vert, connections = read_Mesh(src_fname)
ref_mesh = construct_TotalMesh(vert, connections)
#------------------------------
# Define boundary
#------------------------------
@Takeout (vertices, faces) FROM ref_mesh
facet_IDs = get_BoundaryMesh(ref_mesh)
vIDs = faces.vertex_IDs[:, facet_IDs] 
x1_mean = vec(sum(vertices.x1[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x2_mean = vec(sum(vertices.x2[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x3_mean = vec(sum(vertices.x3[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)

err_scale = 0.01

is_left = (x1_mean .< err_scale) .& (x1_mean .> (.- err_scale))
is_right = (x1_mean .< (L .+ err_scale)) .& (x1_mean .> (L .- err_scale))

wp_ID = add_WorkPiece(ref_mesh; fem_domain = fem_domain)
fixed_bg_ID = add_Boundary(wp_ID, facet_IDs[.~(is_left .| is_right)]; fem_domain = fem_domain)
inflow_bg_ID = add_Boundary(wp_ID, facet_IDs[is_left]; fem_domain = fem_domain)
outflow_bg_ID = add_Boundary(wp_ID, facet_IDs[is_right]; fem_domain = fem_domain)
#------------------------------
# Define physics
#------------------------------
Δx = 0.02 # r = 0.1
ρ = 1e3
μ = 1.
ν = μ / ρ
Cᵇ = 128
τᵇ = μ / ρ * Cᵇ / Δx
τᵖ = Cᵇ * Δx / μ

@Sym u p
@External_Sym (uʷ, CONTROLPOINT_VAR) (τᵐ, CONTROLPOINT_VAR) (τᶜ, CONTROLPOINT_VAR)

@Def begin
    Rc = u{m;m}
    Rm{i} = u{m} * u{i;m} + p{;i} / ρ - μ / ρ * u{i;m,m}
end

@Def begin
    NS_domain_BASE = - ρ * Bilinear(u{i;j}, u{i} * u{j}) - Bilinear(u{i;i}, p) + Bilinear(p, u{i;i}) + μ * Bilinear(u{i;j}, u{i;j})
    NS_domain_SUPG = τᵐ * ρ * Bilinear(u{i;j}, Rm{i} * u{j}) + τᵐ * Bilinear(p{;i}, Rm{i}) + τᶜ * ρ * Bilinear(u{i;i}, Rc)

    NS_boundary_BASE = Bilinear(u{i}, p * n{i}) - μ * Bilinear(u{i}, u{i;j} * n{j})
    NS_boundary_INFLOW = ρ * Bilinear(u{i}, uʷ{i} * uʷ{j} * n{j}) + Bilinear(p, (uʷ{i} - u{i}) * n{i}) +  μ * Bilinear(u{i;j}, (uʷ{i} - u{i}) * n{j}) + τᵇ * ρ * Bilinear(u{i}, u{i} - uʷ{i})
    
    NS_boundary_OUTFLOW = ρ * Bilinear(u{i}, u{i} * u{j} * n{j}) + τᵖ * Bilinear(p, p)
    NS_boundary_FIX = Bilinear(p, - u{i} * n{i}) + μ * Bilinear(u{i;j}, - u{i} * n{j}) + τᵇ * ρ * Bilinear(u{i}, u{i})
end

@Def begin
    WF_domain = NS_domain_BASE + NS_domain_SUPG
    WF_boundary_inflow = NS_boundary_BASE + NS_boundary_INFLOW
    WF_boundary_outflow = NS_boundary_BASE + NS_boundary_OUTFLOW
    WF_boundary_fix = NS_boundary_BASE + NS_boundary_FIX
end

@time begin
    assign_WorkPiece_WeakForm(wp_ID, WF_domain; fem_domain = fem_domain)
    assign_Boundary_WeakForm(wp_ID, inflow_bg_ID, WF_boundary_inflow; fem_domain = fem_domain)
    assign_Boundary_WeakForm(wp_ID, outflow_bg_ID, WF_boundary_outflow; fem_domain = fem_domain)
    assign_Boundary_WeakForm(wp_ID, fixed_bg_ID, WF_boundary_fix; fem_domain = fem_domain)
    initialize_LocalAssembly(fem_domain.dim, fem_domain.workpieces; explicit_max_sd_order = 1)
end
#------------------------------
## Assemble system
#------------------------------
mesh_Classical([wp_ID]; shape = element_shape, itp_order = 2, itg_order = 6, fem_domain = fem_domain)

@time begin
    for wp in fem_domain.workpieces
        update_Mesh(fem_domain.dim, wp, wp.element_space)
    end
    assemble_Global_Variables(fem_domain = fem_domain)
    compile_Updater_GPU(domain_ID = 1, fem_domain = fem_domain)
end
#------------------------------
## Run!
#------------------------------
fem_domain.linear_solver = x -> solver_IDRs(x; Pl_func = precondition_CUDA_Jacobi, max_iter = 5000, max_pass = 10, s = 8)

fem_domain.globalfield.x .= 0.
fem_domain.globalfield.t = 0
dessemble_X(fem_domain.workpieces, fem_domain.globalfield)

@Takeout (controlpoints, facets, elements) FROM fem_domain.workpieces[1].mesh
cp_IDs = findall(controlpoints.is_occupied)
ys = controlpoints.x2[cp_IDs]
zs = controlpoints.x3[cp_IDs]
Um = 0.45
controlpoints.uʷ1[cp_IDs] .= (16 * Um / H ^ 4) .* (ys .* zs .* (H .- ys) .* (H .- zs)) 

tmax = 1
for i = 1:tmax
    dt = fem_domain.time_discretization.dt = 0.2 * Δx / Um
    fem_domain.globalfield.converge_tol = 1e-5

    controlpoints.τᵐ[cp_IDs] .= (9 * 16 * ν ^ 2 * dim * Δx ^ (-4)) ^ (-0.5)
    controlpoints.τᶜ[cp_IDs] .= (controlpoints.τᵐ[cp_IDs] .* (dim * Δx ^ (-2))) .^ (-1.)
    update_OneStep(fem_domain.time_discretization; max_iter = 6, fem_domain = fem_domain)
    dessemble_X(fem_domain.workpieces, fem_domain.globalfield)
end
#------------------------------
## Vtk
#------------------------------
wp = fem_domain.workpieces[1]
write_VTK(joinpath(@__DIR__, "3D_MetaFEM_Result.vtk"), wp)
#------------------------------
## native paraview line plot is not easy to unify format, so we 
#  re-plot the data (sampled in paraview and extracted as CSV) in Julia
#------------------------------
using CSV, DataFrames
using CairoMakie, Colors
COMSOL_y2 = CSV.read(joinpath(@__DIR__, "COMSOL_y2.csv"), DataFrame)
COMSOL_y3 = CSV.read(joinpath(@__DIR__, "COMSOL_y3.csv"), DataFrame)
MetaFEM_y2 = CSV.read(joinpath(@__DIR__, "MetaFEM_y2.csv"), DataFrame)
MetaFEM_y3 = CSV.read(joinpath(@__DIR__, "MetaFEM_y3.csv"), DataFrame)
##
getfield(COMSOL_y2, :colindex)
##
getfield(COMSOL_y2, :columns)
##
getfield(MetaFEM_y2, :colindex)
##
getfield(MetaFEM_y2, :columns)
##
fig1 = Figure(resolution = (1000, 600))
ax1 = fig1[1, 1] = Axis(fig1)
fontsize = 25

ax1.limits = (0., 2.5, -0.05, 0.6)
ax1.xticks = 0:0.2:2.5
ax1.yticks = 0:0.1:0.6
ax1.xlabel = "x(m)"
ax1.ylabel = "u(m/s)" 
ax1.xlabelsize = fontsize
ax1.ylabelsize = fontsize

plot1_COMSOL_y2 = scatter!(ax1, getfield(COMSOL_y2, :columns)[6], getfield(COMSOL_y2, :columns)[1], markersize = 10px, color = :deepskyblue2)
plot1_COMSOL_y3 = scatter!(ax1, getfield(COMSOL_y3, :columns)[6], getfield(COMSOL_y3, :columns)[1], markersize = 10px, color = :blue)
plot1_MetaFEM_y2 = scatter!(ax1, getfield(MetaFEM_y2, :columns)[6], getfield(MetaFEM_y2, :columns)[2], markersize = 10px, color = :coral2)
plot1_MetaFEM_y3 = scatter!(ax1, getfield(MetaFEM_y3, :columns)[6], getfield(MetaFEM_y3, :columns)[2], markersize = 10px, color = :red)

Legend(fig1, [plot1_COMSOL_y2, plot1_COMSOL_y3, plot1_MetaFEM_y2, plot1_MetaFEM_y3], bbox = (600, 900, 150, 300),
["COMSOL, y = 0.2", "COMSOL, y = 0.3", "MetaFEM, y = 0.2", "MetaFEM, y = 0.3"], labelsize = fontsize)
fig1
##
save(string(@__DIR__, "\\", "u_line.png"), fig1)
##
fig2 = Figure(resolution = (1000, 600))
ax2 = fig2[1, 1] = Axis(fig2)
fontsize = 25

ax2.limits = (0., 2.5, 0, 250)
ax2.xticks = 0:0.2:2.5
ax2.yticks = 0:50:250
ax2.xlabel = "x(m)"
ax2.ylabel = "p(Pa)" 
ax2.xlabelsize = fontsize
ax2.ylabelsize = fontsize

plot1_COMSOL_y2 = scatter!(ax2, getfield(COMSOL_y2, :columns)[6], getfield(COMSOL_y2, :columns)[4], markersize = 10px, color = :deepskyblue2)
plot1_COMSOL_y3 = scatter!(ax2, getfield(COMSOL_y3, :columns)[6], getfield(COMSOL_y3, :columns)[4], markersize = 10px, color = :blue)
plot1_MetaFEM_y2 = scatter!(ax2, getfield(MetaFEM_y2, :columns)[6], getfield(MetaFEM_y2, :columns)[1], markersize = 10px, color = :coral2)
plot1_MetaFEM_y3 = scatter!(ax2, getfield(MetaFEM_y3, :columns)[6], getfield(MetaFEM_y3, :columns)[1], markersize = 10px, color = :red)

Legend(fig2, [plot1_COMSOL_y2, plot1_COMSOL_y3, plot1_MetaFEM_y2, plot1_MetaFEM_y3], bbox = (600, 900, 350, 500),
["COMSOL, y = 0.2", "COMSOL, y = 0.3", "MetaFEM, y = 0.2", "MetaFEM, y = 0.3"], labelsize = fontsize)
fig2
##
save(string(@__DIR__, "\\", "p_line.png"), fig2)