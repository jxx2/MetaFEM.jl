# # Incompressible flow pass a cylinder
# In this example we simulate a flow case like this
# ![flow](cylinderflow.png)
# First, we load the package and declare the domain:
using MetaFEM
initialize_Definitions!()
dim = 3
fem_domain = FEM_Domain(; dim = dim)
# ## Geometry
# Load the mesh "3D\_COMSOL\_Mesh.mphtxt" which can be found in [here](https://github.com/jxx2/MetaFEM.jl/tree/main/examples/cylinder_flow) and should be put under the same directory of the script.
L = 2.5
H = 0.41
element_shape = :SIMPLEX
src_fname = joinpath(@__DIR__, "3D_COMSOL_Mesh.mphtxt")
vert, connections = read_Mesh(src_fname)
ref_mesh = construct_TotalMesh(vert, connections)
# Define boundary:
@Takeout (vertices, faces) FROM ref_mesh
facet_IDs = get_BoundaryMesh(ref_mesh)
vIDs = faces.vertex_IDs[:, facet_IDs] 
x1_mean = vec(sum(vertices.x1[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x2_mean = vec(sum(vertices.x2[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)
x3_mean = vec(sum(vertices.x3[vIDs], dims = 1)) ./ size(faces.vertex_IDs, 1)

err_scale = 0.01

is_left = (x1_mean .< err_scale) .& (x1_mean .> (.- err_scale))
is_right = (x1_mean .< (L .+ err_scale)) .& (x1_mean .> (L .- err_scale))

wp_ID = add_WorkPiece!(ref_mesh; fem_domain = fem_domain)
fixed_bg_ID = add_Boundary!(wp_ID, facet_IDs[.~(is_left .| is_right)]; fem_domain = fem_domain)
inflow_bg_ID = add_Boundary!(wp_ID, facet_IDs[is_left]; fem_domain = fem_domain)
outflow_bg_ID = add_Boundary!(wp_ID, facet_IDs[is_right]; fem_domain = fem_domain)
# ## Physics
# The steady state Navier-Stokes (NS) equation with Streamline-Upwind/Peterove-Galerkin (SUPG) stabilization can be formed as follows:
# ```math
# \text{variable}\quad u_i,p,\qquad\text{parameters}\quad\rho,\mu,u^W_i,\tau^b,\tau^c,\tau^m
# ```
# ```math
# Rc\coloneqq u_{k,k},\qquad {Rm}_i\coloneqq\rho u_ku_{i,k}+p_{,i}-\mu u_{i,kk}
# ```
# ```math
# \overbrace{-\rho(u_{i,j},u_iu_j)-(u_{i,i},p)+(p,u_{i,i})+\mu(u_{i,j},u_{i,j})}^{NS}+\overbrace{\tau^m\rho(u_{i,j},{Rm}_iu_j)+\tau^m(p_{,i},{Rm}_i)+\tau^c(u_{i,i},Rc)}^{SUPG} =0,\qquad in\quad\Omega
# ```
# ```math
# (u_i,pn_i)-\mu(u_i,u_{i,j}n_j)=0,\qquad on\quad\partial\Omega
# ```
# ```math
# \rho(u_i,u^w_iu^w_jn_j)+(p,(u^w_i-u_i)n_i)+\mu(u_{i,j},(u^w_i-u_i)n_j)+\tau^b\rho(u_i,u_i-u^w_i)=0,\qquad on\quad(\partial\Omega)_{inflow}
# ```
# ```math
# \rho(u_i,u_iu_jn_j)=0,\qquad on\quad(\partial\Omega)_{outflow}
# ```
# ```math
# (p,-u_in_i)+\mu(u_{i,j},-u_in_j)+\tau^b\rho(u_i,u_i)=0,\qquad on\quad(\partial\Omega)_{fix}
# ```
# where $u_i,p,\rho,\mu$ are the velocity, pressure, density and the dynamic viscosity respectively.
# The code is:
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

assign_WorkPiece_WeakForm!(wp_ID, WF_domain; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, inflow_bg_ID, WF_boundary_inflow; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, outflow_bg_ID, WF_boundary_outflow; fem_domain = fem_domain)
assign_Boundary_WeakForm!(wp_ID, fixed_bg_ID, WF_boundary_fix; fem_domain = fem_domain)
# ## Assemble
initialize_LocalAssembly!(fem_domain; explicit_max_sd_order = 1)
mesh_Classical([wp_ID]; shape = element_shape, itp_order = 2, itg_order = 6, fem_domain = fem_domain)
compile_Updater_GPU(domain_ID = 1, fem_domain = fem_domain)

# ## Run
for wp in fem_domain.workpieces
    update_Mesh(fem_domain.dim, wp, wp.element_space)
end
assemble_Global_Variables!(fem_domain = fem_domain)
fem_domain.linear_solver = x -> iterative_Solve!(x; Sv_func! = idrs!, Pl_func = Pl_Jacobi, maxiter = 2000, max_pass = 10, s = 8)
fem_domain.globalfield.converge_tol = 1e-6

fem_domain.globalfield.x .= 0.
fem_domain.globalfield.t = 0
dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)

@Takeout (controlpoints, facets, elements) FROM fem_domain.workpieces[1].mesh
cp_IDs = findall(controlpoints.is_occupied)
ys = controlpoints.x2[cp_IDs]
zs = controlpoints.x3[cp_IDs]
Um = 0.45
controlpoints.uʷ1[cp_IDs] .= (16 * Um / H ^ 4) .* (ys .* zs .* (H .- ys) .* (H .- zs)) 

tmax = 1
for i = 1:tmax
    dt = fem_domain.globalfield.dt = 0.2 * Δx / Um

    controlpoints.τᵐ[cp_IDs] .= (9 * 16 * ν ^ 2 * dim * Δx ^ (-4)) ^ (-0.5)
    controlpoints.τᶜ[cp_IDs] .= (controlpoints.τᵐ[cp_IDs] .* (dim * Δx ^ (-2))) .^ (-1.)
    update_OneStep!(fem_domain.time_discretization; max_iter = 6, fem_domain = fem_domain)
    dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)
end
# ## Save to VTK
wp = fem_domain.workpieces[1]
write_VTK(joinpath(@__DIR__, "3D_MetaFEM_Result.vtk"), wp)