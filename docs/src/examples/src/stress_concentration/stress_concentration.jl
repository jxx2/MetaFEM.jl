# # Stress concentration of a hole
#
# In an elasticity class, we may learn that the stress concentration factor for a cylinderical hole is 3 while for a spherical hole is 2, as plotted:
# ![sc](sc.png)
#
# In this example do the 3D case, while both 2D/3D source with mesh/data/visualization can be found [here](https://github.com/jxx2/MetaFEM.jl/tree/main/examples/stress_concentration).
#
# First, we load the package and declare the domain:
using MetaFEM
fem_domain = FEM_Domain(dim = 3)
# ## Geometry
# The mesh is generated with Abaqus. Note, we have only the first order mesh loader and the higher order mesh is regenerated inside MetaFEM.
element_shape = :CUBE 
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

# ## Physics
# Only 1/8 domain is simulated to use symmetry.
E = 210e9 
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
# ## Assembly & Run
# Note, the code generated in "compile\_Updater\_GPU" is only used by "update\_OneStep" so its position with the mesh update doesn't matter.
initialize_LocalAssembly!(fem_domain.dim, fem_domain.workpieces)
mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)
for wp in fem_domain.workpieces
    update_Mesh(fem_domain.dim, wp, wp.element_space)
end
assemble_Global_Variables!(; fem_domain = fem_domain)
compile_Updater_GPU(; domain_ID = 1, fem_domain = fem_domain)

fem_domain.linear_solver = x -> iterative_Solve!(x; Sv_func! = idrs!, maxiter = 2000, max_pass = 20, s = 20)
fem_domain.globalfield.converge_tol = 1e-8
σ_external = 1

cp = fem_domain.workpieces[1].mesh.controlpoints
cp.σˡ2 .= σ_external

update_OneStep!(fem_domain.time_discretization; fem_domain = fem_domain)
dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)
# ## VTK
wp = fem_domain.workpieces[1]
write_VTK(string(@__DIR__, "\\", "3D_MetaFEM.vtk"), wp)