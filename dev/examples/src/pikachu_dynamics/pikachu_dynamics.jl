# # Thermal conduction in a pikachu, dynamic version
# ![global](global.mp4)
# ![slice](slice.mp4)
#
# The previous example can be regarded as the final equilibrium state of a dynamic heating problem. To do a dynamic simulation, 
# surely we can use MetaFEM as a pure spatial discretization package and write the customized timestep marching by ourselves, 
# as in many other FEM packages. However, MetaFEM also provides an integrated generalized-alpha temporal discretization with only a little extra code, 
# as in the following script.
# The source is also in [the same folder](https://github.com/jxx2/MetaFEM.jl/tree/main/examples/thermal_conduction) as the last one.
#
# ## Geometry
# The geometry in the same with the last case:
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
# ## Physics
# In the mathematical formulation of a dynamic problem, a bilinear term with the time derivative is simply added:
# ```math
# -C(T, T_{,t})
# ```
# while the full formulation is:
# ```math
# \text{variable}\quad T,\qquad\text{parameters}\quad C,k,h,T_{env},s
# ```
# ```math
# -C(T, T_{,t})-k(T_{,i},T_{,i})+(T,s)=0,\qquad in\quad\Omega
# ```
# ```math
# h(T,T_{env}-T)=0,\qquad on\quad\partial\Omega
# ```
# where the new $C$ stands for the volumetric heat capacity. 
#
# Correspondingly, in the code there is simply another bilinear term with the time derivative, i.e., `-C*Bilinear(T, T{;t})` in the domain physics, i.e., `heat_dissipation`:
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
# ## Assembly
# No change.
initialize_LocalAssembly!(fem_domain; explicit_max_sd_order = 1)
mesh_Classical([wp_ID]; shape = element_shape, itp_type = :Serendipity, itp_order = 2, itg_order = 5, fem_domain = fem_domain)
compile_Updater_GPU(domain_ID = 1, fem_domain = fem_domain)

for wp in fem_domain.workpieces
    update_Mesh(fem_domain.dim, wp, wp.element_space)
end
assemble_Global_Variables!(fem_domain = fem_domain)

# ## Run & Output
# Mostly the same as before, `fem_domain.globalfield.dt` is needed to be set. Note, the default generalized alpha (bi = 0.5, ci = 1) is unconditionally stable.
fem_domain.linear_solver = x -> iterative_Solve!(x; Sv_func! = idrs!, maxiter = 2000, max_pass = 10, s = 8)
fem_domain.globalfield.converge_tol = 1e-6
fem_domain.globalfield.dt = 1

cpts = fem_domain.workpieces[1].mesh.controlpoints
cp_IDs = findall(cpts.is_occupied)
cpts.T[cp_IDs] .= Tₑₙᵥ 
cpts.s[cp_IDs] .= 1600. 
# Also we need to synchronize the initial temperature distribution by `assemble_X!`:
assemble_X!(fem_domain.workpieces, fem_domain.globalfield)
# Finally we simply only run 10 timestep and save each timestep in vtk because writing in VTK is relatively slow (.mp4 is 100 steps).
for i = 1:10
    update_OneStep!(fem_domain.time_discretization; fem_domain = fem_domain)
    dessemble_X!(fem_domain.workpieces, fem_domain.globalfield)

    wp = fem_domain.workpieces[1]
    write_VTK(joinpath(@__DIR__, "history", "3D_MetaFEM_Result_$i.vtk"), wp; scale = 100)
end