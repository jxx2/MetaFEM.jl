abstract type FEM_Geometry{ArrayType} end #2D / 3D, Mesh / bdy
abstract type FEM_Object{ArrayType} end

abstract type FEM_WP_Mesh{ArrayType} end
abstract type FEM_Tool_Mesh{ArrayType} end
abstract type FEM_Spatial_Discretization{ArrayType} end
abstract type FEM_Temporal_Discretization end

#physics
#--------------------------
mutable struct FEM_Physics{ArrayType}
    extra_var::Vector{Symbol}
    boundarys::Vector #each boundary group is a set of ref_edge_IDs
    # boundarys::Vector{ArrayType} #each boundary group is a set of ref_edge_IDs
    boundary_weakform_pairs::Dict{FEM_Int, Symbolic_WeakForm} #bg_ID, wf
    domain_weakform::Symbolic_WeakForm
    FEM_Physics(::Type{ArrayType}) where {ArrayType} = new{ArrayType}(Symbol[], AbstractArray{FEM_Int, 1}[], Dict{FEM_Int, Symbolic_WeakForm}())
end

#assembly
#--------------------------
InnervarInfo = Tuple{Symbol, FEM_Int, Tuple, FEM_Int} #total symbol, time_discretization derivative order, spatial derivative index, basic local position
ExtervarInfo = Tuple{Symbol, Symbol, Symbol, Tuple, Tuple} #total symbol, local_symbol, type symbol, spatial derivative index, c_ids

struct AssembleTerm
    ex::Union{Number, Symbol, Expr}
    innervar_infos::Vector{InnervarInfo}
    extervar_infos::Vector{ExtervarInfo}
end
struct AssembleBilinear
    dual_info::InnervarInfo #time_discretization derivative order, spatial derivative index, basic local position (start with 0)
    base_term::AssembleTerm
    derivative_info::InnervarInfo #time_discretization derivative order, spatial derivative index, basic local position (start with 0)
end
struct AssembleWeakform
    residues::Vector{AssembleBilinear}
    gradients::Vector{AssembleBilinear}

    residueID_by_pos::Dict{FEM_Int, Vector{FEM_Int}} #note the key is basic local position
    gradID_by_pos::Dict{Tuple{FEM_Int, FEM_Int}, Vector{FEM_Int}} #note the key is two basic local positions
end

mutable struct FEM_LocalAssembly
    basic_vars::Vector{Symbol} #basic symbol for x allocation
    local_innervar_infos::Vector{Tuple{Symbol, FEM_Int, FEM_Int}} #(local symbol, basic_pos, time_order) for local var allocation
    local_extervars::Vector{Symbol} #local symbol for local var allocation

    assembled_boundary_weakform_pairs::Dict{FEM_Int, AssembleWeakform} #bg_ID, wf
    assembled_weakform::AssembleWeakform

    sparse_entry_ID::Integer
    sparse_unitsize::Integer
    sparse_mapping::Dict{Tuple{FEM_Int, FEM_Int}, FEM_Int}
end

#object
#--------------------------
"""
    WorkPiece

A `WorkPiece` is a meshed part assigned with some known physics. The attributes are:

* `ref_geometry`, the first order mesh to describe the FEM_Geometry.
* `physics`, the raw PDE weakforms.
* `local_assembly`, the re-organized PDE weakforms with sorted/indexed variables.
* `max_sd_order`, an explicit limit of the maximum spatial derivative order to save memory.
* `element_space`, the information about the spatial discritization, i.e., interpolation and intergration.
* `mesh`, the mesh regenerated according to the `element_space` and actually used in simulation.

To add a `Workpiece` with the geometry `ref_geometry` to the [`FEM_Domain`](@ref) `fem_domain`, the exposed API is:

    add_WorkPiece!(ref_geometry; fem_domain::FEM_Domain)

which returns the `WorkPiece` ID in the `fem_domain`.`workpieces`.
"""
mutable struct WorkPiece{ArrayType} <: FEM_Object{ArrayType}
    ref_geometry::FEM_Geometry{ArrayType}

    physics::FEM_Physics{ArrayType}
    local_assembly::FEM_LocalAssembly

    max_sd_order::Integer
    element_space::FEM_Spatial_Discretization
    mesh::FEM_WP_Mesh{ArrayType}

    function WorkPiece(ref_geometry::FEM_Geometry{ArrayType}) where {ArrayType}
        new{ArrayType}(ref_geometry, FEM_Physics(ArrayType))
    end
end

mutable struct Tool{ArrayType} <: FEM_Object{ArrayType}
    ref_geometry::FEM_Geometry{ArrayType}
    boundarys::Vector #each boundary group is a set of ref_edge_IDs
    mesh::FEM_Tool_Mesh{ArrayType}

    function Tool(ref_geometry::FEM_Geometry{ArrayType}) where {ArrayType}
        new{ArrayType}(ref_geometry, AbstractArray{FEM_Int, 1}[])
    end
end

mutable struct FEM_Contact
    master_id::Integer
    master_type::Symbol
    slave_id::Integer
    bg_pairs::Vector{Tuple{FEM_Int, FEM_Int}}
end

#Time domain
#-------------------
mutable struct GlobalField{ArrayType} #global infos & FEM data, should be separated later
    max_time_level::Integer
    basicfield_size::Integer

    converge_tol::FEM_Float

    t::FEM_Float
    dt::FEM_Float
    global_vars::Dict{Symbol, FEM_Float}

    x::ArrayType #x0 u0 a0
    dx::ArrayType
    x_star::ArrayType

    residue::ArrayType

    K_I::ArrayType
    K_J_ptr::ArrayType
    K_J::ArrayType
    K_val_ids::AbstractArray
    K_linear::ArrayType
    K_total::ArrayType
    GlobalField(::Type{ArrayType}) where {ArrayType} = new{ArrayType}(0, 0, 0., 0., 0., Dict{Symbol, FEM_Float}())
end

"""
    FEM_Domain

A `FEM_Domain` contains everything needed to assemble a linear system `Kx=d` in FEM. The attributes are:

* `dim`, the dimension, 2 or 3.
* `workpieces`, the array of all the `WorkPiece`s in this domain, which will be finally solved in fully coupling.
* `tools`, reserved for the external geometry `Tool`, e.g., for contact. Not implemented.
* `time_discretization`, the temporal discritization scheme. Currently only generalized-α method is implemented.
* `globalfield`, the container for sparse `K`, dense `x` and `d` in `Kx=d`.
* `K_linear_func` the generated function to update the linear part of `K`.
* `K_nonlinear_func` the generated function to update the nonlinear part of `K` and the residue `d`.
* `linear_solver`, the applied linear solver.

To add a `FEM_Domain` of dimension `dim`, the exposed API is:

    FEM_Domain(; dim::Integer)

which returns the new `FEM_Domain`.
"""
mutable struct FEM_Domain{ArrayType} #workgroup
    dim::Integer
    workpieces::Vector{WorkPiece{ArrayType}}
    tools::Vector{Tool{ArrayType}}

    time_discretization::FEM_Temporal_Discretization
    globalfield::GlobalField{ArrayType}

    K_linear_func::Function
    K_nonlinear_func::Function
    linear_solver::Function
    
    FEM_Domain(::Type{ArrayType} = DEFAULT_ARRAYINFO._type; dim::Integer) where {ArrayType} = new{ArrayType}(dim, WorkPiece{ArrayType}[], Tool{ArrayType}[], GeneralAlpha(), GlobalField(ArrayType)) # need to rewrite
end

function add_WorkPiece!(ref_geometry::FEM_Geometry{ArrayType}; fem_domain::FEM_Domain{ArrayType}) where {ArrayType}
    push!(fem_domain.workpieces, WorkPiece(ref_geometry))
    wp_ID = length(fem_domain.workpieces)
    println("Workpiece $wp_ID added!")
    return wp_ID
end

"""
    add_Boundary!(ID::Integer, bdy_ref_edge_IDs::CuVector; fem_domain::FEM_Domain = fem_domain, target::Symbol = :WorkPiece)

In a 2D/3D `FEM_Domain` `fem_domain`, mark the segment/face IDs of the `WorkPiece` `fem_domain`.`workpieces`[`wp_ID`] as a boundary, to assign physics later.
Multiple boundaries are independent from each other and can share the same segment/face IDs. If target = `:Tool`, the boundary is of `fem_domain`.`tools`[`wp_ID`], not implemented.

The function returns the boundary group ID `bg_ID`.
"""
function add_Boundary!(wp_ID::Integer, bdy_ref_edge_IDs::AbstractVector; fem_domain::FEM_Domain{ArrayType} = fem_domain, target::Symbol = :WorkPiece) where {ArrayType}
    if target == :WorkPiece
        boundarys = fem_domain.workpieces[wp_ID].physics.boundarys
    elseif target == :Tool
        boundarys = fem_domain.tools[wp_ID].boundarys
    end
    push!(boundarys, FEM_convert(ArrayType, bdy_ref_edge_IDs))
    bg_ID = length(boundarys)
    println("Boundary $bg_ID added to $target $wp_ID !")
    return bg_ID
end

"""
    assign_WorkPiece_WeakForm!(wp_ID::Integer, this_term::SymbolicTerm; fem_domain::FEM_Domain)
    assign_Boundary_WeakForm!(wp_ID::Integer, bg_ID::Integer, this_term::SymbolicTerm; fem_domain::FEM_Domain)

The functions assign `this_term`, which is either a bilinear term Bilinear(⋅, ⋅), or a sum of the bilinear terms, to the target `WorkPiece` or boundary.
"""
function assign_Boundary_WeakForm!(wp_ID::Integer, bg_ID::Integer, this_term::SymbolicTerm; fem_domain::FEM_Domain)
    fem_domain.workpieces[wp_ID].physics.boundary_weakform_pairs[bg_ID] = parse_WeakForm(this_term, fem_domain.dim)
    if fem_domain.workpieces[wp_ID].physics.boundary_weakform_pairs[bg_ID] == 0 
        delete!(boundary_weakform_pairs, bg_ID)
        println("The weakform for Boundary $bg_ID is removed becase it is 0.")
    end
end
assign_Boundary_WeakForm!(wp_ID::Integer, bg_ID::Integer, this_term::FEM_Float; fem_domain::FEM_Domain) = this_term == 0 ? this_term : error()
function assign_WorkPiece_WeakForm!(wp_ID::Integer, this_term::SymbolicTerm; fem_domain::FEM_Domain)
    fem_domain.workpieces[wp_ID].physics.domain_weakform = parse_WeakForm(this_term, fem_domain.dim)
end
