abstract type FEM_Geometry end #2D / 3D, Mesh / bdy
abstract type FEM_Object end

abstract type FEM_WP_Mesh end
abstract type FEM_Tool_Mesh end
abstract type FEM_Spatial_Discretization end
abstract type FEM_Temporal_Discretization end

#physics
#--------------------------
mutable struct FEM_Physics
    extra_var::Vector{Symbol}
    boundarys::Vector{CuVector{FEM_Int}} #each boundary group is a set of ref_edge_IDs
    boundary_weakform_pairs::Dict{FEM_Int, Symbolic_WeakForm} #bg_ID, wf
    domain_weakform::Symbolic_WeakForm
    FEM_Physics() = new(Symbol[], CuVector{FEM_Int}[], Dict{FEM_Int, Symbolic_WeakForm}())
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
    local_extervars::Vector{Symbol} #local symbol  for local var allocation

    assembled_boundary_weakform_pairs::Dict{FEM_Int, AssembleWeakform} #bg_ID, wf
    assembled_weakform::AssembleWeakform

    sparse_entry_ID::Integer
    sparse_unitsize::Integer
    sparse_mapping::Dict{Tuple{FEM_Int, FEM_Int}, FEM_Int}
end

#object
#--------------------------
mutable struct WorkPiece <: FEM_Object
    self_mergeable::Bool
    ref_geometry::FEM_Geometry

    physics::FEM_Physics
    local_assembly::FEM_LocalAssembly

    max_sd_order::Integer
    element_space::FEM_Spatial_Discretization
    mesh::FEM_WP_Mesh

    function WorkPiece(ref_geometry; self_mergeable::Bool = true)
        new(self_mergeable, ref_geometry, FEM_Physics())
    end
end

mutable struct Tool <: FEM_Object
    ref_geometry::FEM_Geometry
    boundarys::Vector{CuVector{FEM_Int}} #each boundary group is a set of ref_edge_IDs
    mesh::FEM_Tool_Mesh

    function Tool(ref_geometry)
        new(ref_geometry, CuVector{FEM_Int}[])
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
mutable struct GlobalField #global infos & FEM data, should be separated later
    max_time_level::Integer
    basicfield_size::Integer

    converge_tol::FEM_Float

    t::FEM_Float
    dt::FEM_Float
    global_vars::Dict{Symbol, FEM_Float}

    x::CuVector{FEM_Float} #x0 u0 a0
    dx::CuVector{FEM_Float}
    x_star::CuVector{FEM_Float}

    residue::CuVector{FEM_Float}

    K_I::CuVector{FEM_Int}
    K_J::CuVector{FEM_Int}
    K_val_ids::CuVector{FEM_Int}
    
    K_linear::CuVector{FEM_Float}
    K_total::CuVector{FEM_Float}
    GlobalField() = new(0, 0, 0., 0., 0.)
end

mutable struct FEM_Domain #workgroup
    dim::Integer
    workpieces::Vector{WorkPiece}
    tools::Vector{Tool}

    time_discretization::FEM_Temporal_Discretization
    globalfield::GlobalField

    K_linear_func::Function
    K_nonlinear_func::Function
    linear_solver::Function
    
    FEM_Domain(; dim::Integer) = new(dim, WorkPiece[], Tool[], GeneralAlpha(), GlobalField())
end

function add_WorkPiece(ref_geometry; fem_domain::FEM_Domain)
    push!(fem_domain.workpieces, WorkPiece(ref_geometry))
    wp_ID = fem_domain.workpieces |> length |> FEM_Int
    println("Workpiece ", wp_ID, " added!")
    return wp_ID
end

function add_Boundary(ID::Integer, bdy_ref_edge_IDs::CuVector; fem_domain::FEM_Domain = fem_domain, target::Symbol = :WP)
    if target == :WP
        boundarys = fem_domain.workpieces[ID].physics.boundarys
        msg = string("Boundary ", length(boundarys) + 1, " added to Workpiece ", ID, " !")
    elseif target == :Tool
        boundarys = fem_domain.tools[ID].boundarys
        msg = string("Boundary ", length(boundarys) + 1, " added to Tool ", ID, " !")
    end
    push!(boundarys, bdy_ref_edge_IDs)
    bg_ID = boundarys |> length |> FEM_Int
    println(msg)
    return bg_ID
end

function assign_Boundary_WeakForm(wp_ID::Integer, bg_ID::Integer, this_term::SymbolicTerm; fem_domain::FEM_Domain)
    this_sym_weakform = parse_WeakForm(this_term, fem_domain.dim)
    this_sym_weakform == 0 && return
    fem_domain.workpieces[wp_ID].physics.boundary_weakform_pairs[bg_ID] = this_sym_weakform
end
assign_Boundary_WeakForm(wp_ID::FEM_Int, bg_ID::FEM_Int, this_term::FEM_Float; fem_domain::FEM_Domain) = this_term == 0 ? this_term : error()
function assign_WorkPiece_WeakForm(wp_ID::FEM_Int, this_term::SymbolicTerm; fem_domain::FEM_Domain)
    this_sym_weakform = parse_WeakForm(this_term, fem_domain.dim)
    fem_domain.workpieces[wp_ID].physics.domain_weakform = this_sym_weakform
end
