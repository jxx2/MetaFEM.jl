F_S_V_SIMPLEX = [[1, 2], [2, 3], [3, 1]] #f = face = face, s = segment to distinguish
F_S_V_CUBE = [[1, 2], [2, 3], [3, 4], [4, 1]]

B_S_V_SIMPLEX = [[1, 2], [2, 3], [3, 1], [1, 4], [2, 4], [3, 4]]
B_S_V_CUBE = [[1, 2], [2, 3], [3, 4], [4, 1], [1, 5], [2, 6], [3, 7], [4, 8], [5, 6], [6, 7], [7, 8], [8, 5]]

B_F_S_SIMPLEX = [[1, 2, 3], [1, 5, 4], [2, 6, 5], [3, 4, 6]]
B_F_S_CUBE = [[1, 2, 3, 4], [1, 6, 9, 5], [2, 7, 10, 6], [3, 8, 11, 7], [4, 8, 12, 5], [9, 10, 11, 12]]

"""
    construct_TotalMesh(coors::CuArray, connections::CuArray)

The function reads (vert, connections) and declares the first order mesh, `FEM_Geometry`.

There are 4 concepts in a `FEM_Geometry`:
* Each vertex in `vertices` stores the vertex coordinates.
* Each segment in `segments` stores the vertex IDs connected to this segment.
* Each face in `faces` stores both the vertex IDs and the segment IDs connected to this face.
* Each block in `blocks` stores the vertex IDs, the segment IDs and the face IDs connected to this block.
If applicable, the IDs are in order, e.g., segment IDs in a face are clockwise/counter-clockwise (since we can't define an outward normal for a bare face).

The function returns either a `Geo_TotalMesh2D` with attributes (`vertices`, `segments`, `faces`) or a 
`Geo_TotalMesh3D` with attributes (`vertices`, `segments`, `faces`, `blocks`).
"""
construct_TotalMesh(coors::AbstractArray, connections::AbstractArray) = construct_TotalMesh(cu(coors), cu(connections))
function construct_TotalMesh(coors::CuArray, connections::CuArray)
    dim, _ = size(coors)
    if dim == 2 
        ref_geometry = construct_TotalMesh_2D(coors, connections)
    elseif dim == 3 
        ref_geometry = construct_TotalMesh_3D(coors, connections)
    else
        error(dim, "Undefined dimension")
    end
end

get_Next_ID(id, size) = id == size ? 1 : (id + 1)
get_Prev_ID(id, size) = id == 1 ? size : (id - 1)
function construct_TotalMesh_2D(coors::CuArray, connection::CuArray)
    vertex_per_block, block_number = size(connection)
    if vertex_per_block == 3
        mesh_type = :SIMPLEX
        F_S_V_POS = F_S_V_SIMPLEX
    elseif vertex_per_block == 4
        mesh_type = :CUBE
        F_S_V_POS = F_S_V_CUBE
    else
        error("dim", dim, "vertices_per_block", vertices_per_block, "Undefined mesh type")
    end

    ref_geometry = Geo_TotalMesh2D(mesh_type)
    @Takeout (vertices, segments, faces) FROM ref_geometry
    vIDs = allocate_by_length!(vertices, size(coors, 2)) #Note here vIDs start exactly from 1, e.g., 1, 2,..., so connections don't need modification
    vertices.x1[vIDs] .= coors[1, :]
    vertices.x2[vIDs] .= coors[2, :]

    fIDs = allocate_by_length!(faces, block_number)
    faces.vertex_IDs[:, fIDs] .= connection   

    seg_dict = dumb_GPUDict_Init(FEM_Int(0))
    for (f_s_pos, s_v_pos) in enumerate(F_S_V_POS)
        s_vIDs = connection[s_v_pos, :]
        rotate_size = length(s_v_pos)

        max_vIDs, cart_pos = findmax(s_vIDs, dims = 1) .|> vec
        max_pos = getindex.(cart_pos, 1)
        last_dim_ids = getindex.(cart_pos, 2)

        next_pos = get_Next_ID.(max_pos, rotate_size)
        next_vIDs = s_vIDs[CartesianIndex.(next_pos, last_dim_ids)]
        dict_keys = I4I30I30_To_UI64.(0, max_vIDs, next_vIDs)
        
        dict_slots = GPUDict_SetID(seg_dict, dict_keys)
        total_dict_IDs = get_Total_IDs(seg_dict)
        not_allocated = seg_dict.vals[total_dict_IDs] .== 0

        new_sIDs = allocate_by_length!(segments, sum(not_allocated))
        seg_dict.vals[total_dict_IDs[not_allocated]] .= new_sIDs

        local_sIDs = seg_dict.vals[dict_slots]
        faces.segment_IDs[f_s_pos, fIDs] .= local_sIDs

        segments.vertex_IDs[1, local_sIDs] .= max_vIDs
        segments.vertex_IDs[2, local_sIDs] .= next_vIDs
    end
    println("2D geometry constructed with $(length(vIDs)) vertices and $(length(fIDs)) faces, taking $(report_memory(ref_geometry))")
    return ref_geometry
end

function construct_TotalMesh_3D(coors::CuArray, connection::CuArray)
    vertex_per_block, block_number = size(connection)
    if vertex_per_block == 4
        mesh_type = :SIMPLEX
        vertex_per_face = 3
        B_S_V_POS = B_S_V_SIMPLEX
        B_F_S_POS = B_F_S_SIMPLEX
    elseif vertex_per_block == 8
        mesh_type = :CUBE
        vertex_per_face = 4
        B_S_V_POS = B_S_V_CUBE
        B_F_S_POS = B_F_S_CUBE
    else
        error("dim", dim, "vertices_per_block", vertices_per_block, "Undefined mesh type")
    end

    ref_geometry = Geo_TotalMesh3D(mesh_type)
    @Takeout (vertices, segments, faces, blocks) FROM ref_geometry
    vIDs = allocate_by_length!(vertices, size(coors, 2)) #Note here vIDs start exactly from 1, e.g., 1, 2,..., so connections don't need modification
    vertices.x1[vIDs] .= coors[1, :]
    vertices.x2[vIDs] .= coors[2, :]
    vertices.x3[vIDs] .= coors[3, :]

    bIDs = allocate_by_length!(blocks, block_number)
    blocks.vertex_IDs[:, bIDs] .= connection   

    seg_dict = dumb_GPUDict_Init(FEM_Int(0))
    for (b_s_pos, s_v_pos) in enumerate(B_S_V_POS)
        s_vIDs = connection[s_v_pos, :]
        rotate_size = length(s_v_pos)

        max_vIDs, cart_pos = findmax(s_vIDs, dims = 1) .|> vec
        max_pos = getindex.(cart_pos, 1)
        last_dim_ids = getindex.(cart_pos, 2)

        next_pos = get_Next_ID.(max_pos, rotate_size)

        next_vIDs = s_vIDs[CartesianIndex.(next_pos, last_dim_ids)]
        dict_keys = I4I30I30_To_UI64.(0, max_vIDs, next_vIDs)

        dict_slots = GPUDict_SetID(seg_dict, dict_keys)
        total_dict_IDs = get_Total_IDs(seg_dict)
        not_allocated = seg_dict.vals[total_dict_IDs] .== 0

        new_sIDs = allocate_by_length!(segments, sum(not_allocated))
        seg_dict.vals[total_dict_IDs[not_allocated]] .= new_sIDs

        local_sIDs = seg_dict.vals[dict_slots]
        blocks.segment_IDs[b_s_pos, bIDs] .= local_sIDs

        segments.vertex_IDs[1, local_sIDs] .= max_vIDs
        segments.vertex_IDs[2, local_sIDs] .= next_vIDs
    end

    fac_dict = dumb_GPUDict_Init(FEM_Int(0))
    for (b_f_pos, f_s_pos) in enumerate(B_F_S_POS)
        f_sIDs = blocks.segment_IDs[f_s_pos, bIDs]
        rotate_size = length(f_s_pos)

        max_sIDs, cart_pos = findmax(f_sIDs, dims = 1) .|> vec
        max_pos = getindex.(cart_pos, 1)
        last_dim_ids = getindex.(cart_pos, 2)

        prev_pos = get_Prev_ID.(max_pos, rotate_size)
        next_pos = get_Next_ID.(max_pos, rotate_size)
        is_forward = f_sIDs[CartesianIndex.(next_pos, last_dim_ids)] .>= f_sIDs[CartesianIndex.(prev_pos, last_dim_ids)]
        next_pos[.~ is_forward] .= prev_pos[.~ is_forward]

        next_sIDs = f_sIDs[CartesianIndex.(next_pos, last_dim_ids)]
        dict_keys = I4I30I30_To_UI64.(0, max_sIDs, next_sIDs)
        
        dict_slots = GPUDict_SetID(fac_dict, dict_keys)
        total_dict_IDs = get_Total_IDs(fac_dict)
        not_allocated = fac_dict.vals[total_dict_IDs] .== 0

        new_fIDs = allocate_by_length!(faces, sum(not_allocated))
        fac_dict.vals[total_dict_IDs[not_allocated]] .= new_fIDs

        local_fIDs = fac_dict.vals[dict_slots]
        blocks.face_IDs[b_f_pos, bIDs] .= local_fIDs

        faces.segment_IDs[1, local_fIDs] .= max_sIDs
        last_sIDs = max_sIDs
        for i = 1:1:vertex_per_face
            is_first_vID = (segments.vertex_IDs[1, last_sIDs] .== segments.vertex_IDs[1, next_sIDs]) .| 
                           (segments.vertex_IDs[1, last_sIDs] .== segments.vertex_IDs[2, next_sIDs])
            
            faces.vertex_IDs[i, local_fIDs[is_first_vID]] .= segments.vertex_IDs[1, last_sIDs[is_first_vID]]
            faces.vertex_IDs[i, local_fIDs[.~ is_first_vID]] .= segments.vertex_IDs[2, last_sIDs[.~ is_first_vID]]

            (i == vertex_per_face) && break
            faces.segment_IDs[i + 1, local_fIDs] .= next_sIDs

            last_sIDs = next_sIDs
            last_pos = next_pos

            prev_pos = get_Prev_ID.(last_pos, rotate_size)
            next_pos = get_Next_ID.(last_pos, rotate_size)
            next_pos[.~ is_forward] .= prev_pos[.~ is_forward]
            next_sIDs = f_sIDs[CartesianIndex.(next_pos, last_dim_ids)]
        end
    end
    println("3D geometry constructed with $(length(vIDs)) vertices and $(length(bIDs)) blocks, taking $(report_memory(ref_geometry))")
    return ref_geometry
end

function construct_BoundaryMesh(coors::CuArray, connections::CuArray)
    dim, _ = size(coors)
    vertices_per_block, _ = size(connections)

    if dim == 2 && vertices_per_block == 2
        ref_geometry = construct_BoundaryMesh_2D(coors, connections)
    elseif dim == 3 && vertices_per_block == 3
        ref_geometry = construct_BoundaryMesh_3D(coors, connections)
    else
        error("dim", dim, "vertices_per_block", vertices_per_block, "Undefined mesh type")
    end
end

function construct_BoundaryMesh_2D(coors::CuArray, connection::CuArray)
    ref_geometry = Geo_BoundaryMesh2D()
    @Takeout (vertices, segments) FROM ref_geometry
    vIDs = allocate_by_length!(vertices, size(coors, 2)) #Note here vIDs start exactly from 1, e.g., 1, 2,..., so connections don't need modification
    vertices.x1[vIDs] .= coors[1, :]
    vertices.x2[vIDs] .= coors[2, :]

    sIDs = allocate_by_length!(segments, size(connection, 2))
    segments.vertex_IDs[:, sIDs] .= connection   
    return ref_geometry
end

function construct_BoundaryMesh_3D(coors::CuArray, connection::CuArray)
    ref_geometry = Geo_BoundaryMesh3D()
    @Takeout (vertices, segments, faces) FROM ref_geometry
    vIDs = allocate_by_length!(vertices, size(coors, 2)) #Note here vIDs start exactly from 1, e.g., 1, 2,..., so connections don't need modification
    vertices.x1[vIDs] .= coors[1, :]
    vertices.x2[vIDs] .= coors[2, :]
    vertices.x3[vIDs] .= coors[3, :]

    fIDs = allocate_by_length!(faces, size(connection, 2))
    faces.vertex_IDs[:, fIDs] .= connection   

    seg_dict = dumb_GPUDict_Init(FEM_Int(0))
    for (f_s_pos, s_v_pos) in enumerate(F_S_V_POS[mesh_type])
        s_vIDs = connection[s_v_pos, :]
        rotate_size = length(s_v_pos)

        max_vIDs, cart_pos = findmax(s_vIDs, dims = 1) .|> vec
        max_pos = getindex.(cart_pos, 1)
        last_dim_ids = getindex.(cart_pos, 2)

        next_pos = get_Next_ID.(max_pos, rotate_size)
        next_vIDs = s_vIDs[CartesianIndex.(next_pos, last_dim_ids)]
        dict_keys = I4I30I30_To_UI64.(0, max_vIDs, next_vIDs)
        
        dict_slots = GPUDict_SetID(seg_dict, dict_keys)
        total_dict_IDs = get_Total_IDs(seg_dict)
        not_allocated = seg_dict.vals[total_dict_IDs] .== 0

        new_sIDs = allocate_by_length!(segments, sum(not_allocated))
        seg_dict.vals[total_dict_IDs[not_allocated]] .= new_sIDs

        local_sIDs = seg_dict.vals[dict_slots]
        faces.segment_IDs[f_s_pos, fIDs] .= local_sIDs

        segments.vertex_IDs[1, local_sIDs] .= max_vIDs
        segments.vertex_IDs[2, local_sIDs] .= next_vIDs
    end
    return ref_geometry
end

"""
    get_BoundaryMesh(total_mesh::Geo_TotalMesh2D)
    get_BoundaryMesh(total_mesh::Geo_TotalMesh3D)

Helper function to collect all the segment/face IDs which can be marked as boundary.
"""
function get_BoundaryMesh(total_mesh::Geo_TotalMesh2D)
    @Takeout (segments, faces) FROM total_mesh
    f_el_num = CUDA.zeros(FEM_Int, length(segments.is_occupied))
    elIDs = findall(faces.is_occupied)
    @Dumb_CUDA_Batch 256 inc_Num(f_el_num, vec(faces.segment_IDs[:, elIDs]))
    return findall(f_el_num .== 1)
end

function get_BoundaryMesh(total_mesh::Geo_TotalMesh3D)
    @Takeout (faces, blocks) FROM total_mesh
    f_el_num = CUDA.zeros(FEM_Int, length(faces.is_occupied))
    elIDs = findall(blocks.is_occupied)
    @Dumb_CUDA_Batch 256 inc_Num(f_el_num, vec(blocks.face_IDs[:, elIDs]))
    return findall(f_el_num .== 1)
end


