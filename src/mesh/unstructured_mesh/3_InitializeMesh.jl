function allocate_Basic_WP_Mesh_2D(wp::WorkPiece, this_space::Classical_Discretization)
    @Takeout (vertices, segments, faces) FROM wp.ref_geometry
    @Takeout (controlpoints, facets, elements, bg_fIDs) FROM wp.mesh #Note here facets refers to segments
    @Takeout (vertex_cp_ids, segment_cp_ids, face_cp_ids, segment_cp_pos, face_cp_pos, segment_start_vertex) FROM this_space.element_structure

    face_IDs = findall(faces.is_occupied)
    elIDs = allocate_by_length!(elements, length(face_IDs))
    prod(face_IDs .== elIDs) || error("face_IDs should be the same as elIDs")

    vIDs = findall(vertices.is_occupied)
    cp_per_vertex, vertex_per_element = size(vertex_cp_ids)
    v_cpIDs = allocate_by_length!(controlpoints, cp_per_vertex * length(vIDs))
    for i = 1:cp_per_vertex
        batch_cpIDs = v_cpIDs[i:cp_per_vertex:end]
        controlpoints.x1[batch_cpIDs] .= vertices.x1[vIDs]
        controlpoints.x2[batch_cpIDs] .= vertices.x2[vIDs]
    end
    for j = 1:vertex_per_element #separate with above for consistency
        for i = 1:cp_per_vertex 
            elements.controlpoint_IDs[vertex_cp_ids[i, j], elIDs] .= v_cpIDs[(faces.vertex_IDs[j, face_IDs] .- 1) .* cp_per_vertex .+ i]
        end
    end

    sIDs = findall(segments.is_occupied)
    segment_facet_IDs = CUDA.zeros(FEM_Int, length(sIDs))
    for (bg_index, boundary) in enumerate(wp.physics.boundarys)
        bdyIDs = allocate_by_length!(facets, length(boundary))
        bg_fIDs[bg_index] = bdyIDs
        segment_facet_IDs[boundary] .= bdyIDs
    end

    cp_per_segment, segment_per_element = size(segment_cp_ids)
    for j = 1:segment_per_element
        @Dumb_CUDA_Batch 256 specify_eindex(FEM_Int(j), segment_facet_IDs, faces.segment_IDs, facets.element_ID, facets.element_eindex, facets.outer_element_ID, 
        facets.outer_element_eindex, elIDs)
    end

    if cp_per_segment > 0
        s_cpIDs = allocate_by_length!(controlpoints, cp_per_segment * length(sIDs))
        segment_cp_pos = cu(segment_cp_pos)
        for i = 1:cp_per_segment
            batch_cpIDs = s_cpIDs[i:cp_per_segment:end]
            controlpoints.x1[batch_cpIDs] .= sum(segment_cp_pos .* vertices.x1[segments.vertex_IDs[:, sIDs]], dims = 1)[1, :]
            controlpoints.x2[batch_cpIDs] .= sum(segment_cp_pos .* vertices.x2[segments.vertex_IDs[:, sIDs]], dims = 1)[1, :]
        end
        for j = 1:segment_per_element
            is_aligned = segments.vertex_IDs[1, faces.segment_IDs[j, face_IDs]] .== faces.vertex_IDs[segment_start_vertex[j], face_IDs]
            a_elIDs = elIDs[is_aligned]
            n_elIDs = elIDs[.~ is_aligned]
            for i = 1:cp_per_segment
                elements.controlpoint_IDs[segment_cp_ids[i, j], a_elIDs] .= s_cpIDs[(faces.segment_IDs[j, a_elIDs] .- 1) .* cp_per_segment .+ i]
                elements.controlpoint_IDs[segment_cp_ids[i, j], n_elIDs] .= s_cpIDs[ faces.segment_IDs[j, n_elIDs] .* cp_per_segment .- (i - 1)]
            end 
        end
    end

    cp_per_face = length(face_cp_ids) #2d only has one face
    if cp_per_face > 0
        f_cpIDs = allocate_by_length!(controlpoints, cp_per_face * length(face_IDs))
        face_cp_pos = cu(face_cp_pos)
        for i = 1:cp_per_face
            batch_cpIDs = f_cpIDs[i:cp_per_face:end]
            controlpoints.x1[batch_cpIDs] .= sum(face_cp_pos .* vertices.x1[faces.vertex_IDs[:, face_IDs]], dims = 1)[1, :]
            controlpoints.x2[batch_cpIDs] .= sum(face_cp_pos .* vertices.x2[faces.vertex_IDs[:, face_IDs]], dims = 1)[1, :]

            elements.controlpoint_IDs[face_cp_ids[i], elIDs] .= batch_cpIDs
        end
    end
end

function allocate_Basic_WP_Mesh_3D(wp::WorkPiece, this_space::Classical_Discretization)
    @Takeout (vertices, segments, faces, blocks) FROM wp.ref_geometry
    @Takeout (controlpoints, facets, elements, bg_fIDs) FROM wp.mesh #Note here facets refers to segments
    @Takeout (vertex_cp_ids, segment_cp_ids, face_cp_ids, block_cp_ids, segment_cp_pos, face_cp_pos, block_cp_pos, 
             segment_start_vertex, face_start_segments) FROM this_space.element_structure

    block_IDs = findall(blocks.is_occupied)
    elIDs = allocate_by_length!(elements, length(block_IDs))
    prod(block_IDs .== elIDs) || error("block_IDs should be the same as elIDs")

    vIDs = findall(vertices.is_occupied)
    cp_per_vertex, vertex_per_element = size(vertex_cp_ids)
    v_cpIDs = allocate_by_length!(controlpoints, cp_per_vertex * length(vIDs))
    for i = 1:cp_per_vertex
        batch_cpIDs = v_cpIDs[i:cp_per_vertex:end]
        controlpoints.x1[batch_cpIDs] .= vertices.x1[vIDs]
        controlpoints.x2[batch_cpIDs] .= vertices.x2[vIDs]
        controlpoints.x3[batch_cpIDs] .= vertices.x3[vIDs]
    end
    for j = 1:vertex_per_element #separate with above for consistency
        for i = 1:cp_per_vertex 
            elements.controlpoint_IDs[vertex_cp_ids[i, j], elIDs] .= v_cpIDs[(blocks.vertex_IDs[j, block_IDs] .- 1) .* cp_per_vertex .+ i]
        end
    end

    sIDs = findall(segments.is_occupied)
    cp_per_segment, segment_per_element = size(segment_cp_ids)
    if cp_per_segment > 0
        s_cpIDs = allocate_by_length!(controlpoints, cp_per_segment * length(sIDs))
        segment_cp_pos = cu(segment_cp_pos)
        for i = 1:cp_per_segment
            batch_cpIDs = s_cpIDs[i:cp_per_segment:end]
            controlpoints.x1[batch_cpIDs] .= sum(segment_cp_pos .* vertices.x1[segments.vertex_IDs[:, sIDs]], dims = 1)[1, :]
            controlpoints.x2[batch_cpIDs] .= sum(segment_cp_pos .* vertices.x2[segments.vertex_IDs[:, sIDs]], dims = 1)[1, :]
            controlpoints.x3[batch_cpIDs] .= sum(segment_cp_pos .* vertices.x3[segments.vertex_IDs[:, sIDs]], dims = 1)[1, :]
        end

        for j = 1:segment_per_element
            is_aligned = segments.vertex_IDs[1, blocks.segment_IDs[j, block_IDs]] .== blocks.vertex_IDs[segment_start_vertex[j], block_IDs]

            a_elIDs = elIDs[is_aligned]
            n_elIDs = elIDs[.~ is_aligned]
            for i = 1:cp_per_segment
                elements.controlpoint_IDs[segment_cp_ids[i, j], a_elIDs] .= s_cpIDs[(blocks.segment_IDs[j, a_elIDs] .- 1) .* cp_per_segment .+ i]
                elements.controlpoint_IDs[segment_cp_ids[i, j], n_elIDs] .= s_cpIDs[ blocks.segment_IDs[j, n_elIDs] .* cp_per_segment .- (i - 1)]
            end 
        end
    end

    face_IDs = findall(faces.is_occupied)
    face_facet_IDs = CUDA.zeros(FEM_Int, length(face_IDs))
    for (bg_index, boundary) in enumerate(wp.physics.boundarys)
        bdyIDs = allocate_by_length!(facets, length(boundary))
        bg_fIDs[bg_index] = bdyIDs
        face_facet_IDs[boundary] .= bdyIDs
    end

    cp_per_face, face_per_element = size(face_cp_ids)
    for j = 1:face_per_element
        @Dumb_CUDA_Batch 256 specify_eindex(FEM_Int(j), face_facet_IDs, blocks.face_IDs, facets.element_ID, facets.element_eindex, 
        facets.outer_element_ID, facets.outer_element_eindex, elIDs)
    end
    
    if cp_per_face > 0
        if cp_per_face > 1
            error("TO DO face control point matching")
        end
        f_cpIDs = allocate_by_length!(controlpoints, cp_per_face * length(face_IDs))
        face_cp_pos = cu(face_cp_pos)
        for i = 1:cp_per_face
            batch_cpIDs = f_cpIDs[i:cp_per_face:end]
            controlpoints.x1[batch_cpIDs] .= sum(face_cp_pos .* vertices.x1[faces.vertex_IDs[:, face_IDs]], dims = 1)[1, :]
            controlpoints.x2[batch_cpIDs] .= sum(face_cp_pos .* vertices.x2[faces.vertex_IDs[:, face_IDs]], dims = 1)[1, :]
            controlpoints.x3[batch_cpIDs] .= sum(face_cp_pos .* vertices.x3[faces.vertex_IDs[:, face_IDs]], dims = 1)[1, :]
        end

        for j = 1:face_per_element
            elements.controlpoint_IDs[face_cp_ids[1, j], elIDs] .= f_cpIDs[blocks.face_IDs[j, elIDs]]
        end
    end

    cp_per_block = length(block_cp_ids) #2d only has one face
    if cp_per_block > 0
        b_cpIDs = allocate_by_length!(controlpoints, cp_per_block * length(block_IDs))
        block_cp_pos = cu(block_cp_pos)
        for i = 1:cp_per_block
            batch_cpIDs = b_cpIDs[i:cp_per_block:end]
            controlpoints.x1[batch_cpIDs] .= sum(block_cp_pos .* vertices.x1[blocks.vertex_IDs[:, block_IDs]], dims = 1)[1, :]
            controlpoints.x2[batch_cpIDs] .= sum(block_cp_pos .* vertices.x2[blocks.vertex_IDs[:, block_IDs]], dims = 1)[1, :]
            controlpoints.x3[batch_cpIDs] .= sum(block_cp_pos .* vertices.x3[blocks.vertex_IDs[:, block_IDs]], dims = 1)[1, :]

            elements.controlpoint_IDs[block_cp_ids[i], elIDs] .= batch_cpIDs
        end
    end
end

@Dumb_Kernel specify_eindex(f_pos, face_facet_mapping, cell_face_IDs, f_el_ID, eindex, outer_f_el_ID, outer_eindex, elIDs) begin
    this_elID = elIDs[thread_idx]
    # this_face_ID = cell_face_IDs[f_pos, this_elID]
    this_facet_ID = face_facet_mapping[cell_face_IDs[f_pos, this_elID]]
    this_facet_ID == 0 && return

    old_elID = CUDA.atomic_cas!(pointer(f_el_ID) + sizeof(eltype(f_el_ID)) * (this_facet_ID - 1), Int32(0), Int32(this_elID)) #Note dict key cannot be 0
    if old_elID != 0
        outer_f_el_ID[this_facet_ID] = this_elID
        outer_eindex[this_facet_ID] = f_pos
    else
        eindex[this_facet_ID] = f_pos
    end
end

