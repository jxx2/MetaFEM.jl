function write_VTK(fname::String, wp::WorkPiece; scale = 1.)
    @Takeout (element_space.element_attributes, local_assembly.local_innervar_infos) FROM wp
    @Takeout (controlpoints, elements) FROM wp.mesh
    
    dim, shape, itp_order = element_attributes[:dim], element_attributes[:shape], element_attributes[:itp_order]
    raw_cp_IDs = findall(controlpoints.is_occupied) 
    cp_num = length(raw_cp_IDs)

    raw_el_IDs = findall(elements.is_occupied)
    el_num = length(raw_el_IDs)
    el_cpIDs = elements.controlpoint_IDs[:, raw_el_IDs]

    x1s = controlpoints.x1[raw_cp_IDs] 
    x2s = controlpoints.x2[raw_cp_IDs] 
    if dim == 2
        xs = hcat(x1s, x2s, zeros(FEM_Float, size(x1s))) |> collect
        if shape == :CUBE
            if element_attributes[:itp_type] == :Lagrange
                if itp_order == 1
                    cell_type = 9 # VTK_QUAD
                    el_cp_outer_id = [1, 2, 4, 3]
                elseif itp_order == 2
                    cell_type = 23 # VTK_QUADRATIC_QUAD
                    el_cp_outer_id = [1, 3, 9, 7, 2, 6, 8 ,4]
                else
                    error("Wrong itp_order")
                end
            elseif element_attributes[:itp_type] == :Serendipity
                if itp_order == 1
                    cell_type = 9 # VTK_QUAD
                    el_cp_outer_id = [1, 2, 4, 3]
                elseif itp_order == 2
                    cell_type = 23 # VTK_QUADRATIC_QUAD
                    el_cp_outer_id = [1, 2, 4, 3, 5, 8, 6, 7]
                else
                    error("Wrong itp_order")
                end      
            else
                error("Wrong itp type")          
            end
        elseif shape == :SIMPLEX
                if itp_order == 1
                cell_type = 5 # VTK_TRIANGLE
                el_cp_outer_id = [1, 2, 3]
            elseif itp_order == 2
                cell_type = 22 # VTK_QUADRATIC_TRIANGLE
                el_cp_outer_id = [1, 3, 6, 2, 5, 4]
            else
                error("Wrong itp_order")
            end 
        else
            error("Wrong shape")
        end
    elseif dim == 3
        x3s = controlpoints.x3[raw_cp_IDs] 
        xs = hcat(x1s, x2s, x3s) |> collect
        if shape == :CUBE
            if element_attributes[:itp_type] == :Lagrange
                if itp_order == 1
                    cell_type = 12 # VTK_HEXAHEDRON
                    el_cp_outer_id = [1, 2, 4, 3, 5, 6, 8, 7]
                elseif itp_order == 2
                    cell_type = 25 # VTK_QUADRATIC_HEXAHEDRON
                    el_cp_outer_id = [1, 3, 9, 7, 19, 21, 27, 25, 
                                    2, 6, 8, 4, 20, 24, 26, 22,
                                    10, 12, 18, 16]
                else
                    error("Wrong itp_order")
                end
            elseif element_attributes[:itp_type] == :Serendipity
                if itp_order == 1
                    cell_type = 12 # VTK_HEXAHEDRON
                    el_cp_outer_id = [1, 2, 4, 3, 5, 6, 8, 7]
                elseif itp_order == 2
                    cell_type = 25 # VTK_QUADRATIC_HEXAHEDRON
                    el_cp_outer_id = [1, 2, 4, 3, 5, 6, 8, 7, 
                                    9, 14, 10, 13, 11, 16, 12, 15,
                                    17, 18, 20, 19]
                else
                    error("Wrong itp_order")
                end
            else
                error("Wrong itp type")
            end
        elseif shape == :SIMPLEX
                if itp_order == 1
                cell_type = 10 # VTK_TETRA
                el_cp_outer_id = [1, 2, 3, 4]
            elseif itp_order == 2
                cell_type = 24 # VTK_QUADRATIC_TETRA
                el_cp_outer_id = [1, 3, 6, 10, 2, 5, 4, 7, 8, 9]
                # cell_type = 10 # VTK_TETRA
                # el_cp_outer_id = [1, 3, 6, 10]
            else
                error("Wrong itp_order")
            end 
        else
            error("Wrong shape")
        end
    else
        error("Wrong element dim")
    end
    xs = xs.*(scale)
    cell_size = length(el_cp_outer_id)

    isfile(fname) && rm(fname)
    open(fname, "a+") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, fname)
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")

        println(io, join(["POINTS", cp_num, "float"], " "))
        for i = 1:cp_num
            println(io, join(xs[i, :], " "))
        end

        cp_mapping = Dict(collect(raw_cp_IDs) .=> 0:(cp_num - 1)) 
        map_cp(x) = cp_mapping[x]
        println(io, join(["CELLS", el_num, el_num * (1 + cell_size)], " "))
        for i = 1:el_num
            println(io, join([cell_size, map_cp.(collect(el_cpIDs[el_cp_outer_id, i]))...], " "))
        end    

        println(io, join(["CELL_TYPES ", el_num], " "))
        for i = 1:el_num
            println(io, cell_type)
        end    

        local_innervars = getindex.(local_innervar_infos, 1)
        println(io, join(["POINT_DATA", cp_num], " "))
        for sym in local_innervars
            println(io, join(["SCALARS", sym, "float", 1], " "))
            println(io, "LOOKUP_TABLE ", "default")
            local_data = get_Data(controlpoints)[sym][raw_cp_IDs] |> collect
            for i = 1:cp_num
                println(io, local_data[i])
            end
        end
    end  
end
