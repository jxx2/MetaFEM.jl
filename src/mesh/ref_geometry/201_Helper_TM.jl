"""
    make_Square(x::Tuple, n::Tuple, shape = :CUBE)
    make_Brick(x::Tuple, n::Tuple, shape = :CUBE)

Helper functions to creates a grid, i.e., the mesh nodes are positioned at cubic lattice points, with mesh shape = `:CUBE` or `:SIMPLEX` defining the connection.
"""
function make_Square(x::Tuple, n::Tuple, shape = :CUBE)
    dx = x ./ n
    coors = zeros(FEM_Float, 2, prod(n .+ 1))
    coors[1, :] .= [dx[1] * i for i = 0:n[1] for j = 0:n[2]]
    coors[2, :] .= [dx[2] * j for i = 0:n[1] for j = 0:n[2]]

    cube_connections = zeros(FEM_Int, 4, prod(n))
    cube_connections[1, :] .= [(i - 1) * (n[2] + 1) + j     for i = 1:n[1] for j = 1:n[2]]
    cube_connections[2, :] .= [(i    ) * (n[2] + 1) + j     for i = 1:n[1] for j = 1:n[2]]
    cube_connections[3, :] .= [(i    ) * (n[2] + 1) + j + 1 for i = 1:n[1] for j = 1:n[2]]
    cube_connections[4, :] .= [(i - 1) * (n[2] + 1) + j + 1 for i = 1:n[1] for j = 1:n[2]]

    if shape == :CUBE
        connections = cube_connections
    elseif shape == :SIMPLEX
        connections = zeros(FEM_Int, 3, 2 * prod(n))
        el_1 = 1:prod(n)
        el_2 = (1:prod(n)) .+ prod(n)

        connections[:, el_1] .= cube_connections[[1, 2, 4], :] 
        # connections[:, el_2] .= cube_connections[[4, 3, 2], :] #wrong orientation!
        connections[:, el_2] .= cube_connections[[3, 4, 2], :] # OK but all minus
        # connections[:, el_2] .= cube_connections[[4, 2, 3], :] #twisted
    end

    return (coors, connections) 
end


function make_Brick(x::Tuple, n::Tuple, shape = :CUBE)
    dx = x ./ n
    coors = zeros(FEM_Float, 3, prod(n .+ 1))
    coors[1, :] .= [dx[1] * i for i = 0:n[1] for j = 0:n[2] for k = 0:n[3]]
    coors[2, :] .= [dx[2] * j for i = 0:n[1] for j = 0:n[2] for k = 0:n[3]]
    coors[3, :] .= [dx[3] * k for i = 0:n[1] for j = 0:n[2] for k = 0:n[3]]

    cube_connections = zeros(FEM_Int, 8, prod(n))
    cube_connections[1, :] .= [(i - 1) * (n[2] + 1) * (n[3] + 1) + (j - 1) * (n[3] + 1) + (k    ) for i = 1:n[1] for j = 1:n[2] for k = 1:n[3]]
    cube_connections[2, :] .= [(i    ) * (n[2] + 1) * (n[3] + 1) + (j - 1) * (n[3] + 1) + (k    ) for i = 1:n[1] for j = 1:n[2] for k = 1:n[3]]
    cube_connections[3, :] .= [(i    ) * (n[2] + 1) * (n[3] + 1) + (j    ) * (n[3] + 1) + (k    ) for i = 1:n[1] for j = 1:n[2] for k = 1:n[3]]
    cube_connections[4, :] .= [(i - 1) * (n[2] + 1) * (n[3] + 1) + (j    ) * (n[3] + 1) + (k    ) for i = 1:n[1] for j = 1:n[2] for k = 1:n[3]]
    cube_connections[5, :] .= [(i - 1) * (n[2] + 1) * (n[3] + 1) + (j - 1) * (n[3] + 1) + (k + 1) for i = 1:n[1] for j = 1:n[2] for k = 1:n[3]]
    cube_connections[6, :] .= [(i    ) * (n[2] + 1) * (n[3] + 1) + (j - 1) * (n[3] + 1) + (k + 1) for i = 1:n[1] for j = 1:n[2] for k = 1:n[3]]
    cube_connections[7, :] .= [(i    ) * (n[2] + 1) * (n[3] + 1) + (j    ) * (n[3] + 1) + (k + 1) for i = 1:n[1] for j = 1:n[2] for k = 1:n[3]]
    cube_connections[8, :] .= [(i - 1) * (n[2] + 1) * (n[3] + 1) + (j    ) * (n[3] + 1) + (k + 1) for i = 1:n[1] for j = 1:n[2] for k = 1:n[3]]

    if shape == :CUBE
        connections = cube_connections
    elseif shape == :SIMPLEX
        division = 5
        connections = zeros(FEM_Int, 4, division * prod(n))

        el_ids = collect(1:prod(n))
        is_odd = [(i + j + k) % 2 == 1 for i = 1:n[1] for j = 1:n[2] for k = 1:n[3]]

        forward_el_ids = el_ids[is_odd]
        backward_el_ids = el_ids[.~is_odd]

        connections[:, forward_el_ids .+ 0 * prod(n)] .= cube_connections[[1, 2, 4, 5], forward_el_ids]
        connections[:, forward_el_ids .+ 1 * prod(n)] .= cube_connections[[3, 4, 2, 7], forward_el_ids]
        connections[:, forward_el_ids .+ 2 * prod(n)] .= cube_connections[[8, 7, 5, 4], forward_el_ids]
        connections[:, forward_el_ids .+ 3 * prod(n)] .= cube_connections[[6, 5, 7, 2], forward_el_ids]
        connections[:, forward_el_ids .+ 4 * prod(n)] .= cube_connections[[4, 7, 5, 2], forward_el_ids]

        connections[:, backward_el_ids .+ 0 * prod(n)] .= cube_connections[[5, 8, 6, 1], backward_el_ids]
        connections[:, backward_el_ids .+ 1 * prod(n)] .= cube_connections[[2, 1, 6, 3], backward_el_ids]
        connections[:, backward_el_ids .+ 2 * prod(n)] .= cube_connections[[7, 6, 8, 3], backward_el_ids]
        connections[:, backward_el_ids .+ 3 * prod(n)] .= cube_connections[[4, 1, 3, 8], backward_el_ids]
        connections[:, backward_el_ids .+ 4 * prod(n)] .= cube_connections[[1, 3, 8, 6], backward_el_ids]
    end
    return (coors, connections) 
end