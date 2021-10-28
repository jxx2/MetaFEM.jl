function make_SquareChain(start_x1, start_x2, l1, l2)
    x1s = [0., l1, l1, 0.] .+ start_x1
    x2s = [0., 0., l2, l2] .+ start_x2
    return hcat(x1s, x2s)' |> collect |>cu
end

function make_CircleChain(center_x1, center_x2, r, resolution)
    vnum = max(ceil(r * 2 * pi / resolution), 6) |> FEM_Int
    dθ = 2 * pi / vnum
    x1s = [r * cos((vid - 1) * dθ) + center_x1 for vid = 1:vnum]
    x2s = [r * sin((vid - 1) * dθ) + center_x2 for vid = 1:vnum]
    return hcat(x1s, x2s)' |> collect |> cu
end