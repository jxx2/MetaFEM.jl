shift_gauss_point(x) = x ./ 2.0 .+ 0.5
shift_gauss_weight(x) = x ./ 2.0
const GAUSS_POINT_POS_1D_ORIGINAL = ((0.0,), (-1.0/sqrt(3.0), 1.0/sqrt(3.0)), (-sqrt(3.0/5.0), 0.0, sqrt(3.0/5.0)),
                    (-sqrt(3.0/7.0 + 2.0/7.0 * sqrt(6.0/5.0)), -sqrt(3.0/7.0 - 2.0/7.0 * sqrt(6.0/5.0)),
                      sqrt(3.0/7.0 - 2.0/7.0 * sqrt(6.0/5.0)), sqrt(3.0/7.0 + 2.0/7.0 * sqrt(6.0/5.0))))

const GAUSS_POINT_WEIGHT_1D_ORIGINAL = ((2.0,), (1.0, 1.0), (5.0/9.0, 8.0/9.0, 5.0/9.0),
                        ((18.0 - sqrt(30.0))/36.0, (18.0 + sqrt(30.0))/36.0,
                         (18.0 + sqrt(30.0))/36.0, (18.0 - sqrt(30.0))/36.0))

const GAUSS_POINT_POS_1D_SHIFTED = shift_gauss_point.(GAUSS_POINT_POS_1D_ORIGINAL)
const GAUSS_POINT_WEIGHT_1D_SHIFTED = shift_gauss_weight.(GAUSS_POINT_WEIGHT_1D_ORIGINAL)

function init_Domain_Integration_Cube_Gauss(itg_order::Integer, dim::Integer)
    gauss_order = (itg_order + 1) / 2 |> ceil |> FEM_Int
    itg_pos = [getindex.(Ref(GAUSS_POINT_POS_1D_SHIFTED[gauss_order]), this_id) for this_id in Iterators.product([1:gauss_order for i=1:dim]...)] |> vec
    itg_weight = [prod(getindex.(Ref(GAUSS_POINT_WEIGHT_1D_SHIFTED[gauss_order]), this_id)) for this_id in Iterators.product([1:gauss_order for i=1:dim]...)] |> vec 
    return itg_pos, itg_weight
end

function init_Boundary_Integration_Cube_Gauss(itg_order::Integer, dim::Integer)
    gauss_order = (itg_order + 1) / 2 |> ceil |> FEM_Int
    itg_pos, base_itg_weight = init_Domain_Integration_Cube_Gauss(itg_order, dim - 1)
    if dim == 2
        face_ids = [4 2; 1 3]
    elseif dim == 3
        face_ids = [5 3; 2 4; 1 6]
    else
        error("Undefined dim")
    end

    itg_num, face_num = (itg_pos, face_ids) .|> length
    bdy_itg_pos = [fill(const_Tup(0., dim), itg_num) for i = 1:face_num]
    bdy_tangent_directions = [zeros(FEM_Float, itg_num, dim, dim - 1) for i = 1:face_num]

    for normal_dim = 1:dim
        tangent_dim = [(i + normal_dim - 1) % dim + 1 for i = 1:(dim - 1)]
        for outward_direction = 0:1 #left, right
            this_face_id = face_ids[normal_dim, outward_direction + 1]
            raw_tangent = [j == tangent_dim[i] ? 1 : 0 for k = 1:itg_num, j = 1:dim, i = 1:(dim - 1)]
            # println(raw_tangent)
            if dim == 2
                ((outward_direction + normal_dim) == 2) || (raw_tangent .*= -1)
            elseif dim == 3
                (outward_direction == 0) && (raw_tangent[:, :, 1] .*= -1)
            end
            bdy_tangent_directions[this_face_id] .= raw_tangent
            for i = 1:itg_num
                this_pos = zeros(FEM_Float, dim)
                this_pos[tangent_dim] .= itg_pos[i]
                this_pos[normal_dim] = outward_direction
                bdy_itg_pos[this_face_id][i] = Tuple(this_pos)
            end
        end
    end

    return bdy_itg_pos, [copy(base_itg_weight) for i = 1:length(face_ids)], bdy_tangent_directions
end
##
#Order 5, 6, 8
const GAUSS_POINT_POS_TRIANGLE = (((0.10128650732345633880098736191512383,), (0.47014206410511508977044120951344760,), ()), 
((0.06308901449150222834033160287081916,), (0.24928674517091042129163855310701908,), (0.05314504984481694735324967163139815, 0.31035245103378440541660773395655215)),
((), (0.17056930775176020662229350149146450,), (0.05054722831703097545842355059659895,), (0.45929258829272315602881551449416932,), (0.26311282963463811342178578628464359, 0.00839477740995760533721383453929445)))

const GAUSS_POINT_WEIGHT_TRIANGLE = ((0.12593918054482715259568394550018133, 0.13239415278850618073764938783315200, 9.0/40.0),
(0.05084490637020681692093680910686898, 0.11678627572637936602528961138557944, 0.08285107561837357519355345642044245),
(0.14431560767778716825109111048906462, 0.10321737053471825028179155029212903, 0.03245849762319808031092592834178060, 0.09509163426728462479389610438858432, 0.02723031417443499426484469007390892))

# (a,) is perm31 (-a,) is 22
const GAUSS_POINT_POS_TETRAHEDRON = (((0.31088591926330060979734573376345783,), (0.09273525031089122640232391373703061,), (-0.04550370412564964949188052627933943,)), 
((0.21460287125915202928883921938628499,), (0.04067395853461135311557944895641006,), (0.32233789014227551034399447076249213,), (0.06366100187501752529923552760572698, 0.60300566479164914136743113906093969)),
((0.03967542307038990126507132953938949,), (0.31448780069809631378416056269714830,), (0.10198669306270330000000000000000000,), (0.18420369694919151227594641734890918,), (-0.06343628775453989240514123870189827,),
 (0.02169016206772800480266248262493018, 0.71993192203946593588943495335273478), (0.20448008063679571424133557487274534, 0.58057719012880922417539817139062041)))

const GAUSS_POINT_WEIGHT_TETRAHEDRON = ((0.11268792571801585079918565233328633, 0.07349304311636194954371020548632750, 0.04254602077708146643806942812025744),
(0.03992275025816749209969062755747998, 0.01007721105532064294801323744593686, 0.05535718154365472209515327785372602, 27.0/560.0),
(0.00639714777990232132145142033517302, 0.04019044802096617248816115847981783, 0.02430797550477032117486910877192260, 0.05485889241369744046692412399039144,
 0.03571961223409918246495096899661762, 0.00718319069785253940945110521980376, 0.01637218194531911754093813975611913))

function _init_Integration_Triangle_Gauss(itg_order::Integer)
    if itg_order <= 5
        id = 1
    elseif itg_order <= 6
        id = 2
    elseif itg_order <= 8
        id = 3
    else
        error("Wrong integral order")
    end

    itg_pos = Tuple{Vararg{FEM_Float, 3}}[]
    itg_weight = FEM_Float[]

    for (pos, weight) in zip(GAUSS_POINT_POS_TRIANGLE[id], GAUSS_POINT_WEIGHT_TRIANGLE[id])
        if length(pos) == 0
            push!(itg_pos, const_Tup(1/3, 3))
            push!(itg_weight, weight)
        elseif length(pos) == 1
            a = pos[1]
            append!(itg_pos, [basis_Tup(i, 3, (1 - 2 * a), a) for i = 1:3])
            append!(itg_weight, [weight for i = 1:3])
        elseif length(pos) == 2
            c = 1. - sum(pos)
            src_pos = (pos..., c)
            for (i, j) in Iterators.product([1:3 for tmp = 1:2]...)
                (i == j) && continue
                k = 6 - i - j 
                push!(itg_pos, src_pos[[i, j, k]])
                push!(itg_weight, weight)
            end
        end
    end

    return itg_pos, itg_weight
end

function init_Domain_Integration_Triangle_Gauss(itg_order::Integer)
    itg_pos, itg_weight = _init_Integration_Triangle_Gauss(itg_order)
    return getindex.(itg_pos, Ref([2, 3])), itg_weight ./ 2.
end

function init_Boundary_Integration_Triangle_Gauss(itg_order::Integer)
    gauss_order = (itg_order + 1) / 2 |> ceil |> FEM_Int
    itg_pos, base_itg_weight = init_Domain_Integration_Cube_Gauss(gauss_order, 1)
    bdy_itg_weights = [copy(base_itg_weight) for i = 1:3]
    bdy_itg_weights[2] .*= sqrt(2)

    itg_num, face_num, dim = (length(itg_pos), 3, 2)
    bdy_itg_pos = [fill(const_Tup(0., dim), itg_num) for i = 1:face_num]
    bdy_tangent_directions = [zeros(FEM_Float, itg_num, dim, dim - 1) for i = 1:face_num]

    for i = 1:itg_num
        a = itg_pos[i][1]

        bdy_itg_pos[1][i] = (1. - a) .* (0, 0) .+ a .* (1, 0)
        bdy_itg_pos[2][i] = (1. - a) .* (1, 0) .+ a .* (0, 1)
        bdy_itg_pos[3][i] = (1. - a) .* (0, 1) .+ a .* (0, 0)
    end
    bdy_tangent_directions[1][:, :, 1] .= [1. 0.]
    bdy_tangent_directions[2][:, :, 1] .= [-1. 1.] ./ sqrt(2)
    bdy_tangent_directions[3][:, :, 1] .= [0. -1.]
    return bdy_itg_pos, bdy_itg_weights, bdy_tangent_directions
end

function _init_Integration_Tetrahedron_Gauss(itg_order::Integer)
    if itg_order <= 5
        id = 1
    elseif itg_order <= 6
        id = 2
    elseif itg_order <= 8
        id = 3
    else
        error("Wrong integral order")
    end

    itg_pos = Tuple{Vararg{FEM_Float, 4}}[]
    itg_weight = FEM_Float[]

    for (pos, weight) in zip(GAUSS_POINT_POS_TETRAHEDRON[id], GAUSS_POINT_WEIGHT_TETRAHEDRON[id])
        if length(pos) == 0
            push!(itg_pos, const_Tup(1/4, 4))
            push!(itg_weight, weight)
        elseif length(pos) == 1
            a = pos[1]
            if a >= 0
                append!(itg_pos, [basis_Tup(i, 4, (1 - 3 * a), a) for i = 1:4])
                append!(itg_weight, [weight for i = 1:4])
            else
                b = -a
                for (i, j) in Iterators.product([1:4 for tmp = 1:2]...)
                    (i >= j) && continue
                    src_pos = fill(b, 4)
                    src_pos[[i,j]] .= 0.5 - b
                    push!(itg_pos, Tuple(src_pos))
                    push!(itg_weight, weight)
                end
            end
        elseif length(pos) == 2
            a, b = pos
            c = 1 - 2 * a - b
            for (i, j) in Iterators.product([1:4 for tmp = 1:2]...)
                (i == j) && continue
                src_pos = fill(a, 4)
                src_pos[i] = b
                src_pos[j] = c
                push!(itg_pos, Tuple(src_pos))
                push!(itg_weight, weight)
            end
        elseif length(pos) == 3
            d = 1. - sum(pos)
            src_pos = (pos..., d)
            for (i, j, k) in Iterators.product([1:4 for tmp = 1:3]...)
                ((i == j) || (i == k) || (j == k)) && continue
                l = 10 - i - j - k 
                push!(itg_pos, src_pos[[i, j, k, l]])
                push!(itg_weight, weight)
            end
        end
    end
    return itg_pos, itg_weight 
end

function init_Domain_Integration_Tetrahedron_Gauss(itg_order::Integer)
    itg_pos, itg_weight = _init_Integration_Tetrahedron_Gauss(itg_order)
    return getindex.(itg_pos, Ref([2, 3, 4])), itg_weight ./ 6
end

function init_Boundary_Integration_Tetrahedron_Gauss(itg_order::Integer)
    itg_pos, base_itg_weight = _init_Integration_Triangle_Gauss(itg_order)
    bdy_itg_weights = [copy(base_itg_weight) * 0.5 for i = 1:4]
    bdy_itg_weights[3] .*= sqrt(3)

    itg_num, face_num, dim = (length(itg_pos), 4, 3)
    bdy_itg_pos = [fill(const_Tup(0., dim), itg_num) for i = 1:face_num]
    bdy_tangent_directions = [zeros(FEM_Float, itg_num, dim, dim - 1) for i = 1:face_num]

    for i = 1:itg_num
        a, b, c = itg_pos[i]

        bdy_itg_pos[1][i] = a .* (0, 0, 0) .+ b .* (1, 0, 0) .+ c .* (0, 1, 0)
        bdy_itg_pos[2][i] = a .* (0, 0, 0) .+ b .* (1, 0, 0) .+ c .* (0, 0, 1)
        bdy_itg_pos[3][i] = a .* (0, 0, 1) .+ b .* (1, 0, 0) .+ c .* (0, 1, 0)
        bdy_itg_pos[4][i] = a .* (0, 0, 0) .+ b .* (0, 1, 0) .+ c .* (0, 0, 1)
    end
    
    bdy_tangent_directions[1][:, :, 1] .= [-1. 0. 0.]
    bdy_tangent_directions[1][:, :, 2] .= [ 0. 1. 0.]

    bdy_tangent_directions[2][:, :, 1] .= [0. 0. -1.]
    bdy_tangent_directions[2][:, :, 2] .= [1. 0.  0.]

    bdy_tangent_directions[3][:, :, 1] .= [-1. 1. 0.] ./ sqrt(2)
    bdy_tangent_directions[3][:, :, 2] .= [-1. -1. 2.] ./ sqrt(6) #must be orthogonal

    bdy_tangent_directions[4][:, :, 1] .= [0. -1. 0.]
    bdy_tangent_directions[4][:, :, 2] .= [0.  0. 1.]
    return bdy_itg_pos, bdy_itg_weights, bdy_tangent_directions
end
