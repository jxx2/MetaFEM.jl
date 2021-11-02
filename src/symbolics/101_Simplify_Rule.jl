# for simplification
@Define_Semantic_Constraint N₁ ∊ (N₁ isa Number)
@Define_Semantic_Constraint N₂ ∊ (N₂ isa Number)

@Define_Aux_Semantics N₁₊₂(N₁,N₂) = N₁ + N₂
@Define_Aux_Semantics N₁₊₊(N₁) = N₁ + 1
@Define_Aux_Semantics N₁₂(N₁,N₂) = N₁ * N₂
@Define_Aux_Semantics N₁ₚ₂(N₁,N₂) = N₁ ^ N₂

# @Define_Rewrite_Rule Unary_Sub ≔ -(a) => (-1) * a
@Define_Rewrite_Rule Binary_Sub ≔ (b - a => b + (-1) * a)
@Define_Rewrite_Rule Binary_Div ≔ (a / b => a * b ^ (-1))

# @Define_Rewrite_Rule Unary_Add ≔ (+(a) => a) #(shortcutted) built-in for performance
@Define_Rewrite_Rule Add_Splat ≔ ((a...) + (+(b...)) + (c...) => a + b + c)
@Define_Rewrite_Rule Add_Numbers ≔ ((a...) + N₁ + (b...) + N₂ + (c...) => N₁₊₂ + a + b + c)
@Define_Rewrite_Rule Add_Sort ≔ (a + (b...) + N₁ + (c...) => N₁ + a + b + c)
# @Define_Rewrite_Rule Add_0 ≔ (0 + (a...) => +(a)) #(shortcutted) built-in for performance

# @Define_Rewrite_Rule Unary_Mul ≔ (*(a) => a) #(shortcutted) built-in for performance
@Define_Rewrite_Rule Mul_Splat ≔ ((a...) * (*(b...)) * (c...) => a * b * c)
@Define_Rewrite_Rule Mul_Numbers ≔ ((a...) * N₁ * (b...) * N₂ * (c...) => N₁₂ * a * b * c)
@Define_Rewrite_Rule Mul_Sort ≔ (a * (b...) * N₁ * (c...) => N₁ * a * b * c)
# @Define_Rewrite_Rule Mul_0 ≔ (0 * (a...) => 0) #(shortcutted) built-in for performance
# @Define_Rewrite_Rule Mul_1 ≔ (1 * (a...) => *(a)) #(shortcutted) built-in for performance

# @Define_Rewrite_Rule Pow_a0 ≔ (a ^ 0 => 1)
# @Define_Rewrite_Rule Pow_a1 ≔ (a ^ 1 => a)
# @Define_Rewrite_Rule Pow_0a ≔ (0 ^ a => 0)
# @Define_Rewrite_Rule Pow_1a ≔ (1 ^ a => 1)
@Define_Rewrite_Rule Pow_Numbers ≔ (N₁ ^ N₂ => N₁ₚ₂)
@Define_Rewrite_Rule Pow_Splat ≔ ((a ^ b) ^ c => a ^ (b * c))

@Define_Rewrite_Rule Distributive_MP ≔ ((a * (b...)) ^ c => (a ^ c) * (*(b)) ^ c)
@Define_Rewrite_Rule Distributive_AM ≔ ((a...) * (b + (c...)) * (d...) => a * b * d + a * (+(c)) * d)

@Define_Rewrite_Rule Merge_Add_NN ≔ ((a...) + N₁ * (b...) + (c...) + N₂ * (b...) + (d...) => a + c + d + N₁₊₂ * b)
@Define_Rewrite_Rule Merge_Add_N1_single ≔ ((a...) + N₁ * b + (c...) + b + (d...) => a + c + d + N₁₊₊ * b)
@Define_Rewrite_Rule Merge_Add_N1 ≔ ((a...) + N₁ * (b...) + (c...) + *(b...) + (d...) => a + c + d + N₁₊₊ * b)
@Define_Rewrite_Rule Merge_Add_1N_single ≔ ((a...) + b + (c...) + N₁ * b + (d...) => a + c + d + N₁₊₊ * b)
@Define_Rewrite_Rule Merge_Add_1N ≔ ((a...) + *(b...) + (c...) + N₁ * (b...) + (d...) => a + c + d + N₁₊₊ * b)
@Define_Rewrite_Rule Merge_Add_11 ≔ ((a...) + b + (c...) + b + (d...) => a + c + d + 2 * b)

@Define_Rewrite_Rule Merge_Mul_NN ≔ ((a...) * b ^ p1 * (c...) * b ^ p2 * (d...) => a * c * d * b ^ (p1 + p2))
@Define_Rewrite_Rule Merge_Mul_N1 ≔ ((a...) * b ^ p1 * (c...) * b * (d...) => a * c * d * b ^ (1 + p1))
@Define_Rewrite_Rule Merge_Mul_1N ≔ ((a...) * b * (c...) * b ^ p2 * (d...) => a * c * d * b ^ (1 + p2))
@Define_Rewrite_Rule Merge_Mul_11 ≔ ((a...) * b * (c...) * b * (d...) => a * c * d * b ^ 2)

Remove_MinDiv_Rules = [Binary_Sub, Binary_Div]
Simplifer_Add_Rules = [Add_Splat, Add_Numbers, Add_Sort]
Simplifer_Mul_Rules = [Mul_Splat, Mul_Numbers, Mul_Sort]
Simplifer_Pow_Rules = [Pow_Numbers, Pow_Splat]

Distribute_Rules = [Distributive_MP, Distributive_AM]

Merge_Muls_Rules = [Merge_Mul_NN, Merge_Mul_N1, Merge_Mul_1N, Merge_Mul_11]
Merge_Adds_Rules = [Merge_Add_NN, Merge_Add_N1_single, Merge_Add_N1, Merge_Add_1N_single, Merge_Add_1N, Merge_Add_11]

@Compile_Rewrite_Rules remove_MinDiv Remove_MinDiv_Rules
@Compile_Rewrite_Rules simplify_AMP (Simplifer_Add_Rules, Simplifer_Mul_Rules, Simplifer_Pow_Rules)
@Compile_Rewrite_Rules expand_Expression Distribute_Rules

@Compile_Rewrite_Rules merge_Add Merge_Adds_Rules
@Compile_Rewrite_Rules merge_All (Merge_Adds_Rules, Merge_Muls_Rules)

function simplify_Basic(source_term::GroundTerm)

    _, term_remove_MinDiv = remove_MinDiv(source_term)
    term_final = term_remove_MinDiv
    while true
        _, term_AMP_1 = simplify_AMP(term_final)

        changed_Expand, term_Expand = expand_Expression(term_AMP_1)
        changed_AMP_2, term_AMP_2 = simplify_AMP(term_Expand)
        changed_Merge, term_final = merge_Add(term_AMP_2) #note we dont want to rewrite uiui into ui ^ 2 which violate dumb index
        if ~(changed_Expand || changed_AMP_2 || changed_Merge)
            break
        end
    end
    return term_final
end

function simplify_Merge(source_term::GroundTerm)

    _, term_remove_MinDiv = remove_MinDiv(source_term)
    term_final = term_remove_MinDiv
    while true
        _, term_AMP_1 = simplify_AMP(term_final)

        changed_Expand, term_Expand = expand_Expression(term_AMP_1)
        changed_AMP_2, term_AMP_2 = simplify_AMP(term_Expand)
        changed_Merge, term_final = merge_All(term_AMP_2)
        if ~(changed_Expand || changed_AMP_2 || changed_Merge)
            break
        end
    end
    return term_final
end
