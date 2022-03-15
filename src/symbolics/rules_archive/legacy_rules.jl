"""
    This page is a list of all rules in effect. In practice, rule rewriting is good for flexibility/readability/design but not very performant, especially when a large number
of Kleen stars (a...) occurs, e.g., term merging is usually more efficient via dictionary. So we hard-coded a large number of rules as fixed built-in functions.
"""
@Define_Semantic_Constraint N₁ ∊ (N₁ isa Number)
@Define_Semantic_Constraint N₂ ∊ (N₂ isa Number)
@Define_Semantic_Constraint S₁ ∊ (S₁ isa SymbolicWord)
@Define_Semantic_Constraint S₂ ∊ (S₂ isa SymbolicWord)

@Define_Semantic_Constraint f_undef ∊ (~(f_undef in [:+; :-; :*; :/; :^; :log; :inv]))

@Define_Aux_Semantics N₁₊₂(N₁,N₂) = N₁ + N₂
@Define_Aux_Semantics N₁₊₊(N₁) = N₁ + 1
@Define_Aux_Semantics N₁₂(N₁,N₂) = N₁ * N₂
@Define_Aux_Semantics N₁ₚ₂(N₁,N₂) = N₁ ^ N₂

@Define_Rewrite_Rule Unary_Sub ≔ -(a) => (-1) * a
@Define_Rewrite_Rule Binary_Sub ≔ (b - a => b + (-1) * a)
@Define_Rewrite_Rule Binary_Div ≔ (a / b => a * b ^ (-1))

@Define_Rewrite_Rule Unary_Add ≔ (+(a) => a) #(shortcutted) built-in for performance
@Define_Rewrite_Rule Add_Splat ≔ ((a...) + (+(b...)) + (c...) => a + b + c)
@Define_Rewrite_Rule Add_Numbers ≔ ((a...) + N₁ + (b...) + N₂ + (c...) => N₁₊₂ + a + b + c)
@Define_Rewrite_Rule Add_Sort ≔ (a + (b...) + N₁ + (c...) => N₁ + a + b + c)
@Define_Rewrite_Rule Add_0 ≔ (0 + (a...) => +(a)) #(shortcutted) built-in for performance

@Define_Rewrite_Rule Unary_Mul ≔ (*(a) => a) #(shortcutted) built-in for performance
@Define_Rewrite_Rule Mul_Splat ≔ ((a...) * (*(b...)) * (c...) => a * b * c)
@Define_Rewrite_Rule Mul_Numbers ≔ ((a...) * N₁ * (b...) * N₂ * (c...) => N₁₂ * a * b * c)
@Define_Rewrite_Rule Mul_Sort ≔ (a * (b...) * N₁ * (c...) => N₁ * a * b * c)
@Define_Rewrite_Rule Mul_0 ≔ (0 * (a...) => 0) #(shortcutted) built-in for performance
@Define_Rewrite_Rule Mul_1 ≔ (1 * (a...) => *(a)) #(shortcutted) built-in for performance

@Define_Rewrite_Rule Pow_a0 ≔ (a ^ 0 => 1)
@Define_Rewrite_Rule Pow_a1 ≔ (a ^ 1 => a)
@Define_Rewrite_Rule Pow_0a ≔ (0 ^ a => 0)
@Define_Rewrite_Rule Pow_1a ≔ (1 ^ a => 1)
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

@Define_Rewrite_Rule Bilinear_Multiply ≔ (a...) * Bilinear(b, c) * (d...) => Bilinear(b, a * c * d)

@Define_Rewrite_Rule Bilinear_Add_Left_Divide ≔ Bilinear(a + (b...), c) => Bilinear(a, c) + Bilinear(+(b), c)
@Define_Rewrite_Rule Bilinear_Unary_Add_Left ≔ Bilinear(+(a), b) => Bilinear(a, b)
@Define_Rewrite_Rule Bilinear_0_Add_Left ≔ Bilinear(0, b) => 0.
@Define_Rewrite_Rule Bilinear_Mul_Left_Divide ≔ Bilinear(N₁ * (a...), b) => Bilinear(*(a), N₁ * b)

@Define_Rewrite_Rule Bilinear_Add_Right_Merge ≔ (a...) + Bilinear(b, c) + (d...) + Bilinear(b, e) + (f...) => a + d + f + Bilinear(b, c + e)
@Define_Rewrite_Rule Bilinear_Add_Right_Merge ≔ Bilinear(a, 0) => 0

@Define_Rewrite_Rule Bilinear_Add_Right_Divide ≔ Bilinear(a, b + (c...)) => Bilinear(a, b) + Bilinear(a, +(c))
@Define_Rewrite_Rule Bilinear_Unary_Add_Right ≔ Bilinear(a, +(b)) => Bilinear(a, b)

@Define_Rewrite_Rule Basic_Diff1 ≔ ∂(S₁, S₁) => 1
@Define_Rewrite_Rule Basic_Diff2 ≔ ∂(S₂, S₁) => 0
@Define_Rewrite_Rule Basic_Diff3 ≔ ∂(N₁, S₁) => 0

@Define_Rewrite_Rule Basic_Diff4 ≔ ∂({f_undef}(a...), S₁) => 0

@Define_Rewrite_Rule Add_Diff ≔ ∂(a + (b...), S₁) => ∂(a, S₁) + ∂(+(b...), S₁)
@Define_Rewrite_Rule Mul_Diff ≔ ∂(a * (b...), S₁) => ∂(a, S₁) * (b...) + a * ∂(*(b...), S₁)
@Define_Rewrite_Rule Pow_Diff ≔ ∂(a ^ b, S₁) => ∂(a, S₁) * a ^ (b - 1) * b  + ∂(b, S₁) * log(a) * a ^ b

@Define_Rewrite_Rule Spatial_Diff1 ≔  ∂(N₁, a) => 0
@Define_Rewrite_Rule Spatial_Diff2 ≔  ∂({f_undef}(a...), b) => 0
