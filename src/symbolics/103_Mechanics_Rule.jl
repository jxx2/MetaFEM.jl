#may design a better syntax later
@Define_Rewrite_Rule Kronecker_Delta1 ≔ [δ{1,1}] => 1
@Define_Rewrite_Rule Kronecker_Delta2 ≔ [δ{2,2}] => 1
@Define_Rewrite_Rule Kronecker_Delta3 ≔ [δ{3,3}] => 1
@Define_Rewrite_Rule Kronecker_Delta4 ≔ [δ{1,2}] => 0
@Define_Rewrite_Rule Kronecker_Delta5 ≔ [δ{2,1}] => 0
@Define_Rewrite_Rule Kronecker_Delta6 ≔ [δ{2,3}] => 0
@Define_Rewrite_Rule Kronecker_Delta7 ≔ [δ{3,2}] => 0
@Define_Rewrite_Rule Kronecker_Delta8 ≔ [δ{3,1}] => 0
@Define_Rewrite_Rule Kronecker_Delta9 ≔ [δ{1,3}] => 0

@Define_Rewrite_Rule Bilinear_Multiply ≔ (a...) * Bilinear(b, c) * (d...) => Bilinear(b, a * c * d)

@Define_Rewrite_Rule Bilinear_Add_Left_Divide ≔ Bilinear(a + (b...), c) => Bilinear(a, c) + Bilinear(+(b), c)
# @Define_Rewrite_Rule Bilinear_Unary_Add_Left ≔ Bilinear(+(a), b) => Bilinear(a, b)
# @Define_Rewrite_Rule Bilinear_0_Add_Left ≔ Bilinear(0, b) => 0.
@Define_Rewrite_Rule Bilinear_Mul_Left_Divide ≔ Bilinear(N₁ * (a...), b) => Bilinear(*(a), N₁ * b)

@Define_Rewrite_Rule Bilinear_Add_Right_Merge ≔ (a...) + Bilinear(b, c) + (d...) + Bilinear(b, e) + (f...) => a + d + f + Bilinear(b, c + e)
#@Define_Rewrite_Rule Bilinear_Add_Right_Merge ≔ Bilinear(a, 0) => 0

@Define_Rewrite_Rule Bilinear_Add_Right_Divide ≔ Bilinear(a, b + (c...)) => Bilinear(a, b) + Bilinear(a, +(c))
# @Define_Rewrite_Rule Bilinear_Unary_Add_Right ≔ Bilinear(a, +(b)) => Bilinear(a, b)

Kronecker_Delta_Rules = [Kronecker_Delta1, Kronecker_Delta2, Kronecker_Delta3, Kronecker_Delta4, Kronecker_Delta5, 
                         Kronecker_Delta6, Kronecker_Delta7, Kronecker_Delta8, Kronecker_Delta9]
Bilinear_Left_Rules = [Bilinear_Multiply, Bilinear_Add_Left_Divide, Bilinear_Mul_Left_Divide]
Bilinear_Right_Rules = [Bilinear_Add_Right_Merge]

@Compile_Rewrite_Rules Kronecker_Delta Kronecker_Delta_Rules
@Compile_Rewrite_Rules simplify_Bilinear_Left Bilinear_Left_Rules
@Compile_Rewrite_Rules simplify_Bilinear_Right Bilinear_Right_Rules

function simplify_Symbolics(source_term::GroundTerm)
    simplified_term = simplify_Basic(source_term)
    term_changed, simplified_term = simplify_Bilinear_Left(simplified_term)
    term_changed, simplified_term = simplify_Bilinear_Right(simplified_term)
    return simplify_Basic(simplified_term)
end

function simplify_Mechanics(source_term::GroundTerm)

    term_changed, mechanic_term = Kronecker_Delta(source_term)
    simplified_term = simplify_Merge(mechanic_term)

    term_changed, simplified_term = simplify_Bilinear_Left(simplified_term)

    simplified_term = simplify_Merge(simplified_term)
    term_changed, simplified_term = simplify_Bilinear_Right(simplified_term)
    return simplify_Merge(simplified_term)
end
