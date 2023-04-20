module ForwardDiffWorkarounds

import StaticArrays
import ForwardDiff

# TODO ForwardDiff doesn't provide an API to get jacobian without setting Dual number 'tag'
@inline function vector_mode_jacobian_notag(f, x::StaticArrays.StaticArray)
    # T = typeof(Tag(f, eltype(x)))
    T = Nothing
    return _extract_jacobian(T, _static_dual_eval(T, f, x), x)
end

# Workarounds to find extract_jacobian and static_dual_eval in 
# different ForwardDiff and Julia versions
@static if isdefined(ForwardDiff, :extract_jacobian) # ForwardDiff v < 0.10.35
    const _forwarddiff_static_module = ForwardDiff
    const _extract_jacobian = _forwarddiff_static_module.extract_jacobian
    const _static_dual_eval = _forwarddiff_static_module.static_dual_eval
elseif isdefined(ForwardDiff, :ForwardDiffStaticArraysExt) # ForwardDiff >= 0.10.35, Julia < 1.9
    const _forwarddiff_static_module = ForwardDiff.ForwardDiffStaticArraysExt
    const _extract_jacobian = _forwarddiff_static_module.extract_jacobian
    const _static_dual_eval = _forwarddiff_static_module.static_dual_eval
else # ForwardDiff >= 0.10.35, Julia >= 1.9    
    const _forwarddiff_static_module = Base.get_extension(ForwardDiff, :ForwardDiffStaticArraysExt)
    # get_extension fails with Julia 1.9.0-rc2 ??
    if isnothing(_forwarddiff_static_module)
        @warn "isnothing(Base.get_extension(ForwardDiff, :ForwardDiffStaticArraysExt)) - using workaround, reimplementing ForwardDiffStaticArraysExt extract_jacobian etc"

        @generated function _extract_jacobian(::Type{T}, ydual::StaticArrays.StaticArray, x::S) where {T,S<:StaticArrays.StaticArray}
            M, N = length(ydual), length(x)
            result = Expr(:tuple, [:(ForwardDiff.partials(T, ydual[$i], $j)) for i in 1:M, j in 1:N]...)
            return quote
                $(Expr(:meta, :inline))
                V = StaticArrays.similar_type(S, ForwardDiff.valtype(eltype($ydual)), StaticArrays.Size($M, $N))
                return V($result)
            end
        end

        @generated function _dualize(::Type{T}, x::StaticArrays.StaticArray) where T
            N = length(x)
            dx = Expr(:tuple, [:(ForwardDiff.Dual{T}(x[$i], chunk, Val{$i}())) for i in 1:N]...)
            V = StaticArrays.similar_type(x, ForwardDiff.Dual{T,eltype(x),N})
            return quote
                chunk = ForwardDiff.Chunk{$N}()
                $(Expr(:meta, :inline))
                return $V($(dx))
            end
        end

        @inline _static_dual_eval(::Type{T}, f, x::StaticArrays.StaticArray) where T = f(_dualize(T, x))
    else
        const _extract_jacobian = _forwarddiff_static_module.extract_jacobian
        const _static_dual_eval = _forwarddiff_static_module.static_dual_eval
    end
end


end