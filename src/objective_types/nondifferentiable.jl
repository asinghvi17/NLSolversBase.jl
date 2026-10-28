# Used for objectives and solvers where no gradient is available/exists
mutable struct NonDifferentiable{TF<:Union{AbstractArray,Real},TX<:AbstractArray,F_f} <: AbstractObjective
    const f::F_f
    F::TF
    const x_f::TX
    f_calls::Int
    NonDifferentiable{TF,TX,F_f}(args::Vararg{Any,4}) where {TF,TX,F_f} = new{TF,TX,F_f}(args...)
end

# These could be the same if we could restrict g below not to be an AbstractArray
Base.@constprop :aggressive function NonDifferentiable(f, x::AbstractArray, F::Real = real(eltype(x))(NaN); inplace::Bool = true, specialize::Bool = false)
    xnans = x_of_nans(x)
    NonDifferentiable{typeof(F),typeof(xnans)}(f, F, xnans, 0; specialize)
end
Base.@constprop :aggressive function NonDifferentiable(f, x::AbstractArray, F::AbstractArray; inplace::Bool = true, specialize::Bool = false)
    f = inplace ? f : f!_from_f(f, F, inplace)
    xnans = x_of_nans(x)
    NonDifferentiable{typeof(F),typeof(xnans)}(f, F, xnans, 0; specialize)
end

# this is the g referred to above!
Base.@constprop :aggressive NonDifferentiable(f, g,        x::AbstractArray, F::Union{AbstractArray, Real} = real(eltype(x))(NaN); specialize::Bool = false) = NonDifferentiable(f, x, F; specialize)
Base.@constprop :aggressive NonDifferentiable(f, g, h,     x::TX, F; specialize::Bool = false) where TX  = NonDifferentiable(f, x, F; specialize)
Base.@constprop :aggressive NonDifferentiable(f, g, fg, h, x::TX, F; specialize::Bool = false) where TX  = NonDifferentiable(f, x, F; specialize)

# Callable specialization is opt-in to preserve today's interactive latency and
# Any-valued callable layout. `specialize=true` is useful for static compilation.
Base.@constprop :aggressive function NonDifferentiable(
        f::C0,
        F::TF, x_f::TX, f_calls::Int; specialize::Bool = false) where {TF<:Union{AbstractArray,Real}, TX<:AbstractArray, C0}
    _construct_nondifferentiable(Val(specialize), TF, TX, f, F, x_f, f_calls)
end

function _construct_nondifferentiable(::Val{S}, ::Type{TF}, ::Type{TX},
        f::C0, F, x_f, f_calls) where {S, TF, TX, C0}
    NonDifferentiable{TF, TX, S ? C0 : Any}(
        f, F, x_f, f_calls)
end

# Retain the original partially parameterized all-fields construction API.
Base.@constprop :aggressive function NonDifferentiable{TF, TX}(
        f::C0, F, x_f, f_calls; specialize::Bool = false) where {TF, TX, C0}
    _construct_nondifferentiable(Val(specialize), TF, TX, f, F, x_f, f_calls)
end
