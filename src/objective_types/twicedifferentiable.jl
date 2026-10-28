# Used for objectives and solvers where the gradient and Hessian is available/exists
mutable struct TwiceDifferentiable{T<:Real,TDF<:AbstractArray,TJVP<:Number,TH<:AbstractMatrix,THVP<:AbstractArray,TX<:AbstractArray,F_f,F_df,F_fdf,F_jvp,F_fjvp,F_dfh,F_fdfh,F_h,F_hv} <: AbstractObjective
    const f::F_f
    const df::F_df
    const fdf::F_fdf
    const jvp::F_jvp # Jacobian-vector product of objective
    const fjvp::F_fjvp # objective and Jacobian-vector product of objective
    const dfh::F_dfh
    const fdfh::F_fdfh
    const h::F_h
    const hv::F_hv
    F::T
    const DF::TDF
    JVP::TJVP
    const H::TH
    const HVP::THVP
    const x_f::TX
    const x_df::TX
    const x_jvp::TX
    const v_jvp::TX
    const x_h::TX
    const x_hvp::TX
    const v_hvp::TX
    f_calls::Int
    df_calls::Int
    jvp_calls::Int
    h_calls::Int
    hvp_calls::Int
    TwiceDifferentiable{T,TDF,TJVP,TH,THVP,TX,F_f,F_df,F_fdf,F_jvp,F_fjvp,F_dfh,F_fdfh,F_h,F_hv}(args::Vararg{Any,26}) where {T,TDF,TJVP,TH,THVP,TX,F_f,F_df,F_fdf,F_jvp,F_fjvp,F_dfh,F_fdfh,F_h,F_hv} = new{T,TDF,TJVP,TH,THVP,TX,F_f,F_df,F_fdf,F_jvp,F_fjvp,F_dfh,F_fdfh,F_h,F_hv}(args...)
end

# compatibility with old constructor
Base.@constprop :aggressive function TwiceDifferentiable(f::CF, g::CG, fg::CFG, h::CH, x::TX, F::T = real(eltype(x))(NaN), G::TG = alloc_DF(x, F), H::TH = alloc_H(x, F); inplace::Bool = true, specialize::Bool = false) where {T<:Real, TG<:AbstractArray, TH<:AbstractMatrix, TX<:AbstractArray,CF,CG,CFG,CH}
    _typed_twice_differentiable(Val(inplace),Val(specialize),f,g,fg,h,x,F,G,H)
end

Base.@constprop :aggressive function _typed_twice_differentiable(::Val{inplace},::Val{specialize}, f::CF, g::CG, fg::CFG, h::CH, x::TX, F::T, G::TG, H::TH) where {inplace,specialize,T<:Real,TG<:AbstractArray,TH<:AbstractMatrix,TX<:AbstractArray,CF,CG,CFG,CH}
    g! = df!_from_df(g, F, inplace)
    fg! = fdf!_from_fdf(fg, F, inplace)
    h! = h!_from_h(h, F, inplace)
    dfh! = make_dfh(x, F, g!, h!)
    fdfh! = make_fdfh(x, F, fg!, h!)

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_jvp = x_of_nans(x)
    v_jvp = x_of_nans(x)
    x_h = x_of_nans(x)
    x_hvp = x_of_nans(x)
    v_hvp = x_of_nans(x)

    JVP = alloc_JVP(x, F)

    TwiceDifferentiable{T,TG,typeof(JVP),TH,TG,TX}(f, g!, fg!, nothing, nothing, dfh!, fdfh!, h!, nothing,
                                        copy(F), copy(G), JVP, copy(H), copy(G),
                                        x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp,
                                        0, 0, 0, 0, 0; specialize)
end

Base.@constprop :aggressive function TwiceDifferentiable(f, g, h,
                             x::AbstractArray,
                             F::Real = real(eltype(x))(NaN),
                             G::AbstractArray = alloc_DF(x, F),
                             H::AbstractMatrix = alloc_H(x, F); inplace = true, specialize::Bool = false)
    g! = df!_from_df(g, F, inplace)
    fg! = make_fdf(x, F, f, g!)
    h! = h!_from_h(h, F, inplace)
    dfh! = make_dfh(x, F, g!, h!)
    fdfh! = make_fdfh(x, F, fg!, h!)

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_jvp = x_of_nans(x)
    v_jvp = x_of_nans(x)
    x_h = x_of_nans(x)
    x_hvp = x_of_nans(x)
    v_hvp = x_of_nans(x)

    return TwiceDifferentiable(f, g!, fg!, nothing, nothing, dfh!, fdfh!, h!, nothing, copy(F), copy(G), alloc_JVP(x, F), copy(H), copy(G), x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp, 0, 0, 0, 0, 0; specialize)
end



Base.@constprop :aggressive function TwiceDifferentiable(f, g,
                             x_seed::AbstractArray,
                             F::Real = real(eltype(x_seed))(NaN);
                             inplace::Bool = true, specialize::Bool = false,
                             autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    g! = df!_from_df(g, F, inplace)
    fg! = make_fdf(x_seed, F, f, g!)

    x_prep = x_of_values(x_seed)

    hess_prep = DI.prepare_hessian(f, autodiff, x_prep)
    h! = let f = f, hess_prep = hess_prep, autodiff = autodiff
        function (_h, _x)
            DI.hessian!(f, _h, hess_prep, autodiff, _x)
            return _h
        end
    end
    dfh! = make_dfh(x_seed, F, g!, h!)
    fdfh! = make_fdfh(x_seed, F, fg!, h!)

    hvp_prep = DI.prepare_hvp(f, autodiff, x_prep, (x_prep,))
    hvp! = let f = f, hv_prep = hvp_prep, autodiff = autodiff
        function (_hvp, _x, _v)
            DI.hvp!(f, (_hvp,), hv_prep, autodiff, _x, (_v,))
            return nothing
        end
    end

    # TODO: Define dedicated AD-based functions for JVPs as well
    # Currently, this is disabled as it can lead to inconsistencies with gradient calculations with finite differencing (default)
    # We probably need a more fine-grained way for choosing AD backends, as JVP is a prime candidate for forward-mode AD

    x_f = x_of_nans(x_seed)
    x_df = x_of_nans(x_seed)
    x_jvp = x_of_nans(x_seed)
    v_jvp = x_of_nans(x_seed)
    x_h = x_of_nans(x_seed)
    x_hvp = x_of_nans(x_seed)
    v_hvp = x_of_nans(x_seed)

    return TwiceDifferentiable(f, g!, fg!, nothing, nothing, dfh!, fdfh!, h!, hvp!, F, alloc_DF(x_seed, F), F, alloc_H(x_seed, F), alloc_DF(x_seed, F), x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp, 0, 0, 0, 0, 0; specialize)
end

Base.@constprop :aggressive function TwiceDifferentiable(d::NonDifferentiable,
                             x_seed::AbstractArray = d.x_f,
                             F::Real = real(eltype(x_seed))(NaN);
                             specialize::Bool = false, autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    TwiceDifferentiable(d.f, x_seed, F; autodiff, specialize)
end

Base.@constprop :aggressive function TwiceDifferentiable(d::OnceDifferentiable,
                             x_seed::AbstractArray = d.x_f,
                             F::Real = real(eltype(x_seed))(NaN);
                             specialize::Bool = false, autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    x_prep = x_of_values(x_seed)

    hess_prep = DI.prepare_hessian(d.f, autodiff, x_prep)
    h! = let f = d.f, hess_prep = hess_prep, autodiff = autodiff
        function (_h, _x)
            DI.hessian!(f, _h, hess_prep, autodiff, _x)
            return _h
        end
    end
    dfh! = make_dfh(x_seed, F, d.df, h!)
    fdfh! = make_fdfh(x_seed, F, d.fdf, h!)

    hvp_prep = DI.prepare_hvp(d.f, autodiff, x_prep, (x_prep,))
    hvp! = let f = d.f, hv_prep = hvp_prep, autodiff = autodiff
        function (_hvp, _x, _v)
            DI.hvp!(f, (_hvp,), hv_prep, autodiff, _x, (_v,))
            return nothing
        end
    end

    # TODO: Define dedicated AD-based functions for JVPs as well
    # Currently, this is disabled as it can lead to inconsistencies with gradient calculations with finite differencing (default)
    # We probably need a more fine-grained way for choosing AD backends, as JVP is a prime candidate for forward-mode AD

    x_f = x_of_nans(x_seed)
    x_df = x_of_nans(x_seed)
    x_jvp = x_of_nans(x_seed)
    v_jvp = x_of_nans(x_seed)
    x_h = x_of_nans(x_seed)
    x_hvp = x_of_nans(x_seed)
    v_hvp = x_of_nans(x_seed)

    return TwiceDifferentiable(d.f, d.df, d.fdf, nothing, nothing, dfh!, fdfh!, h!, hvp!, F, alloc_DF(x_seed, F), F, alloc_H(x_seed, F), alloc_DF(x_seed, F), x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp, 0, 0, 0, 0, 0; specialize)
end

Base.@constprop :aggressive function TwiceDifferentiable(f, x::AbstractArray, F::Real = real(eltype(x))(NaN);
                             inplace::Bool = true, specialize::Bool = false,
                             autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    x_prep = x_of_values(x)

    grad_prep = DI.prepare_gradient(f, autodiff, x_prep)
    g! = let f = f, grad_prep = grad_prep, autodiff = autodiff
        function (_g, _x)
            DI.gradient!(f, _g, grad_prep, autodiff, _x)
            return nothing
        end
    end
    fg! = let f = f, grad_prep = grad_prep, autodiff = autodiff
        function (_g, _x)
            y, _ = DI.value_and_gradient!(f, _g, grad_prep, autodiff, _x)
            return y
        end
    end
    hess_prep = DI.prepare_hessian(f, autodiff, x_prep)
    gh! = let f = f, hess_prep = hess_prep, autodiff = autodiff
        function (_g, _h, _x)
            DI.value_gradient_and_hessian!(f, _g, _h, hess_prep, autodiff, _x)
            return nothing
        end
    end
    fgh! = let f = f, hess_prep = hess_prep, autodiff = autodiff
        function (_g, _h, _x)
            y, _, _ = DI.value_gradient_and_hessian!(f, _g, _h, hess_prep, autodiff, _x)
            return y
        end
    end
    h! = let f = f, hess_prep = hess_prep, autodiff = autodiff
        function (_h, _x)
            DI.hessian!(f, _h, hess_prep, autodiff, _x)
            return nothing
        end
    end
    hvp_prep = DI.prepare_hvp(f, autodiff, x_prep, (x_prep,))
    hvp! = let f = f, hv_prep = hvp_prep, autodiff = autodiff
        function (_hvp, _x, _v)
            DI.hvp!(f, (_hvp,), hv_prep, autodiff, _x, (_v,))
            return nothing
        end
    end

    # TODO: Define dedicated AD-based functions for JVPs as well
    # Currently, this is disabled as it can lead to inconsistencies with gradient calculations with finite differencing (default)
    # We probably need a more fine-grained way for choosing AD backends, as JVP is a prime candidate for forward-mode AD

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_jvp = x_of_nans(x)
    v_jvp = x_of_nans(x)
    x_h = x_of_nans(x)
    x_hvp = x_of_nans(x)
    v_hvp = x_of_nans(x)

    TwiceDifferentiable(f, g!, fg!, nothing, nothing, gh!, fgh!, h!, hvp!, F, alloc_DF(x, F), F, alloc_H(x, F), alloc_DF(x, F), x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp, 0, 0, 0, 0, 0; specialize)
end

# Callable specialization is opt-in to preserve today's interactive latency and
# Any-valued callable layout. `specialize=true` is useful for static compilation.
Base.@constprop :aggressive function TwiceDifferentiable(
        f::C0, df::C1, fdf::C2, jvp::C3, fjvp::C4, dfh::C5, fdfh::C6, h::C7, hv::C8,
        F::T, DF::TDF, JVP::TJVP, H::TH, HVP::THVP, x_f::TX, x_df::TX, x_jvp::TX, v_jvp::TX, x_h::TX, x_hvp::TX, v_hvp::TX, f_calls::Int, df_calls::Int, jvp_calls::Int, h_calls::Int, hvp_calls::Int; specialize::Bool = false) where {T<:Real, TDF<:AbstractArray, TJVP<:Number, TH<:AbstractMatrix, THVP<:AbstractArray, TX<:AbstractArray, C0, C1, C2, C3, C4, C5, C6, C7, C8}
    _construct_twicedifferentiable(Val(specialize), T, TDF, TJVP, TH, THVP, TX, f, df, fdf, jvp, fjvp, dfh, fdfh, h, hv, F, DF, JVP, H, HVP, x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp, f_calls, df_calls, jvp_calls, h_calls, hvp_calls)
end

function _construct_twicedifferentiable(::Val{S}, ::Type{T}, ::Type{TDF}, ::Type{TJVP}, ::Type{TH}, ::Type{THVP}, ::Type{TX},
        f::C0, df::C1, fdf::C2, jvp::C3, fjvp::C4, dfh::C5, fdfh::C6, h::C7, hv::C8, F, DF, JVP, H, HVP, x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp, f_calls, df_calls, jvp_calls, h_calls, hvp_calls) where {S, T, TDF, TJVP, TH, THVP, TX, C0, C1, C2, C3, C4, C5, C6, C7, C8}
    TwiceDifferentiable{T, TDF, TJVP, TH, THVP, TX, S ? C0 : Any, S ? C1 : Any, S ? C2 : Any, S ? C3 : Any, S ? C4 : Any, S ? C5 : Any, S ? C6 : Any, S ? C7 : Any, S ? C8 : Any}(
        f, df, fdf, jvp, fjvp, dfh, fdfh, h, hv, F, DF, JVP, H, HVP, x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp, f_calls, df_calls, jvp_calls, h_calls, hvp_calls)
end

# Retain the original partially parameterized all-fields construction API.
Base.@constprop :aggressive function TwiceDifferentiable{T, TDF, TJVP, TH, THVP, TX}(
        f::C0, df::C1, fdf::C2, jvp::C3, fjvp::C4, dfh::C5, fdfh::C6, h::C7, hv::C8, F, DF, JVP, H, HVP, x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp, f_calls, df_calls, jvp_calls, h_calls, hvp_calls; specialize::Bool = false) where {T, TDF, TJVP, TH, THVP, TX, C0, C1, C2, C3, C4, C5, C6, C7, C8}
    _construct_twicedifferentiable(Val(specialize), T, TDF, TJVP, TH, THVP, TX, f, df, fdf, jvp, fjvp, dfh, fdfh, h, hv, F, DF, JVP, H, HVP, x_f, x_df, x_jvp, v_jvp, x_h, x_hvp, v_hvp, f_calls, df_calls, jvp_calls, h_calls, hvp_calls)
end
