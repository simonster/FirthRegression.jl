module FirthRegression
using Reexport, Compat
@reexport using GLM
using GLM: wrkwt!, installbeta!
export FirthGLM, penalized_deviance, anova

mueta2(::LogitLink, η) = (e = exp(η); (e*(1-e))/(1+e)^3)
mueta2(::LogLink, η) = exp(η)
# Need to figure out how to compute the deviance for non-canonical links if we want to use them
# mueta2(::CauchitLink, η) = -2*η/(pi*(1+η^2)^2)
# mueta2(::ProbitLink, η) = -exp(-η^2/2)*η/sqrt2π
# mueta2(::InverseLink, η) = 2/η^3

@compat typealias FirthGlmResp{T,D<:Union{Binomial,Poisson},L} GlmResp{T,D,L}

type FirthGLM{G<:FirthGlmResp,L<:LinPred} <: GLM.AbstractGLM
    rr::G
    pp::L
    fit::Bool
end

type FirthGLMTest{G<:FirthGlmResp,L<:LinPred} <: GLM.AbstractGLM
    rr::G
    pp::L
    pp_full::L
    fit::Bool
end

function updatefact!{T}(p::GLM.DensePredChol{T}, wt::Vector{T})
    scr = scale!(p.scratch, wt, p.X)
    cholfact!(At_mul_B!(GLM.cholfactors(p.chol), scr, p.X), :U)
end

function updatefact!{T}(p::GLM.SparsePredChol{T}, wt::Vector{T})
    scr = scale!(p.scratch, wt, p.X)
    XtX = p.Xt*scr
    c = p.chol = $(if VERSION >= v"0.4.0-dev+3307"
        :(cholfact(Symmetric{eltype(XtX),typeof(XtX)}(XtX, 'U')))
    else
        :(cholfact(scale!(XtX + XtX', convert(eltype(XtX), 1/2))))
    end)
end

# TODO make this more efficient
hatdiag(p::GLM.DensePredChol, wt::Vector) = sum(p.X.*(p.chol\p.scratch')', 2)
hatdiag(p::GLM.SparsePredChol, wt::Vector) = sum(p.X.*(p.chol\p.Xt)', 2).*wt

hatdiag(m::FirthGLM, wt::Vector) = hatdiag(m.pp, wt)
hatdiag(m::FirthGLMTest, wt::Vector) = hatdiag(m.pp_full, wt)

function adjustwrkresid!(m::@compat(Union{FirthGLM,FirthGLMTest}), wt::Vector)
    r = m.rr
    link = r.l
    eta = r.eta
    mueta = r.mueta
    var = r.var
    wrkresid = r.wrkresid
    h = hatdiag(m, wt)
    @inbounds @simd for i = 1:length(eta)
        # See Kosmidis and Firth, 2009, 4⋅3
        wrkresid[i] += h[i]*mueta2(link, eta[i])/(2*mueta[i]^3/var[i])
    end
    r
end

penalized_deviance(m::FirthGLM) = deviance(m.rr) - logdet(m.pp.chol)
function penalized_deviance(m::FirthGLMTest)
    updatefact!(m.pp_full, m.rr.wrkwts)
    deviance(m.rr) - logdet(m.pp_full.chol)
end

function GLM._fit!(m::@compat(Union{FirthGLM,FirthGLMTest}), verbose::Bool, maxIter::Integer, minStepFac::Real,
                  convTol::Real, start)
    m.fit && return m
    maxIter >= 1 || error("maxIter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

    cvg = false; p = m.pp; r = m.rr
    lp = r.mu
    if start != nothing
        copy!(p.beta0, start)
        fill!(p.delbeta, 0)
        linpred!(lp, p)
        updatemu!(r, lp)
    end
    delbeta!(p, wrkresp(r), wrkwt!(r))
    if isa(m, FirthGLMTest)
        updatefact!(m.pp_full, m.rr.wrkwts)
    end
    installbeta!(p)
    oldwrkwts = similar(r.wrkwts)
    devold = Inf
    for i=1:maxIter
        f = 1.0
        local dev
        copy!(oldwrkwts, r.wrkwts)
        try
            linpred!(lp, p)
            updatemu!(r, lp)
            adjustwrkresid!(m, oldwrkwts)
            delbeta!(p, r.wrkresid, wrkwt!(r))
            dev = penalized_deviance(m)
        catch e
            isa(e, DomainError) ? (dev = Inf) : rethrow(e)
        end
        while dev > devold
            f /= 2.; f > minStepFac || error("step-halving failed at beta0 = $(p.beta0)")
            try
                updatemu!(r, linpred(p, f))
                adjustwrkresid!(m, oldwrkwts)
                delbeta!(p, r.wrkresid, wrkwt!(r))
                dev = penalized_deviance(m)
            catch e
                isa(e, DomainError) ? (dev = Inf) : rethrow(e)
            end
        end
        installbeta!(p, f)
        crit = (devold - dev)/dev
        verbose && println("$i: $dev, $crit")
        if crit < convTol; cvg = true; break end
        devold = dev
    end
    cvg || error("failure to converge in $maxIter iterations")
    m.fit = true
    m
end

function adjustwrkresid!(m::FirthGLMTest, wt::Vector)
    r = m.rr
    link = r.l
    eta = r.eta
    mueta = r.mueta
    var = r.var
    wrkresid = r.wrkresid
    h = hatdiag(m.pp_full, wt)
    @inbounds @simd for i = 1:length(eta)
        # See Kosmidis and Firth, 2009, 4⋅3
        wrkresid[i] += h[i]*mueta2(link, eta[i])/(2*mueta[i]^3/var[i])
    end
    r
end

cols(X::AbstractMatrix) = [sub(X, :, i) for i = 1:size(X, 2)]
cols(X::SparseMatrixCSC) = [X[:, i] for i = 1:size(X, 2)]

immutable ANOVAResult
    chisq::Float64
    df::Int
    p::Float64
end

"""
Performs analysis of deviance for two nested design matrices with
a Firth penalized approach. The penalty for both models is determined
based on the Fisher information matrix for the larger model.
"""
function anova(m1::FirthGLM, m2::FirthGLM; fitargs...)
    m1.rr.y == m2.rr.y || throw(ArgumentError("model responses (y) do not match"))
    (m1.rr.d == m2.rr.d && m1.rr.l == m2.rr.l) ||
        throw(ArgumentError("model distributions or link functions do not match"))
    m1.rr.wts == m2.rr.wts || throw(ArgumentError("model weights do not match"))
    m1.rr.offset == m2.rr.offset || throw(ArgumentError("model offsets do not match"))

    if size(m1.pp.X, 2) == size(m2.pp.X, 2)
        throw(ArgumentError("one design matrix must be nested inside the other"))
    elseif size(m1.pp.X, 2) > size(m2.pp.X, 2)
        full, reduced = m1, m2
    else
        full, reduced = m2, m1
    end

    # Figure out which columns are nested
    fullcols = cols(full.pp.X)
    reduced_colidx = findin(fullcols, cols(reduced.pp.X))
    if length(reduced_colidx) != size(reduced.pp.X, 2)
        throw(ArgumentError("one design matrix must be nested inside the other"))
    end

    # Fit the full model if necessary
    if !full.fit
        fit!(full; fitargs...)
    end
    fulldev = penalized_deviance(full)

    # Fit the reduced model
    if !reduced.fit
        # If the second model has not been fit, we can reuse its
        # response and linear predictor
        rr = reduced.rr
        pp = reduced.pp
        startcoef = coef(full)[reduced_colidx]
    else
        rr = typeof(reduced.rr)(reduced.rr.y, reduced.rr.d, reduced.rr.l,
                                copy(reduced.rr.eta), similar(reduced.rr.y),
                                reduced.rr.offset, reduced.rr.wts)
        pp = GLM.cholpred(reduced.pp.X)
        startcoef = coef(reduced)
    end
    reducedmodel = FirthGLMTest(rr, pp, GLM.cholpred(full.pp.X), false)
    fit!(reducedmodel; start=startcoef, fitargs...)
    reduceddev = penalized_deviance(reducedmodel)

    chisq = reduceddev - fulldev
    df = size(full.pp.X, 2) - length(reduced_colidx)
    ANOVAResult(chisq, df, ccdf(Chisq(df), chisq))
end

end # module
