module FirthRegression
using Reexport, Compat
@reexport using GLM
using GLM: wrkwt!, installbeta!
export FirthGLM, penalized_deviance

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

# TODO make this more efficient
hatdiag(p::GLM.DensePredChol, wt::Vector) = sum(p.X.*(p.chol\p.scratch')', 2)

function adjustwrkresid!(m::FirthGLM, wt::Vector)
    r = m.rr
    link = r.l
    eta = r.eta
    mueta = r.mueta
    var = r.var
    wrkresid = r.wrkresid
    h = hatdiag(m.pp, wt)
    @inbounds @simd for i = 1:length(eta)
        # See Kosmidis and Firth, 2009, 4⋅3
        wrkresid[i] += h[i]*mueta2(link, eta[i])/(2*mueta[i]^3/var[i])
    end
    r
end

function GLM._fit(m::FirthGLM, verbose::Bool, maxIter::Integer, minStepFac::Real,
                  convTol::Real, start)
    m.fit && return m
    maxIter >= 1 || error("maxIter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

    cvg = false; p = m.pp; r = m.rr
    lp = r.mu
    if start != nothing
        copy!(p.beta0, start)
        fill!(p.delbeta, 0)
    else
        delbeta!(p, wrkresp(r), wrkwt!(r))
        installbeta!(p)
    end
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

penalized_deviance(m::FirthGLM) = deviance(m.rr) - logdet(m.pp.chol)

end # module
