using FirthRegression, Base.Test, DataFrames

## Firth regression with admit data
Base.disable_threaded_libs()
df = readtable(Pkg.dir("GLM","data","admit.csv.gz"))
df[:rank] = pool(df[:rank])
gm = fit(FirthGLM, admit ~ gre + gpa + rank, df, Binomial(), convTol=1e-14)
@test_approx_eq_eps coef(gm) [-3.9076341278382402677, 0.0022173644223770295, 0.7875352644980999628, -0.6657797116943170446, -1.3177582553490327921, -1.5111669280800947845] 1e-8
@test_approx_eq penalized_deviance(gm.model) 431.06681398743257
