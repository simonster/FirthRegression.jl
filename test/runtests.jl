using FirthRegression, Base.Test, DataFrames

## Firth regression with admit data
df = readtable(Pkg.dir("GLM","data","admit.csv.gz"))
df[:rank] = pool(df[:rank])
gm = fit(FirthGLM, admit ~ gre + gpa + rank, df, Binomial(), convTol=1e-14)
@test_approx_eq_eps coef(gm) [-3.9076341278382402677, 0.0022173644223770295, 0.7875352644980999628, -0.6657797116943170446, -1.3177582553490327921, -1.5111669280800947845] 1e-8
@test_approx_eq penalized_deviance(gm.model) 431.06681398743257

## With sparse matrices
srand(1)
X = sprand(1000, 10, 0.01)
β = randn(10)
y = Bool[rand() < x for x in logistic(X * β)]

gmdense = fit(FirthGLM, full(X), y, Binomial())
gmsparse = fit(FirthGLM, X, y, Binomial())
@test_approx_eq deviance(gmsparse) deviance(gmdense)
@test_approx_eq coef(gmsparse) coef(gmdense)

## ANOVA
anv = anova(fit(FirthGLM, admit ~ gre + gpa + rank, df, Binomial(), convTol=1e-14).model,
            fit(FirthGLM, admit ~ gre + gpa, df, Binomial(), convTol=1e-14).model, convTol=1e-14)
@test_approx_eq anv.chisq 21.373030365595071
@test anv.df == 3
@test_approx_eq anv.p 8.8071001596912168e-05

## ANOVA w/o prefit models
anv = anova(fit(FirthGLM, admit ~ gre + gpa + rank, df, Binomial(), convTol=1e-14, dofit=false).model,
            fit(FirthGLM, admit ~ gre + gpa, df, Binomial(), convTol=1e-14, dofit=false).model, convTol=1e-14)
@test_approx_eq anv.chisq 21.373030365595071
@test anv.df == 3
@test_approx_eq anv.p 8.8071001596912168e-05

