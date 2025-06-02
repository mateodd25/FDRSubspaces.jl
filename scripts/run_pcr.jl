using LinearAlgebra
using Statistics
using Random
using Distributions
using Plots
using FDRControlSubspaceSelection

# -----------------------------
# Synthetic‑data generator
# -----------------------------
function generate_dataset(; n_train::Int, n_test::Int, p::Int,
                           Σz_diag::Vector{Float64},
    σ_η::Float64, σ_ϵ::Float64, range::UnitRange{Int})
    """
    generate_dataset(; n_train, n_test, p, Σz_diag, σ_η, σ_ϵ)

    Generate one training / test split according to the hierarchical model

    x_i = U z_i + η_i,     z_i ∼ 𝒩(0, Σ_z),  η_i ∼ 𝒩(0, σ_η² I_p)
    y_i = x_i' β + ϵ_i,    ϵ_i ∼ 𝒩(0, σ_ϵ²)

    `U` (p×k) has orthonormal columns; β lives in the span of U.
    Returns a NamedTuple with X_train, y_train, X_test, y_test, and the
    true β (so one can compute oracle quantities if desired).
    """
    k = length(Σz_diag)                 # intrinsic rank of the signal

    # Orthonormal basis for the signal subspace (p × k)
    U = qr(randn(p, k)).Q[:, 1:k]

    # Sample latent scores for all observations
    Z_train = randn(n_train, k) * Diagonal(sqrt.(Σz_diag))
    Z_test  = randn(n_test,  k) * Diagonal(sqrt.(Σz_diag))

    # Noise for the covariates
    E_train = σ_η * randn(n_train, p)
    E_test  = σ_η * randn(n_test,  p)

    # Covariate matrices
    X_train_noiseless = Z_train * U'  # (n_train × p)
    X_test_noiseless = Z_test * U'  # (n_test × p)
    X_train = X_train_noiseless + E_train   # (n_train × p)
    X_test = X_test_noiseless + E_test

    # True coefficient vector β (in the span of U)
    β_latent = randn(k)
    β = U * β_latent

    # Responses
    y_train = X_train_noiseless * β .+ σ_ϵ * randn(n_train)
    y_test = X_test_noiseless * β .+ σ_ϵ * randn(n_test)

    return (X_train = X_train,
        X_train_noiseless=X_train_noiseless,
            y_train = y_train,
            X_test  = X_test,
            y_test  = y_test,
            β = β)
end

# -----------------------------
# Rank‑selection rules
# -----------------------------
# Helper: Marčenko–Pastur edge for noise singular values
mp_edge(σ_η2::Float64, n::Int, p::Int) = σ_η2 * (1 + sqrt(p / n))^2

"""select_rank_FDR(X, σ_η) -> k
Keep all singular values of X whose squared values exceed the MP edge.
This is a (liberal) FDR‑style rule: anything that sticks out of the
bulk is declared "signal".
"""
function select_rank_FDR(X::AbstractMatrix, σ_η::Float64)
    n, p = size(X)
    edge = mp_edge(σ_η^2, n, p)
    λ2 = svdvals(X).^2 ./ (n - 1)           # empirical covariance eigenvalues
    return count(>(edge), λ2)               # number above the edge
end

"""select_rank_MSE(X, σ_η; factor = 2.0) -> k
Conservative MSE rule: require singular‑value energy to exceed
`factor × edge` before keeping a component.
"""
function select_rank_MSE(X::AbstractMatrix, σ_η::Float64; factor::Float64 = 2.0)
    n, p = size(X)
    edge = factor * mp_edge(σ_η^2, n, p)
    λ2 = svdvals(X).^2 ./ (n - 1)
    return count(>(edge), λ2)
end

# -----------------------------
# PCR estimator
# -----------------------------
function pcr_beta(X::AbstractMatrix, y::AbstractVector, k::Int)
    """pcr_beta(X, y, k) -> β̂
    Compute the PCR estimator using the first k left‑singular vectors.
    """
    if k == 0                                # pathological but possible
        return zeros(size(X, 2))
    end
    U, S, Vt = svd(X, full = false)
    V_k = Vt'[ :, 1:k]                       # p × k loading matrix
    Z = X * V_k                              # scores (n × k)
    w = Z \ y                                # least‑squares in score space
    return V_k * w                           # back to p‑dimensional β̂
end

# -----------------------------
# Single replicate of the experiment
# -----------------------------
function run_once(params)
    data = generate_dataset(; params...)
    Xtr, ytr = data.X_train, data.y_train
    Xte, yte = data.X_test,  data.y_test
    σ_η = params.σ_η
    errors = Float64[]
    # fit PCR models
    for k in params.range
        β̂ = pcr_beta(Xtr, ytr, k)
        push!(errors, mean((Xte * β̂ .- yte).^2))
    end

    return (; errors)
end

# -----------------------------
# Batched simulation
# -----------------------------
function run_experiment(; N = 100, rng = MersenneTwister(2025))
    max_k = 50
    params = (
        n_train=300,
        n_test=2000,
        p=200,
        Σz_diag=vcat(fill(12.0, 10), fill(10.0, 20)),  # 5 strong + 15 weak
        σ_η=0.01,
        σ_ϵ=0.01,
        range=1:max_k,
    )

    errors = zeros(max_k, N)

    for j in 1:N
        errors[:, j] = run_once(params).errors
    end
    avg_errors = mean(errors, dims=2)


    return avg_errors
end

            # -----------------------------
# Main entry point
# -----------------------------
function main()
    println("Running PCR FDR vs MSE experiment …")
    stats = run_experiment()
    println(stats)

    # println("\nAverage selected rank (FDR): ", mean(stats.ks_fdr))
    # println("Average selected rank (MSE): ", mean(stats.ks_mse))
    # println("Average test MSE  (FDR): ",  mean(stats.mses_fdr))
    # println("Average test MSE  (MSE): ",  mean(stats.mses_mse))

    # Figure
    p = plot(stats, label="Average MSE", xlabel="Rank",
        ylabel="Test MSE")
    # scatter!(p, stats.mses_fdr, label = "PCR‑FDR", marker = (:circle, 6))
    savefig(p, "pcr_fdr_vs_mse.pdf")
    println("Figure saved to pcr_fdr_vs_mse.png")
end

# run if this file is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


# import FDRControlSubspaceSelection
# using Plots
# using Random
# using LaTeXStrings

# fntsm = font("serif-roman", pointsize = 18)
# fntlg = font("serif-roman", pointsize = 18)
# default(
#     titlefont = fntlg,
#     guidefont = fntlg,
#     tickfont = fntsm,
#     legendfont = fntsm,
# )

# Random.seed!(1)
# d = 1000
# ranks = [20, 40]
# upper_bound_rank = 80
# fds = []
# fdrs = []
# rank = 0


# function main()
#     println("Running PCR experiment")

# end

# main()
