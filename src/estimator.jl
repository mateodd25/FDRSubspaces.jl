using LinearAlgebra
using Integrals
using Statistics

struct SpectralDistribution
    lower_bound::Float64
    upper_bound::Float64
    density::Function
end

abstract type AbstractMatrixEnsemble end

struct WignerEnsemble <: AbstractMatrixEnsemble
    fixed_spectrum::Vector{Float64}
end

struct WishartEnsemble <: AbstractMatrixEnsemble
    fixed_spectrum::Vector{Float64}
    dimension_proportion::Float64
end

struct ResultProcedure
    """ A result is a tuple of the best, the fdr, rank estimate, the threshold, the spacings."""
    best_k::Int
    fdr::Vector{Float64}
    rank_estimate::Int
    threshold::Float64
    spacings::Vector{Float64}
    eigenvalues::Vector{Float64}
end

function cauchy_transform(ensemble::AbstractMatrixEnsemble)::Function
    """ Returns the Cauchy transform of the matrix ensemble."""
    distribution = limiting_spectral_distribution(ensemble)
    return z -> solve(IntegralProblem((x, p) -> distribution.density(x) / (z - x), distribution.lower_bound, distribution.upper_bound), QuadGKJL()).u
end

function cauchy_transform_der(ensemble::AbstractMatrixEnsemble)::Function
    """ Returns the derivative of the Cauchy transform of the matrix ensemble."""
    distribution = limiting_spectral_distribution(ensemble)
    return z -> -solve(IntegralProblem((x, p) -> distribution.density(x) / (z - x)^2, distribution.lower_bound, distribution.upper_bound), QuadGKJL()).u
end

function fixed_spectrum(ensemble::AbstractMatrixEnsemble)::Vector{Float64}
    """ Returns the fixed spectrum of the matrix ensemble."""
    return ensemble.fixed_spectrum
end

function limiting_spectral_distribution(ensemble::AbstractMatrixEnsemble)::SpectralDistribution
    """ Returns the limiting spectral distribution of the matrix ensemble."""
    if isa(ensemble, WignerEnsemble)
        limiting_distribution = SpectralDistribution(-2, 2, x -> sqrt(4 - x^2) / (2 * pi))
    else
        isa(ensemble, WishartEnsemble)
        lb = (1 - sqrt(ensemble.dimension_proportion))^2
        ub = (1 + sqrt(ensemble.dimension_proportion))^2
        limiting_distribution = SpectralDistribution(lb, ub, x -> sqrt((ub - x) * (x - lb)) / (2 * pi * ensemble.dimension_proportion * x))
    end
    return limiting_distribution
end

function generate_noise(ensemble::AbstractMatrixEnsemble, dimension::Int)::Symmetric{Float64,Matrix{Float64}}
    """ Generates a random matrix from the matrix ensemble."""
    if isa(ensemble, WignerEnsemble)
        noise = randn(dimension, dimension) / sqrt(dimension)
        noise = (noise + noise') / 2
    else
        isa(ensemble, WishartEnsemble)
        noise = randn(dimension, Int(ensemble.dimension_proportion * dimension)) / sqrt(dimension)
        noise = noise * noise'
    end
    return Symmetric(noise)
end

function true_and_noisy_matrix(ensemble::AbstractMatrixEnsemble, dimension::Int)::Tuple{Symmetric{Float64,Matrix{Float64}},Symmetric{Float64,Matrix{Float64}}}
    """ Generates a true signal matrix and a noisy matrix."""
    rank = length(ensemble.fixed_spectrum)
    true_signal = Symmetric(diagm([ensemble.fixed_spectrum; zeros(dimension - rank)]))
    noisy_matrix = (true_signal + generate_noise(ensemble, dimension))
    return true_signal, noisy_matrix
end

function estimate_fdr(noisy_matrix::Symmetric{Float64,Matrix{Float64}}, rank_estimate::Int, eigenvalues::Union{Vector{Float64}, Nothing}=nothing)::Vector{Float64}
    """ Estimates the false discovery rate for different thresholds."""

    if isnothing(eigenvalues)
        eigenvalues = sort(eigvals(noisy_matrix), rev=true)
    end
    n = length(eigenvalues)

    function cauchy_transform_estimate(z)
        """Computes the Cauchy transform estimate at z."""
        return sum((z .- eigenvalues[(rank_estimate+1):n]) .^ (-1)) / (n - rank_estimate)
    end
    function cauchy_transform_der_estimate(z)
        """Computes the derivative of the Cauchy transform estimate at z."""
        return -sum((z .- eigenvalues[(rank_estimate+1):n]) .^ (-2)) / (n - rank_estimate)
    end

    costs = ones(n)
    FDR = zeros(n)
    running_sum = 0
    for (i, lambda) in enumerate(eigenvalues)
        if i <= rank_estimate
            costs[i] += (cauchy_transform_estimate(lambda)^2 / cauchy_transform_der_estimate(lambda))
        end
        running_sum += costs[i]
        FDR[i] = running_sum / i
    end
    return FDR
end


function estimate_rank(noisy_matrix::Symmetric{Float64,Matrix{Float64}})::Tuple{Int,Float64,Vector{Float64},Vector{Float64}}
    """ Estimates the observable rank of the signal matrix.

    Returns the rank estimate, the threshold, the spacings, and the eigenvalues.
    """
    eigenvalues = sort(eigvals(noisy_matrix), rev=true)
    n = length(eigenvalues)
    spacings = [eigenvalues[i] - eigenvalues[i+1] for i in 1:(n-1)]
    threshold = Statistics.median(spacings) * n^(1 / 2) * .3
    rank_estimate = maximum(findall(>(threshold), spacings))
    return rank_estimate, threshold, spacings, eigenvalues
end

function best_k(fdr::Vector{Float64}, level::Float64)::Int
    """ Computes the best k for a given FDR level."""
    return minimum(findall(>(level), fdr)) - 1
end

function control_fdr(noisy_matrix::Symmetric{Float64,Matrix{Float64}}, level::Float64, rank_estimate::Union{Int, Nothing}=nothing)::ResultProcedure
    """ Computes the FDR for different thresholds."""
    if isnothing(rank_estimate)
        rank_estimate, threshold, spacings, eigenvalues = estimate_rank(noisy_matrix)
        fdr = estimate_fdr(noisy_matrix, rank_estimate, eigenvalues)
        return ResultProcedure(best_k(fdr, level), fdr, rank_estimate, threshold, spacings, eigenvalues)
    else
        fdr = estimate_fdr(noisy_matrix, rank_estimate)
        return ResultProcedure(best_k(fdr, level), fdr, rank_estimate, -1, [], [])
    end
end

function get_top_eigenvectors(A::Symmetric{Float64,Matrix{Float64}}, k::Int)
    n = size(A)[1]
    ef = eigen(Symmetric(A), (n-k+1):n)   #k top eigenvalues/vectors
    return ef
end

function compute_fdp(U::Matrix{Float64}, Uhat::Matrix{Float64})::Float64
    """ Compute false discovery proportion."""
    k = size(Uhat)[2]
    return tr(Uhat * (Uhat' * (I - U * U'))) / k
end

function estimate_true_fdr(ensemble::AbstractMatrixEnsemble, dimension::Int, upper_bound::Union{Int,Nothing}=nothing)::Vector{Float64}
    """ Estimate the true FDR for a given matrix ensemble in dimension via Monte Carlo."""
    N = 100
    if isnothing(upper_bound)
        upper_bound = dimension
    end
    fdrs = zeros(upper_bound)
    for _ in 1:N
        true_signal, noisy_matrix = true_and_noisy_matrix(ensemble, dimension)
        _, U = get_top_eigenvectors(true_signal, upper_bound)
        _, Uh = get_top_eigenvectors(noisy_matrix, upper_bound)
        for k in 1:upper_bound
            fdrs[k] += compute_fdp(U, Uh[:, (upper_bound-k+1):upper_bound])
        end
    end
    return fdrs / N
end

function compute_limiting_fdr(ensemble::AbstractMatrixEnsemble, dimension::Int, upper_bound::Union{Int,Nothing}=nothing)::Vector{Float64}
    """ Compute the limiting FDR for a given matrix ensemble in dimension."""
    if isnothing(upper_bound)
        upper_bound = dimension
    end
    fdrs = zeros(upper_bound)
    G = cauchy_transform(ensemble)
    Gd = cauchy_transform_der(ensemble)
    dist_upper_bound = limiting_spectral_distribution(ensemble).upper_bound
    dist_lower_bound = limiting_spectral_distribution(ensemble).lower_bound
    running_sum = 0
    for k in 1:upper_bound
        running_sum += 1
        if k <= length(ensemble.fixed_spectrum) && 1 / ensemble.fixed_spectrum[k] < G(dist_upper_bound + 10 * eps()) && 1 / ensemble.fixed_spectrum[k] > G(dist_lower_bound - 10 * eps())
            running_sum += G(ensemble.fixed_spectrum[k])^2 / Gd(ensemble.fixed_spectrum[k])
        end
        fdrs[k] = running_sum / k
    end
    return fdrs
end
