using LinearAlgebra

struct SpectralDistribution
    lower_bound::Float64
    upper_bound::Float64
    density::Function
end

abstract type AbstractMatrixEnsemble end

struct WignerEnsemble <: AbstractMatrixEnsemble
    """ A matrix ensemble is a distribution on symmetric matrices."""
    fixed_spectrum::Vector{Float64}
end

struct WishartEnsemble <: AbstractMatrixEnsemble
    """ A matrix ensemble is a distribution on symmetric matrices."""
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
end

function fixed_spectrum(ensemble::AbstractMatrixEnsemble)
    """ Returns the fixed spectrum of the matrix ensemble."""
    return ensemble.fixed_spectrum
end

function limiting_spectral_distribution(ensemble::AbstractMatrixEnsemble, dimension::Int)
    """ Returns the limiting spectral distribution of the matrix ensemble."""
    if isa(ensemble, WignerEnsemble)
        limiting_distribution = SpectralDistribution(-2, 2, x -> sqrt(4 - x^2) / (2 * pi))
    else isa(ensemble, WishartEnsemble)
        lb = (1 - sqrt(ensemble.dimension_proportion))^2
        ub = (1 + sqrt(ensemble.dimension_proportion))^2
        limiting_distribution = SpectralDistribution(lb, ub, x -> sqrt((ub - x) * (x - lb)) / (2 * pi * ensemble.dimension_proportion * x))
    end
    return limiting_distribution
end

function generate_noise(ensemble::AbstractMatrixEnsemble, dimension::Int)
    """ Generates a random matrix from the matrix ensemble."""
    if isa(ensemble, WignerEnsemble)
        noise = randn(dimension, dimension) / sqrt(dimension)
        noise = (noise + noise') / 2
    else isa(ensemble, WishartEnsemble)
        noise = randn(dimension, int(ensemble.dimension_proportion * dimension))/sqrt(dimension)
        noise = noise * noise'
    end
    return noise
end

function true_and_noisy_matrix(ensemble::MatrixEnsemble, dimension::Int)
    """ Generates a true signal matrix and a noisy matrix."""
    rank = length(ensemble.fixed_spectrum)
    true_signal = diagn([ensemble.fixed_spectrum; zeros(dimension - rank)])
    noisy_matrix = (true_signal + generate_noise(ensemble, dimension))
    return true_signal, noisy_matrix
end

function estimate_fdr(noisy_matrix::Symmetric{Float64,Matrix{Float64}}, rank_estimate::Int, eigenvalues::Vector{Float64})
    """ Estimates the false discovery rate for different thresholds."""

    if isnothing(eigenvalues)
        eigenvalues = eigvals(noisy_matrix)
    end
    n = length(eigenvalues)

    function cauchy_transform_estimate(z)
        """Computes the Cauchy transform estimate at z."""
        return sum((z .- eigenvalues[1:(n-rank_estimate)]) .^ (-1)) / (n - rank_estimate)
    end
    function cauchy_transform_der_estimate(z)
        """Computes the derivative of the Cauchy transform estimate at z."""
        return -sum((z .- eigenvalues[1:(n-rank_estimate)]) .^ (-2)) / (n - rank_estimate)
    end

    costs = ones(n)
    FDR = zeros(n)
    running_sum = 0
    for (i, lambda) in enumerate(eigenvalues)
        if i <= rank_estimate + 1
            costs[i] += (cauchy_transform_estimate(lambda)^2 / cauchy_transform_der_estimate(lambda))
        end
        running_sum += costs[i]
        FDR[i] = running_sum / i
    end
    return FDR
end


function estimate_rank(noisy_matrix::Symmetric{Float64,Matrix{Float64}})
    """ Estimates the observable rank of the signal matrix."""
    eigenvalues = sort(eigvals(noisy_matrix), rev=true)
    n = length(eigenvalues)
    spacings = [eigenvalues[i] - eigenvalues[i+1] for i in 1:(n-1)]
    threshold = (eigenvalues[1] - eigenvalues[n]) * n^(-1 / 2)
    rank_estimate = maximum(findall(>(threshold), spacings))
    return rank_estimate, threshold, spacings, eigenvalues
end

function best_k(fdr::Vector{Float64}, level::Float64)
    """ Computes the best k for a given FDR level."""
    return minimum(findall(>(level), fdr)) - 1
end

function control_fdr(noisy_matrix::Symmetric{Float64, Matrix{Float64}}, level::Float64)
    """ Computes the FDR for different thresholds."""
    rank_estimate, threshold, spacings, eigenvalues = estimate_rank(noisy_matrix)
    fdr = estimate_fdr(noisy_matrix, rank_estimate, eigenvalues)
    return ResultProcedure(best_k(fdr, level), fdr, rank_estimate, threshold, spacings)
end

function get_top_eigenvectors(A, k)
    n = size(A)[1]
    ef = eigen(Symmetric(A), (n-k+1):n)   #k top eigenvalues/vectors
    return ef
end

function compute_fdp(U, Uhat)
    """ Compute false discovery proportion."""
    n = size(U)[1]
    k = size(Uhat)[2]
    return tr(Uhat*(Uhat'*(I - U*U')))/k
end

function estimate_true_fdr(ensemble::MatrixEnsemble, dimension::Int, upper_bound::Union{Int, nothing} = nothing):
    """ Estimate the true FDR for a given matrix ensemble in dimension via Monte Carlo."""
    N = 100
    if isnothing(upper_bound)
        upper_bound = dimension
    end
    fdrs = zeros(upper_bound)
    for i in 1:N
        true_signal, noisy_matrix = true_and_noisy_matrix(ensemble, dimension)
        _, U = get_top_eigenvectors(true_signal, upper_bound)
        _, Uh = get_top_eigenvectors(noisy_matrix, upper_bound)
        for k in 1:upper_bound
            fdrs[k] += compute_fdp(U, Uh[:, (upper_bound-k+1):upper_bound])
        end
    end
    return fdrs / N
end

function compute_limiting_fdr(ensemble::MatrixEnsemble, dimension::Int, upper_bound::Union{Int, nothing} = nothing)
    """ Compute the limiting FDR for a given matrix ensemble in dimension."""
    if isnothing(upper_bound)
        upper_bound = dimension
    end
    fdrs = zeros(upper_bound)
    for k in 1:upper_bound
        fdrs[k] = ensemble.limiting_fdr(dimension, k)
    end
    return fdrs
end
