using LinearAlgebra
using Integrals
using Statistics

struct SpectralDistribution
    lower_bound::Float64
    upper_bound::Float64
    density::Function
end

abstract type AbstractMatrixEnsemble end

abstract type AbstractAsymMatrixEnsemble end

struct WignerEnsemble <: AbstractMatrixEnsemble
    fixed_spectrum::Vector{Float64}
end

struct WishartEnsemble <: AbstractMatrixEnsemble
    fixed_spectrum::Vector{Float64}
    dimension_proportion::Float64
end

struct WishartFactorEnsemble <: AbstractAsymMatrixEnsemble
    fixed_spectrum::Vector{Float64}
    dimension_proportion::Float64
end

struct FisherEnsemble <: AbstractMatrixEnsemble
    fixed_spectrum::Vector{Float64}
end

struct FisherFactorEnsemble <: AbstractAsymMatrixEnsemble
    fixed_spectrum::Vector{Float64}
    dimension_proportion::Float64
end

struct UniformEnsemble <: AbstractMatrixEnsemble
    fixed_spectrum::Vector{Float64}
end

struct UniformFactorEnsemble <: AbstractAsymMatrixEnsemble
    fixed_spectrum::Vector{Float64}
    dimension_proportion::Float64
end

struct FDRResult
    """ A result is a tuple of the best, the fdr, rank estimate, the threshold, the spacings."""
    best_k::Int
    fdr::Vector{Float64}
    rank_estimate::Int
    threshold::Float64 # spacings threshold
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

function limiting_spectral_distribution(ensemble::Union{AbstractMatrixEnsemble,AbstractAsymMatrixEnsemble})::SpectralDistribution
    """ Returns the limiting spectral distribution of the matrix ensemble."""
    if isa(ensemble, WignerEnsemble)
        limiting_distribution = SpectralDistribution(-2, 2, x -> sqrt(4 - x^2) / (2 * pi))
    elseif  isa(ensemble, WishartEnsemble)
        lb = (1 - sqrt(ensemble.dimension_proportion))^2
        ub = (1 + sqrt(ensemble.dimension_proportion))^2
        limiting_distribution = SpectralDistribution(lb, ub, x -> sqrt((ub - x) * (x - lb)) / (2 * pi * ensemble.dimension_proportion * x))
    elseif isa(ensemble, WishartFactorEnsemble)
        lb = (1 - sqrt(ensemble.dimension_proportion))^2
        ub = (1 + sqrt(ensemble.dimension_proportion))^2
        limiting_distribution = SpectralDistribution(lb, ub, x -> sqrt((ub - x^2) * (x^2 - lb)) / (pi * ensemble.dimension_proportion * x))
    else
        error("Unknown matrix ensemble.")
    end
    return limiting_distribution
end

function _generate_noise_fisher_factor(dimension, m)
    factor = randn(3 * m, m) / sqrt(3 * m)
    noise = randn(dimension, m) / sqrt(m)
    noise = noise * (factor' * factor)
    return noise
end

function _generate_wishart_factor(dimension, m)
    return randn(dimension, m) / sqrt(m)
end

function _generate_correlated_gaussian(dimension, m, diagonal)
    return Matrix(((randn(dimension, m) / sqrt(m))' .* sqrt.(diagonal))')
end

# TODO check the use of the dimension_proportion
function generate_noise(ensemble::Union{AbstractMatrixEnsemble,AbstractAsymMatrixEnsemble}, dimension::Int)::Union{Symmetric{Float64,Matrix{Float64}},Matrix{Float64}}
    """ Generates a random matrix from the matrix ensemble."""
    if isa(ensemble, WignerEnsemble)
        noise = randn(dimension, dimension) / sqrt(dimension)
        noise = (noise + noise') / 2
        return Symmetric(noise)
    elseif isa(ensemble, WishartEnsemble)
        # noise = randn(dimension, Int(ensemble.dimension_proportion * dimension)) / sqrt(dimension)
        noise = _generate_wishart_factor(dimension, Int(ensemble.dimension_proportion * dimension))
        noise = noise * noise'
        return Symmetric(noise)
    elseif isa(ensemble, WishartFactorEnsemble)
        m = Int(dimension / ensemble.dimension_proportion)
        noise = randn(dimension, m) / sqrt(m)
        return noise
    elseif isa(ensemble, FisherEnsemble)
        noise = _generate_noise_fisher_factor(dimension, dimension)
        return Symmetric((noise + noise') / 2)
    elseif isa(ensemble, FisherFactorEnsemble)
        m = Int(dimension / ensemble.dimension_proportion)
        return _generate_noise_fisher_factor(dimension, m)
    elseif isa(ensemble, UniformEnsemble)
        m = dimension
        diagonal = 10 * rand(m)
        noise = _generate_correlated_gaussian(dimension, m, diagonal)
        noise = (noise + noise') / 2
        return Symmetric(noise)
    elseif isa(ensemble, UniformFactorEnsemble)
        m = Int(dimension / ensemble.dimension_proportion)
        diagonal = 10 * rand(m)
        return _generate_correlated_gaussian(dimension, m, diagonal)
    else
        error("Unknown matrix ensemble.")
    end
end

function true_and_noisy_matrix(ensemble::Union{AbstractMatrixEnsemble,AbstractAsymMatrixEnsemble}, dimension::Int)::Union{Tuple{Symmetric{Float64,Matrix{Float64}},Symmetric{Float64,Matrix{Float64}}},Tuple{Matrix{Float64},Matrix{Float64}}}
    """ Generates a true signal matrix and a noisy matrix."""
    rank = length(ensemble.fixed_spectrum)
    if isa(ensemble, AbstractMatrixEnsemble)
        true_signal = Symmetric(diagm([ensemble.fixed_spectrum; zeros(dimension - rank)]))
    elseif isa(ensemble, AbstractAsymMatrixEnsemble)
        m = Int(dimension / ensemble.dimension_proportion)
        true_signal = diagm([ensemble.fixed_spectrum; zeros(m - rank)])[1:dimension, :]
    else
        error("Unknown matrix ensemble.")
    end
    noisy_matrix = (true_signal + generate_noise(ensemble, dimension))
    return true_signal, noisy_matrix
end

function estimate_fdr(noisy_matrix::Symmetric{Float64,Matrix{Float64}}, rank_estimate::Int, eigenvalues::Union{Vector{Float64},Nothing}=nothing)::Vector{Float64}
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

function estimate_rank(noisy_matrix::Symmetric{Float64,Matrix{Float64}}; threshold_coefficient=0.4)::Tuple{Int,Float64,Vector{Float64},Vector{Float64}}
    """ Estimates the observable rank of the signal matrix.

    Returns the rank estimate, the threshold, the spacings, and the eigenvalues.
    """
    println("Estimating symmetric rank...")
    eigenvalues = sort(eigvals(noisy_matrix), rev=true)
    n = length(eigenvalues)
    spacings = [eigenvalues[i] - eigenvalues[i+1] for i in 1:Int(n * 1 / 2)]
    threshold = Statistics.median(spacings) * n^(1 / 2) * threshold_coefficient
    rank_estimate = maximum(findall(>(threshold), spacings))
    println("Rank estimate: $rank_estimate")
    return rank_estimate, threshold, spacings, eigenvalues
end

function best_k(fdr::Vector{Float64}, level::Float64)::Int
    """ Computes the best k for a given FDR level."""
    controllers = findall(>(level), fdr)
    if length(controllers) == 0
        return 0
    end
    return minimum(controllers) - 1
end

function control_fdr(noisy_matrix::Symmetric{Float64,Matrix{Float64}}, level::Float64; rank_estimate::Union{Int,Nothing}=nothing, compute_spacings=false)::FDRResult
    """ Computes the FDR for different thresholds."""
    eigenvalues = []
    spacings = []
    threshold = -1
    if compute_spacings || isnothing(rank_estimate)
        rank_estimate_sp, threshold, spacings, eigenvalues = estimate_rank(noisy_matrix)
    end
    if isnothing(rank_estimate)
        rank_estimate = rank_estimate_sp
        fdr = estimate_fdr(noisy_matrix, rank_estimate, eigenvalues)
    else
        fdr = estimate_fdr(noisy_matrix, rank_estimate)
    end
    return FDRResult(best_k(fdr, level), fdr, rank_estimate, threshold, spacings, eigenvalues)
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

# function estimate_true_fdr(ensemble::AbstractMatrixEnsemble, dimension::Int, upper_bound::Union{Int,Nothing}=nothing)::Vector{Float64}
#     """ Estimate the true FDR for a given matrix ensemble in dimension via Monte Carlo."""
#     N = 100
#     if isnothing(upper_bound)
#         upper_bound = dimension
#     end
#     fdrs = zeros(upper_bound)
#     for _ in 1:N
#         true_signal, noisy_matrix = true_and_noisy_matrix(ensemble, dimension)
#         _, U = get_top_eigenvectors(true_signal, upper_bound)
#         _, Uh = get_top_eigenvectors(noisy_matrix, upper_bound)
#         for k in 1:upper_bound
#             fdrs[k] += compute_fdp(U, Uh[:, (upper_bound-k+1):upper_bound])
#         end
#     end
#     return fdrs / N
# end

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


# ----------------- Asymmetric ----------------- #

function phi_transform(ensemble::AbstractMatrixEnsemble, q::Float64)::Function
    """ Returns the phi transform of the matrix ensemble for a given q."""
    distribution = limiting_spectral_distribution(ensemble)
    return z -> q * solve(IntegralProblem((x, p) -> z / (z^2 - x^2), distribution.lower_bound, distribution.upper_bound), QuadGKJL()).u + (1 - q) / z
end

function phi_transform_der(ensemble::AbstractMatrixEnsemble, q::Float64)::Function
    """ Returns the derivative of the phi transform of the matrix ensemble for a given q."""
    distribution = limiting_spectral_distribution(ensemble)
    return z -> -q * solve(IntegralProblem((x, p) -> (z^2 + x^2) / (z^2 - x^2)^2, distribution.lower_bound, distribution.upper_bound), QuadGKJL()).u - (1 - q) / z^2
end

function estimate_fdr(noisy_matrix::Matrix{Float64}, rank_estimate::Int,
                      singular_values::Union{Vector{Float64},Nothing}=nothing)::Vector{Float64}
    """ Estimates the false discovery rate for different thresholds of asymmetric matrices."""

    if isnothing(singular_values)
        singular_values = svdvals(noisy_matrix)
    end
    n = size(noisy_matrix, 1)
    m = size(noisy_matrix, 2)

    if length(singular_values) != min(n, m)
        error("The number of singular values is not equal to the minimum of the dimensions.")
    end

    # TODO: Implement code to transpose the matrix, run everything, and transpose back in the end.
    if n > m
        error("Implementation only supports n <= m (more columns than rows).")
    end

    function phi_transform_estimate(z, q)
        """Computes the phi transform estimate at y."""
        return (q / (n - rank_estimate)) * sum(z ./ (z^2 .- singular_values[(rank_estimate+1):n] .^ 2)) + (1 - q) / z
    end

    function phi_transform_der_estimate(z, q)
        """Computes the derivative of the phi transform estimate at y."""
        return (-q / (n - rank_estimate)) * sum((z^2 .+ singular_values[(rank_estimate+1):n] .^ 2) ./ ((z^2 .- singular_values[(rank_estimate+1):n] .^ 2) .^ 2)) - (1 - q) / z^2
    end

    function D_transform_estimate(z)
        """Computes the D transform estimate at y."""
        phi1 = phi_transform_estimate(z, 1)
        phi2 = phi_transform_estimate(z, n / m)
        return phi1 * phi2
    end

    function D_transform_der_estimate(z)
        """Computes the derivative of the D transform estimate at y."""
        phi1 = phi_transform_estimate(z, 1)
        phi2 = phi_transform_estimate(z, n / m)
        phi1_der = phi_transform_der_estimate(z, 1)
        phi2_der = phi_transform_der_estimate(z, n / m)
        return phi1_der * phi2 + phi1 * phi2_der
    end

    costs = ones(n)
    FDR = zeros(n)
    running_sum = 0
    for (i, sigma) in enumerate(singular_values)
        if i <= rank_estimate
            D_val = D_transform_estimate(sigma)
            D_der_val = D_transform_der_estimate(sigma)
            costs[i] += 2 * D_val * phi_transform_estimate(sigma, 1) / D_der_val
        end
        running_sum += costs[i]
        FDR[i] = running_sum / i
    end
    return FDR
end

function estimate_rank(noisy_matrix::Matrix{Float64}; threshold_coefficient=0.40)::Tuple{Int,Float64,Vector{Float64},Vector{Float64}}
    """ Estimates the observable rank of the signal matrix for asymmetric matrices.

    Returns the rank estimate, the threshold, the spacings, and the eigenvalues.
    """
    println("Estimating asymmetric rank...")
    singular_values = svdvals(noisy_matrix)
    n = size(noisy_matrix, 1)
    m = size(noisy_matrix, 2)
    spacings = [singular_values[i] - singular_values[i+1] for i in 1:Int(length(singular_values) / 2)]
    threshold = Statistics.median(spacings) * n^(1 / 2) * threshold_coefficient
    rank_estimate = maximum(findall(>(threshold), spacings))
    println("Rank estimate: $rank_estimate")
    return rank_estimate, threshold, spacings, singular_values
end

function control_fdr(noisy_matrix::Matrix{Float64}, level::Float64; rank_estimate::Union{Int,Nothing}=nothing, compute_spacings=false)::FDRResult
    """ Computes the FDR for different thresholds."""
    singular_values = []
    spacings = []
    threshold = -1
    if compute_spacings || isnothing(rank_estimate)
        rank_estimate_sp, threshold, spacings, singular_values = estimate_rank(noisy_matrix)
    end
    if isnothing(rank_estimate)
        rank_estimate = rank_estimate_sp
        fdr = estimate_fdr(noisy_matrix, rank_estimate, singular_values)
    else
        fdr = estimate_fdr(noisy_matrix, rank_estimate)
    end
    return FDRResult(best_k(fdr, level), fdr, rank_estimate, threshold, spacings, singular_values)
end

function get_top_singular_vectors(A::Matrix{Float64}, k::Int)
    n = size(A, 1)
    if n < k
        error("The number of singular vectors requested is larger than the dimension.")
    end
    U, _, _ = svd(A)
    return U[:, 1:k]
end


function estimate_true_fdr(ensemble::Union{AbstractMatrixEnsemble,AbstractAsymMatrixEnsemble}, dimension::Int, upper_bound::Union{Int,Nothing}=nothing)::Vector{Float64}
    """ Estimate the true FDR for a given matrix ensemble in dimension via Monte Carlo."""
    N = 100
    if isnothing(upper_bound)
        upper_bound = dimension
    end
    true_rank = length(ensemble.fixed_spectrum)
    fdrs = zeros(upper_bound)
    for _ in 1:N
        true_signal, noisy_matrix = true_and_noisy_matrix(ensemble, dimension)
        if isa(ensemble, AbstractMatrixEnsemble)
            _, U = get_top_eigenvectors(true_signal, true_rank)
            _, Uh = get_top_eigenvectors(noisy_matrix, upper_bound)
            for k in 1:upper_bound
                fdrs[k] += compute_fdp(U, Uh[:, (upper_bound-k+1):upper_bound])
            end
        else
            U = get_top_singular_vectors(true_signal, true_rank)
            Uh = get_top_singular_vectors(noisy_matrix, upper_bound)
            for k in 1:upper_bound
                fdrs[k] += compute_fdp(U, Uh[:, 1:min(upper_bound, k)])
            end
        end
    end
    return fdrs / N
end
