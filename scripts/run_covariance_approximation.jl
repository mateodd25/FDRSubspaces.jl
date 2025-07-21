import FDRControlSubspaceSelection
using Plots
using Random
using LinearAlgebra
using Statistics
using LaTeXStrings

# Fix plotting issues on NixOS
ENV["GKSwstype"] = "nul"  # Disable problematic GKS socket connections
gr()  # Use GR backend which works better with NixOS

fntsm = font("serif-roman", pointsize = 16)
fntlg = font("serif-roman", pointsize = 16)
default(
    titlefont = fntlg,
    guidefont = fntlg,
    tickfont = fntsm,
    legendfont = fntsm,
)

Random.seed!(1)

function low_rank_approximation(cov_matrix, k)
    """
    Compute rank-k approximation of covariance matrix using top k eigenvectors.
    Returns the low-rank approximation matrix.
    """
    # Get eigenvectors and eigenvalues
    eigenvals, eigenvecs = eigen(cov_matrix)
    
    # Sort in descending order and take top k
    sorted_indices = sortperm(eigenvals, rev=true)
    
    if k == 0
        return zeros(size(cov_matrix))
    end
    
    # Take top k eigenvectors and eigenvalues
    top_eigenvecs = eigenvecs[:, sorted_indices[1:k]]
    top_eigenvals = eigenvals[sorted_indices[1:k]]
    
    # Reconstruct low-rank approximation: V_k * Λ_k * V_k'
    return top_eigenvecs * Diagonal(top_eigenvals) * top_eigenvecs'
end

function compute_approximation_error(cov_true, cov_noisy, k, error_metric="frobenius")
    """
    Compute the error between low-rank approximation of noisy covariance 
    and the true covariance matrix.
    """
    # Get low-rank approximation of noisy covariance
    cov_noisy_approx = low_rank_approximation(cov_noisy, k)
    
    # Compute error
    error_matrix = cov_noisy_approx - cov_true
    
    if error_metric == "frobenius"
        return norm(error_matrix, 2)  # Frobenius norm
    elseif error_metric == "spectral"
        return opnorm(error_matrix, 2)  # Spectral norm (largest singular value)
    elseif error_metric == "nuclear"
        return sum(svdvals(error_matrix))  # Nuclear norm
    else
        error("Unknown error metric: $error_metric")
    end
end

function main()
    d = 400
    ranks = [5, 20, 40]
    proportion = 1/5
    spectrum = 1.25
    n_repetitions = 20  # Number of repetitions for averaging
    
    # Error metrics to compute
    error_metrics = ["frobenius", "spectral"]
    metric_labels = ["Frobenius Norm", "Spectral Norm"]
    
    results_by_rank = []
    
    for r in ranks
        println("Processing rank r = $r")
        
        # Generate signal spectrum
        fixed_spectrum = [spectrum for i in 1:r]
        
        # Create ensemble
        ensemble = FDRControlSubspaceSelection.WishartFactorEnsemble(fixed_spectrum, proportion)
        
        # Test for k from 0 to 2*r
        max_k = 2 * r
        
        # Store results for each metric
        errors_by_metric = Dict()
        for metric in error_metrics
            errors_by_metric[metric] = zeros(max_k + 1)  # +1 for k=0
        end
        
        for rep in 1:n_repetitions
            if rep % 5 == 1
                println("  Repetition $rep/$n_repetitions")
            end
            
            # Generate data
            (true_signal, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
            
            # Compute covariance matrices
            # Center the data
            true_centered = true_signal #.- mean(true_signal, dims=2)
            noisy_centered = noisy_matrix #.- mean(noisy_matrix, dims=2)
            
            # Compute sample covariance matrices
            n_samples = size(true_signal, 2)
            cov_true = (true_centered * true_centered') / (n_samples - 1)
            cov_noisy = (noisy_centered * noisy_centered') / (n_samples - 1)
            
            # Compute errors for different values of k
            for k in 0:max_k
                for metric in error_metrics
                    error_val = compute_approximation_error(cov_true, cov_noisy, k, metric)
                    errors_by_metric[metric][k + 1] += error_val / n_repetitions
                end
            end
        end
        
        push!(results_by_rank, errors_by_metric)
        
        # Create plots for this rank
        for (i, metric) in enumerate(error_metrics)
            plot(0:max_k, errors_by_metric[metric], 
                 label="$(metric_labels[i]) Error (r = $r)", 
                 line=(4, :solid), 
                 xlabel=L"Approximation rank $k$",
                 ylabel="Approximation Error",
                 title="Covariance Matrix Approximation Error",
                 fg_legend=:transparent, 
                 legend_background_color=:transparent)
            
            # Mark the true rank
            vline!([r], label="True rank r = $r", line=(2, :dash), color=:red)
            
            # Save plot
            savefig("results/pcr_experiment/cov_approx_$(metric)_r$(r)_d$(d)_spectrum$(spectrum).pdf")
        end
    end
    
    # Create combined plots for all ranks
    colors = [12, 7, 2]
    lines = [:solid, :dash, :dot]
    
    for (metric_idx, metric) in enumerate(error_metrics)
        plot()
        for (i, r) in enumerate(ranks)
            max_k = 2 * r
            errors = results_by_rank[i][metric]
            plot!(0:max_k, errors[1:(max_k+1)], 
                  label="r = $r", 
                  line=(4, lines[i]), 
                  color=colors[i])
            vline!([r], line=(2, :dash), color=colors[i], alpha=0.7, label="")
        end
        
        xlabel!(L"Approximation rank $k$")
        ylabel!("$(metric_labels[metric_idx]) Error")
        title!("Covariance Matrix Approximation: $(metric_labels[metric_idx])")
        
        # Save combined plot
        savefig("results/pcr_experiment/cov_approx_$(metric)_combined_d$(d)_spectrum$(spectrum).pdf")
    end
    
    # Also create a plot showing the ratio of approximation error to baseline error (k=0)
    plot()
    for (i, r) in enumerate(ranks)
        max_k = 2 * r
        errors = results_by_rank[i]["frobenius"]
        baseline_error = errors[1]  # k=0 error
        relative_errors = errors[1:(max_k+1)] ./ baseline_error
        
        plot!(0:max_k, relative_errors, 
              label="r = $r", 
              line=(4, lines[i]), 
              color=colors[i])
        vline!([r], line=(2, :dash), color=colors[i], alpha=0.7, label="")
    end
    
    xlabel!(L"Approximation rank $k$")
    ylabel!("Relative Error (normalized by k=0)")
    title!("Relative Covariance Matrix Approximation Error")
    
    # Save relative error plot
    savefig("results/pcr_experiment/cov_approx_relative_d$(d)_spectrum$(spectrum).pdf")
    
    println("Covariance approximation experiment completed. Results saved to results/pcr_experiment/")
end

main()