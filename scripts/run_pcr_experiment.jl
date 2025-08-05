import FDRControlSubspaceSelection
using Plots
using Random
using LinearAlgebra
using Statistics
using LaTeXStrings
using JSON

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

function pcr_experiment(X_train, y_train, X_test, y_test, k)
    """
    Perform Principal Component Regression with k components.
    Returns the test error.
    """
    d, n = size(X_train)
    
    # Compute sample covariance matrix
    # X_centered = X_train .- mean(X_train, dims=2)
    X_centered = X_train 
    cov_matrix = (X_centered * X_centered') / (n - 1)
    
    # Get top k eigenvectors (principal components)
    eigenvals, eigenvecs = eigen(cov_matrix)
    
    # Sort in descending order and take top k
    sorted_indices = sortperm(eigenvals, rev=true)
    V_k = eigenvecs[:, sorted_indices[1:k]]  # 400 × k matrix
    
    # Project training features to k-dimensional space
    W_train = V_k' * X_train  # k × n matrix
    
    # Solve least squares: W_train' * gamma_k = y_train
    # gamma_k = (W_train * W_train') \ (W_train * y_train)
    gamma_k = W_train' \ y_train  # k × 1 vector
    
    # Estimate beta_k = V_k * gamma_k
    beta_k = V_k * gamma_k  # 400 × 1 vector
    
    # Compute test error
    y_pred = beta_k' * X_test  # 1 × n_test vector
    test_error = mean((y_pred[:] .- y_test).^2)
    
    return test_error
end

function main()
    d = 400
    ranks = [5, 20, 40]
    proportion = 1/5
    spectrum = 1.1
    n_test_reps = 50  # Number of test repetitions per k
    noise_std = 0.1   # Standard deviation for label noise
    
    test_errors_by_rank = []
    fdrs_by_rank = []
    mses_by_rank = []
    upper_bound_rank = maximum(ranks) + 10
    
    for r in ranks
        println("Processing rank r = $r")
        
        # Generate signal spectrum
        fixed_spectrum = [spectrum for i in 1:r]
        
        # Create ensemble
        ensemble = FDRControlSubspaceSelection.WishartFactorEnsemble(fixed_spectrum, proportion)
        println("BBP transition at: ",  FDRControlSubspaceSelection.estimate_bbp_transition_point(ensemble))

        # Estimate true FDR and MSE for this ensemble
        println("  Estimating true FDR and MSE...")
        true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, upper_bound_rank)
        true_mse = FDRControlSubspaceSelection.estimate_true_mse(ensemble, d, upper_bound_rank)
        push!(fdrs_by_rank, true_fdr)
        push!(mses_by_rank, true_mse)
        
        # Generate training data
        (true_signal, noisy_matrix_train) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
        
        # Features X_i are columns of noisy_matrix (d × n matrix)
        X_train = noisy_matrix_train  # 400 × 2000 matrix
        n_train = size(X_train, 2)
        
        # Labels y_i = <X_i_true, ones(400)> + noise (generated from true signal, not noisy observations)
        true_beta = ones(d)  # The true coefficient vector
        y_train = true_beta' * true_signal + noise_std * randn(1, n_train)
        y_train = y_train[:]  # Convert to vector
        
        # Generate test data once per rank (reuse across all k values)
        println("  Generating test data...")
        test_data = []
        for rep in 1:n_test_reps
            # Generate test data
            (true_signal_test, noisy_matrix_test) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
            X_test = noisy_matrix_test  # This is what we observe (noisy)
            
            # Generate test labels using true signal (same as training)
            y_test = true_beta' * true_signal_test + noise_std * randn(1, size(X_test, 2))
            y_test = y_test[:]
            
            push!(test_data, (X_test, y_test))
        end
        
        # Test for k from 1 to 2*r
        max_k = 2 * r
        test_errors = zeros(max_k)
        
        for k in 1:max_k
            println("  Testing k = $k")
            
            # Perform multiple test repetitions using the same test data
            test_errors_k = zeros(n_test_reps)
            
            for rep in 1:n_test_reps
                X_test, y_test = test_data[rep]
                
                # Perform PCR and compute test error
                test_errors_k[rep] = pcr_experiment(X_train, y_train, X_test, y_test, k)
            end
            
            # Average test error over repetitions
            test_errors[k] = mean(test_errors_k)
        end
        
        push!(test_errors_by_rank, test_errors)
        
        # Plot test error for this rank
        println("  Creating test error plot...")
        plot(1:max_k, test_errors, 
             label=false, 
             line=(4, :solid), 
             xlabel=L"Truncation rank $k$",
             ylabel="Test Error",
             fg_legend=:transparent, 
             legend_background_color=:transparent)
        vline!([r], label="True rank", line=(2, :dash), color=:red)
        savefig("results/pcr_experiment/pcr_test_error_r$(r)_d$(d)_spectrum$(spectrum).pdf")
        
        # Plot combined FDR and MSE for this rank (dual-axis)
        println("  Creating combined FDR+MSE plot...")
        rank_minimum = argmin(true_mse)
        mse_minimum_value = minimum(true_mse)
        println("  MSE minimum: $(mse_minimum_value) at rank $(rank_minimum)")
        range = 1:(min(r+15, length(true_mse)))
        
        # Determine if this is the first plot (for legend)
        show_legend = (r == ranks[1])
        
        # Plot FDR on left axis (no legend initially)
        p1 = plot(range, true_fdr[range], 
                  label="", 
                  line=(4, :solid), 
                  color=1,  # Default blue
                  xlabel=L"Truncation rank $k$",
                  ylabel="False Discovery Rate",
                  ylims=(0, 1),
                  legend=false)
        
        # Add MSE on right axis using twinx (no legend initially)
        p2 = twinx(p1)
        plot!(p2, range, true_mse[range], 
              label="", 
              line=(4, :solid), 
              marker=(:circle, 4),  # Add circular markers
              color=:orange,  # Orange
              ylabel="Mean Squared Error",
              legend=false)
        
        # Add vertical line for MSE minimum
        vline!(p2, [rank_minimum], label="", line=(2, :dot), color=5, alpha=0.7)  # Pink/magenta
        
        # Add combined legend only for the first plot
        if show_legend
            # Create invisible plots for legend entries
            plot!(p2, Float64[], Float64[], label="True FDR", line=(4, :solid), color=1)
            plot!(p2, Float64[], Float64[], label="True MSE", line=(4, :solid), marker=(:circle, 6), color=:orange)  # Orange
            plot!(p2, Float64[], Float64[], label="MSE minimum", line=(2, :dot), color=5)  # Pink/magenta
            plot!(p2, legend=:bottomright, fg_legend=:transparent, legend_background_color=:transparent, markersize=6)  # Explicit legend marker size
        end
        
        savefig("results/pcr_experiment/pcr_fdr_mse_r$(r)_d$(d)_spectrum$(spectrum).pdf")
    end
    
    # Create combined plot for all ranks
    colors = [12, 7, 2]
    lines = [:solid, :dash, :dot]
    
    plot()
    for (i, r) in enumerate(ranks)
        max_k = 2 * r
        plot!(1:max_k, test_errors_by_rank[i], 
              label="r = $r", 
              line=(4, lines[i]), 
              color=colors[i])
        vline!([r], line=(2, :dash), color=colors[i], alpha=0.7, label="")
    end
    
    xlabel!(L"Truncation rank $k$")
    ylabel!("Test Error")
    # title!("PCR Test Error vs Number of Components")
    
    # Save combined plot
    savefig("results/pcr_experiment/pcr_test_error_combined_d$(d)_spectrum$(spectrum).pdf")
    
    # Save all data to JSON files
    println("Saving data...")
    experiment_data = Dict(
        "parameters" => Dict(
            "d" => d,
            "ranks" => ranks,
            "proportion" => proportion,
            "spectrum" => spectrum,
            "n_test_reps" => n_test_reps,
            "noise_std" => noise_std,
            "upper_bound_rank" => upper_bound_rank
        ),
        "test_errors_by_rank" => test_errors_by_rank,
        "fdrs_by_rank" => fdrs_by_rank,
        "mses_by_rank" => mses_by_rank
    )
    
    # Save main data file
    open("results/pcr_experiment/pcr_experiment_data_d$(d)_spectrum$(spectrum).json", "w") do f
        JSON.print(f, experiment_data, 2)  # Pretty print with indentation
    end
    
    # Plot FDR results
    println("Creating FDR plots...")
    plot()
    for (i, r) in enumerate(ranks)
        plot!(1:upper_bound_rank, fdrs_by_rank[i][1:upper_bound_rank], 
              label=L"True FDR $r = $"*string(r), 
              line=(4, lines[i]), 
              color=colors[i],
              fg_legend=:transparent, 
              legend_background_color=:transparent)
    end
    xlabel!(L"Truncation rank $k$")
    ylabel!("False Discovery Rate")
    savefig("results/pcr_experiment/pcr_fdr_combined_d$(d)_spectrum$(spectrum).pdf")
    
    # Individual MSE plots are now generated as we go (in the main loop)
    
    println("Experiment completed. Results saved to results/pcr_experiment/")
end

main()