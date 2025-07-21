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
    spectrum = 1.25
    n_test_reps = 40  # Number of test repetitions per k
    noise_std = 0.1   # Standard deviation for label noise
    
    test_errors_by_rank = []
    
    for r in ranks
        println("Processing rank r = $r")
        
        # Generate signal spectrum
        fixed_spectrum = [spectrum for i in 1:r]
        
        # Create ensemble
        ensemble = FDRControlSubspaceSelection.WishartFactorEnsemble(fixed_spectrum, proportion)
        
        # Generate training data
        (true_signal, noisy_matrix_train) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
        
        # Features X_i are columns of noisy_matrix (d × n matrix)
        X_train = noisy_matrix_train  # 400 × 2000 matrix
        n_train = size(X_train, 2)
        
        # Labels y_i = <X_i_true, ones(400)> + noise (generated from true signal, not noisy observations)
        true_beta = ones(d)  # The true coefficient vector
        y_train = true_beta' * true_signal + noise_std * randn(1, n_train)
        y_train = y_train[:]  # Convert to vector
        
        # Test for k from 1 to 2*r
        max_k = 2 * r
        test_errors = zeros(max_k)
        
        for k in 1:max_k
            println("  Testing k = $k")
            
            # Perform multiple test repetitions
            test_errors_k = zeros(n_test_reps)
            
            for rep in 1:n_test_reps
                # Generate fresh test data
                (true_signal_test, noisy_matrix_test) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
                X_test = noisy_matrix_test  # This is what we observe (noisy)
                n_test = size(X_test, 2)
                
                # Generate test labels using true signal (same as training)
                y_test = true_beta' * true_signal_test + noise_std * randn(1, n_test)
                y_test = y_test[:]
                
                # Perform PCR and compute test error
                test_errors_k[rep] = pcr_experiment(X_train, y_train, X_test, y_test, k)
            end
            
            # Average test error over repetitions
            test_errors[k] = mean(test_errors_k)
        end
        
        push!(test_errors_by_rank, test_errors)
        
        # Plot for this rank
        plot(1:max_k, test_errors, 
             label="PCR Test Error (r = $r)", 
             line=(4, :solid), 
             xlabel=L"Number of components $k$",
             ylabel="Test Error (MSE)",
             title="Principal Component Regression Test Error",
             fg_legend=:transparent, 
             legend_background_color=:transparent)
        
        # Mark the true rank
        vline!([r], label="True rank r = $r", line=(2, :dash), color=:red)
        
        # Save plot
        savefig("results/pcr_experiment/pcr_test_error_r$(r)_d$(d)_spectrum$(spectrum).pdf")
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
    
    xlabel!(L"Number of components $k$")
    ylabel!("Test Error (MSE)")
    title!("PCR Test Error vs Number of Components")
    
    # Save combined plot
    savefig("results/pcr_experiment/pcr_test_error_combined_d$(d)_spectrum$(spectrum).pdf")
    
    println("Experiment completed. Results saved to results/pcr_experiment/")
end

main()