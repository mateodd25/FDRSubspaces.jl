using FDRControlSubspaceSelection
using Plots
using Random
using LinearAlgebra
using Statistics
using LaTeXStrings
using JSON
using MAT

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

function load_sp500_data()
    """Load S&P 500 return data from MAT file."""
    data_file = "data/sp500_returns.mat"
    
    if !isfile(data_file)
        error("Data file not found: $data_file. Please run 'python scripts/setup_sp500_returns.py' first.")
    end
    
    println("Loading S&P 500 return data from $data_file...")
    data = matread(data_file)
    
    X = data["X"]  # T × N matrix (days × assets)
    tickers = data["tickers"]
    
    println("Data loaded:")
    println("  Shape: $(size(X)) (days × assets)")
    println("  Time period: $(data["start_date"]) to $(data["end_date"])")
    println("  Number of assets: $(size(X, 2))")
    println("  Number of trading days: $(size(X, 1))")
    
    return X', tickers  # Return N × T (assets × days) for consistency with other experiments
end

function run_sp500_fdr_analysis(X, tickers; alpha=0.05, max_components=50)
    """
    Run FDR analysis on S&P 500 return data.
    
    X: N × T matrix (assets × days)
    tickers: list of asset tickers
    alpha: FDR level
    max_components: maximum number of components to consider
    """
    N, T = size(X)
    println("Running FDR analysis on S&P 500 data...")
    println("  Matrix size: $N assets × $T days")
    println("  FDR level: $alpha")
    
    # Compute sample covariance matrix
    println("Computing sample covariance matrix...")
    X_centered = X .- mean(X, dims=2)  # Center each asset
    Σ_sample = (X_centered * X_centered') / (T - 1)
    
    # Ensure symmetry
    Σ_sample = Symmetric(Σ_sample)
    
    # Run FDR control
    println("Running FDR control procedure...")
    fdr_result = FDRControlSubspaceSelection.control_fdr(Σ_sample, alpha)
    
    println("FDR Analysis Results:")
    println("  Estimated rank: $(fdr_result.rank_estimate)")
    println("  Best k (at α=$alpha): $(fdr_result.best_k)")
    println("  Rank estimation threshold: $(fdr_result.threshold)")
    
    # Eigenvalue decomposition for further analysis
    eigenvals, eigenvecs = eigen(Σ_sample)
    sorted_indices = sortperm(eigenvals, rev=true)
    eigenvals_sorted = eigenvals[sorted_indices]
    eigenvecs_sorted = eigenvecs[:, sorted_indices]
    
    return Dict(
        "fdr_result" => fdr_result,
        "eigenvalues" => eigenvals_sorted,
        "eigenvectors" => eigenvecs_sorted,
        "sample_covariance" => Σ_sample,
        "tickers" => tickers,
        "N" => N,
        "T" => T
    )
end

function plot_sp500_results(results, output_dir="results/sp500_experiment")
    """Generate plots for S&P 500 FDR analysis."""
    
    # Create output directory
    mkpath(output_dir)
    
    fdr_result = results["fdr_result"]
    eigenvals = results["eigenvalues"]
    N = results["N"]
    T = results["T"]
    
    # Plot 1: Eigenvalues with rank estimates
    println("Creating eigenvalue plot...")
    n_plot = min(100, length(eigenvals))  # Plot first 100 eigenvalues
    
    plot(1:n_plot, eigenvals[1:n_plot],
         line=(3, :solid),
         marker=(:circle, 3),
         xlabel=L"Index $i$",
         ylabel="Eigenvalue",
         title="S&P 500 Sample Covariance Eigenvalues",
         legend=false,
         yscale=:log10)
    
    # Mark the estimated rank and best k
    vline!([fdr_result.rank_estimate], 
           line=(3, :dash), 
           color=:red, 
           label="Estimated rank ($(fdr_result.rank_estimate))")
    
    if fdr_result.best_k > 0
        vline!([fdr_result.best_k], 
               line=(3, :dot), 
               color=:blue, 
               label="Best k ($(fdr_result.best_k))")
    end
    
    plot!(legend=:topright)
    savefig(joinpath(output_dir, "sp500_eigenvalues.pdf"))
    
    # Plot 2: FDR curve
    println("Creating FDR curve plot...")
    k_range = 1:min(50, length(fdr_result.fdr))
    
    plot(k_range, fdr_result.fdr[k_range],
         line=(3, :solid),
         marker=(:circle, 3),
         xlabel=L"Number of components $k$",
         ylabel="False Discovery Rate",
         title="S&P 500 FDR Control",
         legend=false,
         color=:blue)
    
    # Mark the best k
    if fdr_result.best_k > 0 && fdr_result.best_k <= length(k_range)
        scatter!([fdr_result.best_k], [fdr_result.fdr[fdr_result.best_k]],
                marker=(:star, 8),
                color=:red,
                label="Best k")
        plot!(legend=:topright)
    end
    
    savefig(joinpath(output_dir, "sp500_fdr_curve.pdf"))
    
    # Plot 3: Eigenvalue spacings
    if length(fdr_result.spacings) > 0
        println("Creating spacings plot...")
        
        plot(1:length(fdr_result.spacings), fdr_result.spacings,
             line=(2, :solid),
             marker=(:circle, 2),
             xlabel="Index",
             ylabel="Eigenvalue Spacing",
             title="S&P 500 Eigenvalue Spacings",
             legend=false)
        
        hline!([fdr_result.threshold],
               line=(3, :dash),
               color=:red,
               label="Threshold ($(round(fdr_result.threshold, digits=4)))")
        
        plot!(legend=:topright)
        savefig(joinpath(output_dir, "sp500_spacings.pdf"))
    end
    
    println("Plots saved to: $output_dir")
end

function analyze_top_components(results, n_components=5)
    """Analyze the top principal components and their loadings."""
    
    eigenvecs = results["eigenvectors"]
    eigenvals = results["eigenvalues"]
    tickers = results["tickers"]
    
    println("\nTop $n_components Principal Components Analysis:")
    
    for i in 1:min(n_components, size(eigenvecs, 2))
        println("\nComponent $i (eigenvalue: $(round(eigenvals[i], digits=4))):")
        
        # Get loadings for this component
        loadings = eigenvecs[:, i]
        
        # Find top positive and negative loadings
        sorted_indices = sortperm(abs.(loadings), rev=true)
        
        println("  Top 5 positive loadings:")
        pos_loadings = filter(j -> loadings[j] > 0, sorted_indices[1:min(10, length(sorted_indices))])
        for (rank, idx) in enumerate(pos_loadings[1:min(5, length(pos_loadings))])
            ticker = length(tickers) >= idx ? tickers[idx] : "Unknown"
            println("    $rank. $ticker: $(round(loadings[idx], digits=4))")
        end
        
        println("  Top 5 negative loadings:")
        neg_loadings = filter(j -> loadings[j] < 0, sorted_indices[1:min(10, length(sorted_indices))])
        for (rank, idx) in enumerate(neg_loadings[1:min(5, length(neg_loadings))])
            ticker = length(tickers) >= idx ? tickers[idx] : "Unknown"
            println("    $rank. $ticker: $(round(loadings[idx], digits=4))")
        end
    end
end

function main()
    println("=== S&P 500 FDR Subspace Selection Experiment ===\n")
    
    try
        # Load data
        X, tickers = load_sp500_data()
        
        # Run FDR analysis
        results = run_sp500_fdr_analysis(X, tickers, alpha=0.05)
        
        # Generate plots
        plot_sp500_results(results)
        
        # Analyze top components
        analyze_top_components(results)
        
        # Save results
        println("\nSaving results...")
        output_dir = "results/sp500_experiment"
        mkpath(output_dir)
        
        # Save summary results to JSON
        summary = Dict(
            "parameters" => Dict(
                "alpha" => 0.05,
                "N_assets" => results["N"],
                "T_days" => results["T"]
            ),
            "results" => Dict(
                "estimated_rank" => results["fdr_result"].rank_estimate,
                "best_k" => results["fdr_result"].best_k,
                "threshold" => results["fdr_result"].threshold,
                "top_10_eigenvalues" => results["eigenvalues"][1:min(10, length(results["eigenvalues"]))],
                "fdr_curve" => results["fdr_result"].fdr[1:min(50, length(results["fdr_result"].fdr))]
            )
        )
        
        open(joinpath(output_dir, "sp500_fdr_results.json"), "w") do f
            JSON.print(f, summary, 2)
        end
        
        println("Results saved to: $output_dir")
        println("\n=== S&P 500 Experiment Completed Successfully ===")
        
    catch e
        println("Error in S&P 500 experiment: $e")
        println("\nMake sure to run the data setup first:")
        println("  python scripts/setup_sp500_returns.py")
        return 1
    end
    
    return 0
end

# Run the experiment
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end