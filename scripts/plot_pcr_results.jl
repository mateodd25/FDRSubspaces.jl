using Plots
using JSON
using LaTeXStrings
using ArgParse

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

function parse_arguments()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--data-file", "-f"
            help = "Path to the JSON data file"
            arg_type = String
            default = "results/pcr_experiment/pcr_experiment_data_d400_spectrum1.20.json"
        "--output-dir", "-o"
            help = "Output directory for plots"
            arg_type = String
            default = "results/pcr_experiment/"
    end
    return parse_args(s)
end

function load_experiment_data(data_file::String)
    """Load experiment data from JSON file."""
    if !isfile(data_file)
        error("Data file not found: $data_file")
    end
    
    println("Loading data from: $data_file")
    data = JSON.parsefile(data_file)
    return data
end

function plot_individual_test_errors(data, output_dir::String)
    """Plot individual test error plots for each rank."""
    println("Creating individual test error plots...")
    
    params = data["parameters"]
    ranks = params["ranks"]
    d = params["d"]
    spectrum = params["spectrum"]
    test_errors_by_rank = data["test_errors_by_rank"]
    
    for (i, r) in enumerate(ranks)
        max_k = 2 * r
        test_errors = test_errors_by_rank[i]
        
        plot(1:max_k, test_errors, 
             label=false, 
             line=(4, :solid), 
             xlabel=L"Truncation rank $k$",
             ylabel="Test Error",
             fg_legend=:transparent, 
             legend_background_color=:transparent)
        vline!([r], label="True rank", line=(2, :dash), color=:red)
        
        savefig(joinpath(output_dir, "pcr_test_error_r$(r)_d$(d)_spectrum$(spectrum).pdf"))
        println("  Saved: pcr_test_error_r$(r)_d$(d)_spectrum$(spectrum).pdf")
    end
end

function plot_combined_test_errors(data, output_dir::String)
    """Plot combined test error plot for all ranks."""
    println("Creating combined test error plot...")
    
    params = data["parameters"]
    ranks = params["ranks"]
    d = params["d"]
    spectrum = params["spectrum"]
    test_errors_by_rank = data["test_errors_by_rank"]
    
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
    
    savefig(joinpath(output_dir, "pcr_test_error_combined_d$(d)_spectrum$(spectrum).pdf"))
    println("  Saved: pcr_test_error_combined_d$(d)_spectrum$(spectrum).pdf")
end

function plot_individual_fdr_mse(data, output_dir::String)
    """Plot individual dual-axis FDR+MSE plots for each rank."""
    println("Creating individual FDR+MSE plots...")
    
    params = data["parameters"]
    ranks = params["ranks"]
    d = params["d"]
    spectrum = params["spectrum"]
    fdrs_by_rank = data["fdrs_by_rank"]
    mses_by_rank = data["mses_by_rank"]
    
    for (i, r) in enumerate(ranks)
        true_fdr = fdrs_by_rank[i]
        true_mse = mses_by_rank[i]
        
        rank_minimum = argmin(true_mse)
        mse_minimum_value = minimum(true_mse)
        println("  Rank $r - MSE minimum: $(mse_minimum_value) at rank $(rank_minimum)")
        
        range = 1:(min(r+10, length(true_mse)))
        
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
              color=7,  # Orange
              ylabel="Mean Squared Error",
              legend=false)
        
        # Add vertical line for MSE minimum
        vline!(p2, [rank_minimum], label="", line=(2, :dot), color=12, alpha=0.7)  # Pink/magenta
        
        # Add combined legend only for the first plot
        if show_legend
            # Create invisible plots for legend entries
            plot!(p2, Float64[], Float64[], label="True FDR", line=(4, :solid), color=1)
            plot!(p2, Float64[], Float64[], label="True MSE", line=(4, :solid), marker=(:circle, 6), color=7)  # Orange
            plot!(p2, Float64[], Float64[], label="MSE minimum", line=(2, :dot), color=12)  # Pink/magenta
            plot!(p2, legend=:bottomright, fg_legend=:transparent, legend_background_color=:transparent, markersize=6)  # Explicit legend marker size
        end
        
        savefig(joinpath(output_dir, "pcr_fdr_mse_r$(r)_d$(d)_spectrum$(spectrum).pdf"))
        println("  Saved: pcr_fdr_mse_r$(r)_d$(d)_spectrum$(spectrum).pdf")
    end
end

function plot_combined_fdr(data, output_dir::String)
    """Plot combined FDR plot for all ranks."""
    println("Creating combined FDR plot...")
    
    params = data["parameters"]
    ranks = params["ranks"]
    d = params["d"]
    spectrum = params["spectrum"]
    upper_bound_rank = params["upper_bound_rank"]
    fdrs_by_rank = data["fdrs_by_rank"]
    
    colors = [12, 7, 2]
    lines = [:solid, :dash, :dot]
    
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
    
    savefig(joinpath(output_dir, "pcr_fdr_combined_d$(d)_spectrum$(spectrum).pdf"))
    println("  Saved: pcr_fdr_combined_d$(d)_spectrum$(spectrum).pdf")
end

function main()
    args = parse_arguments()
    data_file = args["data-file"]
    output_dir = args["output-dir"]
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Load data
    data = load_experiment_data(data_file)
    
    # Generate all plots
    plot_individual_test_errors(data, output_dir)
    plot_combined_test_errors(data, output_dir)
    plot_individual_fdr_mse(data, output_dir)
    plot_combined_fdr(data, output_dir)
    
    println("\nAll plots generated successfully!")
    println("Output directory: $output_dir")
end

# Run the script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end