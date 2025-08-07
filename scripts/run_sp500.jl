using MAT
using ArgParse
using LinearAlgebra
using Statistics
include("plot_utils.jl")
using FDRControlSubspaceSelection

function read_matrix_from_matlab_file(filename::String, matrix_name::String)
    file = matopen(filename)
    matrix = read(file, matrix_name)
    close(file)
    return convert(Matrix{Float64}, matrix)
end

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--alpha"
        help = "The desired FDR level"
        arg_type = Float64
        required = true
        default = 0.05
        "--data_file"
        help = "The file that contains the data. This is a MATLAB file."
        arg_type = String
        required = false
        default = "data/sp500_returns.mat"
        "--matrix_name"
        help = "The name of the matrix in the MATLAB file."
        arg_type = String
        default = "X"
        "--output_folder"
        help = "The folder where the results will be saved."
        arg_type = String
        required = false
        default = "-1"
    end
    return parse_args(s)
end

function plot_sp500_histogram(eigenvalues::Vector{Float64}, output_path::String)
    general_setup()
    
    # Filter eigenvalues to range [0, 10] to remove outliers for S&P 500
    filtered_eigenvals = eigenvalues[eigenvalues .<= 10]
    
    histogram(filtered_eigenvals, 
             normalize=:pdf, 
             bins=min(trunc(Int, length(filtered_eigenvals) / 3), 200),
             linecolor=:transparent, 
             legend=false, 
             xlims=(0, 10))
    xaxis!(L"Eigenvalue $\lambda$")
    yaxis!("Normalized frequency")
    savefig(output_path)
end

function main()
    println("Running S&P 500 experiment")
    parsed_args = parse_cli()
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("   $arg  =>  $val")
    end
    data = read_matrix_from_matlab_file(parsed_args["data_file"], parsed_args["matrix_name"])
    println("The size of the data is $(size(data))")
    
    # Transpose to match expected format (assets × days)
    data = data'
    
    # Convert to correlation matrix for financial data stability
    correlation_matrix = cor(data')
    Σ_sample = Symmetric(correlation_matrix)
    
    results_underestimation = FDRControlSubspaceSelection.control_fdr(Σ_sample, parsed_args["alpha"], threshold_coefficient_rank=0.75)
    results = FDRControlSubspaceSelection.control_fdr(Σ_sample, parsed_args["alpha"])
    results_overestimation = FDRControlSubspaceSelection.control_fdr(Σ_sample, parsed_args["alpha"], threshold_coefficient_rank=0.25)
    
    output_path = parsed_args["output_folder"]
    if output_path == "-1"
        output_path = joinpath("results", "sp500", generate_data_time_string())
    end
    create_folder(output_path)
    save_results(results, parsed_args["alpha"], "SP500", output_path, rank_upper_bound=100)
    plot_wrong_estimates(nothing, results_underestimation, results, results_overestimation, -1, output_path, rank_upper_bound=100, threshold=parsed_args["alpha"])
    
    # Create custom histogram for S&P 500 with [0, 10] range
    if length(results.eigenvalues) > 0
        plot_sp500_histogram(results.eigenvalues, joinpath(output_path, "empirical_density.pdf"))
    end
end

main()
