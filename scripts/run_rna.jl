using MAT
using ArgParse
using LinearAlgebra
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
        default = "data/preprocessed_rna_data.mat"
        "--matrix_name"
        help = "The name of the matrix in the MATLAB file."
        arg_type = String
        default = "data"
        "--output_folder"
        help = "The folder where the results will be saved."
        arg_type = String
        required = false
        default = "-1"
    end
    return parse_args(s)
end

function main()
    println("Running RNA experiment")
    parsed_args = parse_cli()
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("   $arg  =>  $val")
    end
    data = read_matrix_from_matlab_file(parsed_args["data_file"], parsed_args["matrix_name"])
    println("The size of the data is $(size(data))")
    results_underestimation = FDRControlSubspaceSelection.control_fdr(data, parsed_args["alpha"], threshold_coefficient_rank=1.0)
    results = FDRControlSubspaceSelection.control_fdr(data, parsed_args["alpha"])
    results_overestimation = FDRControlSubspaceSelection.control_fdr(data, parsed_args["alpha"], threshold_coefficient_rank=0.25)
    output_path = parsed_args["output_folder"]
    if output_path == "-1"
        output_path = joinpath("results", "rna", generate_data_time_string())
    end
    create_folder(output_path)
    save_results(results, parsed_args["alpha"], "PBMC", output_path, rank_upper_bound=30)
    plot_wrong_estimates(nothing, results_underestimation, results, results_overestimation, -1, output_path, rank_upper_bound=30, threshold=parsed_args["alpha"])
end

main()
