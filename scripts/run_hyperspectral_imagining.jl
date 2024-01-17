using MAT
using ArgParse
using LinearAlgebra
include("plot_utils.jl")
using FDRControlSubspaceSelection

function read_matrix_from_matlab_file(filename::String, matrix_name::String)
    file = matopen(filename)
    matrix = read(file, matrix_name)
    close(file)
    return matrix
end

# TODO: Standarize features?
# TODO: Make this computation faster. (Use matrices instead of outer products of vectors)
function construct_covariance_matrix_from_data(data::Array{Float64,3})::Symmetric{Float64,Matrix{Float64}}
    n = size(data, 1)
    m = size(data, 2)
    d = size(data, 3)
    covariance_matrix = zeros(d, d)
    for i in 1:n
        for j in 1:m
            covariance_matrix += data[i, j, :] * data[i, j, :]'
        end
    end
    covariance_matrix /= (n * m)
    return Symmetric(covariance_matrix)
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
        default = "data/Indian_pines_corrected.mat"
        "--matrix_name"
        help = "The name of the matrix in the MATLAB file."
        arg_type = String
        default = "indian_pines_corrected"
        "--output_folder"
        help = "The folder where the results will be saved."
        arg_type = String
        required = false
        default = "-1"
    end
    return parse_args(s)
end

function main()
    println("Running hyperspectral imaging experiment")
    parsed_args = parse_cli()
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("   $arg  =>  $val")
    end
    data = read_matrix_from_matlab_file(parsed_args["data_file"], parsed_args["matrix_name"])
    covariance_matrix = construct_covariance_matrix_from_data(data)
    results = FDRControlSubspaceSelection.control_fdr(covariance_matrix, parsed_args["alpha"])
    output_path = parsed_args["output_folder"]
    if output_path == "-1"
        output_path = joinpath("results", "hyperspectral", generate_data_time_string())
    end
    create_folder(output_path)
    save_results(results, parsed_args["alpha"], output_path)
end

main()
