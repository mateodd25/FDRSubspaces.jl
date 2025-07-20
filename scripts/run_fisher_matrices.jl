import FDRControlSubspaceSelection
using ArgParse
using Plots
using Random
using LaTeXStrings
using Match

# Fix plotting issues on NixOS
ENV["GKSwstype"] = "nul"  # Disable problematic GKS socket connections
gr()  # Use GR backend which works better with NixOS

include("plot_utils.jl")

@enum ExperimentType fisher = 1 wishart = 2 fisher_factor = 3 wishart_factor = 4 wigner = 5 uniform = 6 uniform_factor = 7

function name_to_experiment(name::String)
        @match name begin
                "fisher" => fisher
                "fisher_factor" => fisher_factor
                "wishart" => wishart
                "wishart_factor" => wishart_factor
                "wigner" => wigner
                "uniform" => uniform
                "uniform_factor" => uniform_factor
                _ => error("Unknown experiment name")
        end
end

function experiment_name(exp_type::ExperimentType)
        @match exp_type begin
                $fisher => "fisher"
                $fisher_factor => "fisher_factor"
                $wishart => "wishart"
                $wishart_factor => "wishart_factor"
                $wigner => "wigner"
                $uniform => "uniform"
                $uniform_factor => "uniform_factor"
                _ => error("Unknown experiment type")
        end
end

function experiment_struct(exp_type::ExperimentType)
        @match exp_type begin
                $fisher => FDRControlSubspaceSelection.FisherEnsemble
                $fisher_factor => FDRControlSubspaceSelection.FisherFactorEnsemble
                $wishart => FDRControlSubspaceSelection.WishartEnsemble
                $wishart_factor => FDRControlSubspaceSelection.WishartFactorEnsemble
                $wigner => FDRControlSubspaceSelection.WignerEnsemble
                $uniform => FDRControlSubspaceSelection.UniformEnsemble
                $uniform_factor => FDRControlSubspaceSelection.UniformFactorEnsemble
                _ => error("Unknown experiment type")
        end
end

function has_proportion(exp_type::ExperimentType)
        return :dimension_proportion in fieldnames(experiment_struct(exp_type))
end

function experiment_fixed_spectrum(exp_type::ExperimentType; proportion=1.0, hardness=3)
    theta = 1.30
    shift = 0.0
    if hardness == 1
        shift = 1.0
    elseif hardness == 3
        shift = -10.0 * theta^(-9)
    end
    println("    Shift is $shift")
    @match proportion begin
                1.0 => begin
                    @match exp_type begin
                        $fisher => [1.4 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $fisher_factor => [2.3 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $wishart => [2.1 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $wishart_factor => [1.2 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $wigner => [1 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $uniform => [2.1 + shift + 10 * theta^(1 - i) for i in 1:20]
                        $uniform_factor => [3.1 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        _ => error("Unknown experiment type")
                        end
                end
                0.5 => begin
                    @match exp_type begin
                        $fisher => [1.4 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $fisher_factor => [1.9 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $wishart => [1.4 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $wishart_factor => [1.1 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $wigner => [1 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        $uniform => [2.1 + shift + 10 * theta^(1 - i) for i in 1:20]
                        $uniform_factor => [2.5 + shift + 10 * theta^(-i + 1) for i in 1:20]
                        _ => error("Unknown experiment type")
                        end
                end
        end
end

function hardness_to_name(hardness::Int)
    @match hardness begin
        1 => "easy"
        2 => "medium"
        3 => "hard"
        _ => error("Unknown hardness")
    end
end

function parse_cli()
        s = ArgParseSettings()
        @add_arg_table s begin
                "--alpha"
                help = "The desired FDR level"
                arg_type = Float64
                required = true
                default = 0.05
                "--experiments"
                help = "Experiments to run separated by commas. If empty, it runs everything."
                arg_type = Union{String,Nothing}
                required = false
                default = nothing
        end
        return parse_args(s)
end


function main()
    println("Running expriments ...")
    parsed_args = parse_cli()
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("   $arg  =>  $val")
        end
        dimensions = [100, 200, 500, 1000]
        proportions = [0.5, 1]
        experiments = instances(ExperimentType)
        if !isnothing(parsed_args["experiments"])
                experiments = [name_to_experiment(string(instance)) for instance in split(parsed_args["experiments"])]
        end
        alpha = parsed_args["alpha"]
    date_time = generate_data_time_string()
    for (i, d) in enumerate(dimensions)
        for exp_type in experiments
            # if exp_type != fisher
            # continue
            # end
            output_folder = joinpath("results", "generated_distributions", date_time, experiment_name(exp_type))
            create_folder(output_folder)
            println("\nExperiment is $(experiment_name(exp_type))")

            for (j, p) in enumerate(proportions)
                            for hardness in [1, 2, 3]
                                hardness_name = hardness_to_name(hardness)
                                if !has_proportion(exp_type) && j > 1
                                        continue
                                end
                                Random.seed!(1)
                                println("\nRunning $(experiment_name(exp_type)) experiment with dimension=$d")
                                specific_folder = "dim_$d"
                                if has_proportion(exp_type)
                                        println("    Proportion=$p")
                                        specific_folder *= "_prop_$p"
                                end
                                specific_output_folder = joinpath(output_folder, hardness_name, specific_folder)
                                create_folder(specific_output_folder)
                                fixed_spectrum = experiment_fixed_spectrum(exp_type, proportion=p, hardness=hardness)
                                rank = length(fixed_spectrum)
                                println("    Fixed spectrum size is $rank")
                                upper_bound_rank = 2 * rank
                                # ensemble = FDRControlSubspaceSelection.FisherEnsemble(fixed_spectrum, p)
                                if !has_proportion(exp_type)
                                        ensemble = experiment_struct(exp_type)(fixed_spectrum)
                                else
                                        ensemble = experiment_struct(exp_type)(fixed_spectrum, p)
                                end
                                (_, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
                                true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, upper_bound_rank)
                                results_unspecified_rank = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha)
                                results = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, rank_estimate=rank)
                                results_underestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, rank_estimate=Int(rank * 0.5))
                                results_overestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, rank_estimate=Int(1.5 * rank))
                                add_legend = false
                                if i == 1 && hardness == 1
                                    add_legend = true
                                end
                                plot_wrong_estimates(true_fdr, results_underestimate, results, results_overestimate, rank, specific_output_folder, add_legend=add_legend)
                                save_results(results, alpha, experiment_name(exp_type), specific_output_folder, true_fdr=true_fdr, name="exact", add_legend=add_legend)
                                save_results(results_unspecified_rank, alpha, experiment_name(exp_type), specific_output_folder, true_fdr=true_fdr, name="unspecified", add_legend=add_legend)
                            end
                        end
                end
        end
end

main()
