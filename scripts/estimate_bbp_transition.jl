import FDRControlSubspaceSelection
using ArgParse
using Plots
using Random
using Match

# Fix plotting issues on NixOS
ENV["GKSwstype"] = "nul"  # Disable problematic GKS socket connections
gr()  # Use GR backend which works better with NixOS

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


function parse_cli()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--experiments"
        help = "Experiments to run separated by commas. If empty, it runs everything."
        arg_type = Union{String,Nothing}
        required = false
        default = nothing
    end
    return parse_args(s)
end


function main()
    parsed_args = parse_cli()
    experiments = instances(ExperimentType)
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("   $arg  =>  $val")
    end
    if !isnothing(parsed_args["experiments"])
        experiments = [name_to_experiment(string(instance)) for instance in split(parsed_args["experiments"],",")]
    end
    for exp_type in experiments
        if has_proportion(exp_type)
            for proportion in [0.5, 1.0]
                ensemble = experiment_struct(exp_type)([], proportion)
                s = FDRControlSubspaceSelection.estimate_bbp_transition_point(ensemble)
                println("    $exp_type and $proportion  =>  $s")
            end
        else
            ensemble = experiment_struct(exp_type)([])
            s = FDRControlSubspaceSelection.estimate_bbp_transition_point(ensemble)
            println("    $exp_type  =>  $s")
        end
    end
end

main()
