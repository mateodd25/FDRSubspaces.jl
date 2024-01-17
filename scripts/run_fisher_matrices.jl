import FDRControlSubspaceSelection
using Plots
using Random
using LaTeXStrings
include("plot_utils.jl")

Random.seed!(1)

dimensions = [500]
proportions = [0.5, 1]
for (i, d) in enumerate(dimensions)
        for (j, p) in enumerate(proportions)
                # fixed_spectrum = [2 + 10 * 1.5^(-i + 1) for i in 1:20]
                fixed_spectrum = [22, 20]
                rank = length(fixed_spectrum)
            upper_bound_rank = 2 * rank
                # ensemble = FDRControlSubspaceSelection.FisherEnsemble(fixed_spectrum, p)
                ensemble = FDRControlSubspaceSelection.FisherFactorEnsemble(fixed_spectrum, p)
                (true_signal, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
                alpha = 0.05
                true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, upper_bound_rank)
                results = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, rank_estimate=rank, compute_spacings=true)
            results_underestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, rank_estimate=Int(rank * 0.5))
                results_overestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, rank_estimate=Int(1.5 * rank))
                add_legend = false
                if i == 1 && j == 1
                        add_legend = true
                end
                output_folder = "results/fisher/"
                create_folder(output_folder)
                plot_wrong_estimates(true_fdr, results_underestimate, results, results_overestimate, rank, d, ensemble.dimension_proportion, output_folder, add_legend=add_legend)
                save_results(results, alpha, "fisher", output_folder)
        end
end
