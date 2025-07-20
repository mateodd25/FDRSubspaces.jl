import FDRControlSubspaceSelection
using Plots
using Random
using LaTeXStrings

# Fix plotting issues on NixOS
ENV["GKSwstype"] = "nul"  # Disable problematic GKS socket connections
gr()  # Use GR backend which works better with NixOS

fntsm = font("serif-roman", pointsize=16)
fntlg = font("serif-roman", pointsize=16)
default(
    titlefont=fntlg,
    guidefont=fntlg,
    tickfont=fntsm,
    legendfont=fntsm,
)


Random.seed!(1)
# dimensions = [500, 2000]
dimensions = [500]
for (i, d) in enumerate(dimensions)
    fixed_spectrum = [1 + 10 * 1.5^(-i + 1) for i in 1:20]
    rank = length(fixed_spectrum)
    upper_bound_rank = 2 * rank
    ensemble = FDRControlSubspaceSelection.WishartFactorEnsemble(fixed_spectrum, 0.5)
    (true_signal, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
    alpha = 0.05
    true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, upper_bound_rank)
    results = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, rank)
    results_underestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, Int(rank * 0.5))
    results_overestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, Int(1.5 * rank))
    if i == 1
        plot(1:upper_bound_rank, true_fdr, label="True FDR", line=(3, :solid))
    else
        plot(1:upper_bound_rank, true_fdr, label="True FDR", legend=false, line=(3, :solid))
    end
    xaxis!(L"Subspace dimension $k$")
    plot!(1:upper_bound_rank, results_underestimate.fdr[1:upper_bound_rank], label=L"FDR estimate $\hat r = r/2$", line=(3, :dash))
    plot!(1:upper_bound_rank, results.fdr[1:upper_bound_rank], label=L"FDR estimate $\hat r = r$", line=(3, :dot))
    plot!(1:upper_bound_rank, results_overestimate.fdr[1:upper_bound_rank], label=L"FDR estimate $\hat{r} = 2r$", line=(3, :dashdot))

    savefig("results/wrong_rank_asym/fdrs" * string(rank) * "-" * string(d) * "-" * string(ensemble.dimension_proportion) * ".pdf")
end
