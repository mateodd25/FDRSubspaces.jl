import FDRControlSubspaceSelection
using Plots
using Random
using LaTeXStrings

fntsm = font("serif-roman", pointsize = 18)
fntlg = font("serif-roman", pointsize = 18)
default(
    titlefont = fntlg,
    guidefont = fntlg,
    tickfont = fntsm,
    legendfont = fntsm,
)

Random.seed!(1)
dimensions = [500, 2000]
for (i, d) in enumerate(dimensions)
    fixed_spectrum = [1 + 10 * 1.5^(-i + 1) for i in 1:20]
    rank = length(fixed_spectrum)
    upper_bound_rank = 2*rank
    ensemble = FDRControlSubspaceSelection.WignerEnsemble(fixed_spectrum)
    (true_signal, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
    alpha = 0.05
    true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, 2*rank)
    results = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, rank_estimate=rank)
    results_underestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, rank_estimate=Int(rank*.5))
    results_overestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha,rank_estimate= Int(1.5*rank))
    if i == 1
        plot(1:upper_bound_rank, true_fdr, label="True FDR", line = (4, :solid), color=12,  fg_legend = :transparent, legend_background_color = :transparent)
    else
        plot(1:upper_bound_rank, true_fdr, label="True FDR", legend = false, line = (4, :solid), color=12,  fg_legend = :transparent, legend_background_color = :transparent)
    end
    xaxis!(L"Subspace dimension $k$")
    plot!(1:upper_bound_rank, results_underestimate.fdr[1:upper_bound_rank], label=L"FDR estimate $\hat r = r/2$", line = (4, :dash), color=7)
    plot!(1:upper_bound_rank, results.fdr[1:upper_bound_rank], label=L"FDR estimate $\hat r = r$", line = (4, :dot), color=1)
    plot!(1:upper_bound_rank, results_overestimate.fdr[1:upper_bound_rank], label=L"FDR estimate $\hat{r} = 2r$", line = (4, :dashdot), color=9)

    savefig("results/wrong_rank/fdrs"*string(rank)*"-"*string(d)*".pdf")
end
