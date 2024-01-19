import FDRControlSubspaceSelection
using Plots
using Random
using LaTeXStrings

fntsm = font("serif-roman", pointsize = 16)
fntlg = font("serif-roman", pointsize = 16)
default(
    titlefont = fntlg,
    guidefont = fntlg,
    tickfont = fntsm,
    legendfont = fntsm,
)
Random.seed!(1)
dimensions = [500, 800, 2000]
fixed_spectrum = [1+10*1.5^(-i+1) for i in 1:20]
rank = length(fixed_spectrum)
upper_bound_rank = 2*rank
ensemble = FDRControlSubspaceSelection.WignerEnsemble(fixed_spectrum)
for (i, d) in enumerate(dimensions)
    (true_signal, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
    alpha = 0.05
    results = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha)
    if i == 1
        scatter(1:(2*rank), results.eigenvalues[1:(2*rank)],seriestype=:scatter, label="Eigenvalues")
        yaxis!(L"Eigenvalues $\lambda_{k}$")
    else
        scatter(1:(2*rank), results.eigenvalues[1:(2*rank)],seriestype=:scatter, label="Eigenvalues", legend = false)
    end
    xaxis!(L"Index $k$")
    savefig("results/spacings/eigenvalue"*string(rank)*"-"*string(d)*".pdf")

    delta = 15
    if i == 1
        scatter((rank-delta):(rank+6), results.eigenvalues[(rank-delta):(rank+6)],seriestype=:scatter, label="Eigenvalues")
        yaxis!(L"Eigenvalues $\lambda_{k}$")
    else
        scatter((rank-delta):(rank+6), results.eigenvalues[(rank-delta):(rank+6)],seriestype=:scatter, label="Eigenvalues", legend = false)
    end
    xaxis!(L"Index $k$")
    savefig("results/spacings/zoomed_eigenvalue"*string(rank)*"-"*string(d)*".pdf")


    if i == 1
        scatter(1:(2*rank), results.spacings[1:(2*rank)], yscale = :log10, label="Spacings")
        yaxis!(L"Spacing $\lambda_{k} - \lambda_{k+1}$")
    else
        scatter(1:(2*rank), results.spacings[1:(2*rank)], yscale = :log10, label="Spacings", legend = false)
    end
    plot!(ones(2*rank) * results.threshold, label="Threshold")
    xaxis!(L"Index $k$")
    savefig("results/spacings/spacings"*string(rank)*"-"*string(d)*".pdf")
end
