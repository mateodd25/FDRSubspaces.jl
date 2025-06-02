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

function main()
    d = 500
    mses = []
    fdrs = []
    ranks = [5, 20, 40]
    upper_bound_rank = ranks[3] + 10
    rank = 0
    for r in ranks
        fixed_spectrum = [1.2 for i in 1:r]
        for j in 1:2
            fixed_spectrum[j] = 2.0
        end
        #fixed_endspectrum = [2 for i in 1:20]
        rank = r
        ensemble = FDRControlSubspaceSelection.WignerEnsemble(fixed_spectrum)
        (true_signal, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
        alpha = 0.05
        true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, upper_bound_rank)
        true_mse = FDRControlSubspaceSelection.estimate_true_mse(ensemble, d, upper_bound_rank)
        push!(fdrs, true_fdr)
        push!(mses, true_mse)
        println(true_fdr)
        println(true_mse)
    end

    r = rank
    alpha = .2
    plot(1:upper_bound_rank, fdrs[1][1:upper_bound_rank], label=L"True FDR $r = 5$", line = (4, :solid), color =12, fg_legend = :transparent, legend_background_color = :transparent)
    plot!(1:upper_bound_rank, fdrs[2][1:upper_bound_rank], label=L"True FDR $r = 20$", line = (4, :dash), color= 7)
    plot!(1:upper_bound_rank, fdrs[3][1:upper_bound_rank], label=L"True FDR $r = 40$", line = (4, :dot), color = 2)
    plot!(ones(upper_bound_rank) * 0.2, label=L"\alpha = 0.2")
    xaxis!(L"Truncation rank $k$")
    savefig("results/fdr_vs_mse/fdrs"*string(r)*"-"*string(d)*".pdf")
    plot(1:upper_bound_rank, mses[1][1:upper_bound_rank], label=L"True MSE $r = 5$", line = (4, :solid), color =12, fg_legend = :transparent, legend_background_color = :transparent, yscale = :log10, ylabel = L"Mean Squared Error (MSE)", xlabel = L"Truncation rank $k$")
    plot!(1:upper_bound_rank, mses[2][1:upper_bound_rank], label=L"True MSE $r = 20$", line = (4, :dash), color=7)
    plot!(1:upper_bound_rank, mses[3][1:upper_bound_rank], label=L"True MSE $r = 40$", line = (4, :dot), color=2)

    # plot!(ones(upper_bound_rank) * 4, label=L"\alpha = 4")
    xaxis!(L"Truncation rank $k$")
    savefig("results/fdr_vs_mse/mse"*string(r)*"-"*string(d)*".pdf")

end

main()
