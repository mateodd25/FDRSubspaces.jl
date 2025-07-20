import FDRControlSubspaceSelection
using Plots
using Random
using LaTeXStrings

# Fix plotting issues on NixOS
ENV["GKSwstype"] = "nul"  # Disable problematic GKS socket connections
gr()  # Use GR backend which works better with NixOS

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
    d = 800
    mses = []
    fdrs = []
    ranks = [5, 20, 40]
    upper_bound_rank = ranks[argmax(ranks)] + 10
    rank = 0
    rank_minima = []
    for r in ranks
        fixed_spectrum = [1.19 for i in 1:r]
        # for j in 1:2
        #     fixed_spectrum[j] = 1.3
        # end
        # fixed_spectrum = [1.15+100.0*2.0^(-i+1) for i in 1:r]
        println("Fixed spectrum: ", fixed_spectrum)
        #fixed_endspectrum = [2 for i in 1:20]
        rank = r
        ensemble = FDRControlSubspaceSelection.WignerFactorEnsemble(fixed_spectrum)
        (true_signal, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
        alpha = 0.35
        true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, upper_bound_rank)
        true_mse = FDRControlSubspaceSelection.estimate_true_mse(ensemble, d, upper_bound_rank)
        push!(rank_minima, argmin(true_mse))
        println("Rank minima: ", argmin(true_mse))
        push!(fdrs, true_fdr)
        push!(mses, true_mse)
        println(true_fdr)
        println(true_mse)
    end

    colors = [12, 7, 2]
    lines = [:solid, :dash, :dot]
    println("Rank minima: ", rank_minima)
    r = rank
    alpha = .35
    i = 1
    plot(1:upper_bound_rank, fdrs[1][1:upper_bound_rank], label=L"True FDR $r = $"*string(ranks[i]), line = (4, lines[i]), color =colors[i], fg_legend = :transparent, legend_background_color = :transparent)
    i = 2
    while i <= length(ranks)
        r = ranks[i]
        plot!(1:upper_bound_rank, fdrs[i][1:upper_bound_rank], label=L"True FDR $r = $"*string(r), line = (4, lines[i]), color= colors[i])
        # plot!(1:upper_bound_rank, fdrs[3][1:upper_bound_rank], label=L"True FDR $r = 40$", line = (4, :dot), color = 2)
        i+= 1
    end
    plot!(ones(upper_bound_rank) * alpha, label=L"\alpha = "*string(alpha))
    xaxis!(L"Truncation rank $k$")
    savefig("results/fdr_vs_mse/fdrs"*string(r)*"-"*string(d)*".pdf")
    i = 1
    for r in ranks
        range = (max(rank_minima[i]-10,1)):(min(r+2,length(mses[i])))
        plot(range, mses[i][range], label=L"True MSE $r = $"*string(r), line = (4, :solid), color =12, fg_legend = :transparent, legend_background_color = :transparent, ylabel = "Mean Squared Error (MSE)", xlabel = L"Truncation rank $k$")
        plot!(rank_minima[i], seriestype="vline", label=L"Rank minimum $r = $"*string(r), line = (4, :solid), color=12, legend = false)
        xaxis!(L"Truncation rank $k$")
        savefig("results/fdr_vs_mse/mse"*string(r)*"-"*string(d)*".pdf")
        i += 1
    end
    # plot!(1:upper_bound_rank, mses[2][1:upper_bound_rank], label=L"True MSE $r = 20$", line = (4, :dash), color=7)
    # vline!(rank_minima[2], label=L"Rank minimum $r = 20$", line = (4, :dash), color=7, legend = false)
    # plot!(1:upper_bound_rank, mses[3][1:upper_bound_rank], label=L"True MSE $r = 40$", line = (4, :dot), color=2)
    # vline!(rank_minima[3], label=L"Rank minimum $r = 40$", line = (4, :dot), color=2, legend = false)

    # plot!(ones(upper_bound_rank) * 4, label=L"\alpha = 4")
    # savefig("results/fdr_vs_mse/mse"*string(r)*"-"*string(d)*".pdf")
end

main()
