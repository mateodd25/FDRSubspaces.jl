import FDRControlSubspaceSelection
using Plots
using Random
using LaTeXStrings

# Fix plotting issues on NixOS
ENV["GKSwstype"] = "nul"  # Disable problematic GKS socket connections
gr()  # Use GR backend which works better with NixOS

fntsm = font("serif-roman", pointsize = 18)
fntlg = font("serif-roman", pointsize = 18)
default(
    titlefont = fntlg,
    guidefont = fntlg,
    tickfont = fntsm,
    legendfont = fntsm,
)

Random.seed!(1)
d = 1000
ranks = [20, 40]
upper_bound_rank = 80
fds = []
fdrs = []
rank = 0
for r in ranks
    fixed_spectrum = [1.52 for i in 1:r]
    rank = r
    ensemble = FDRControlSubspaceSelection.WignerEnsemble(fixed_spectrum)
    (true_signal, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
    alpha = 0.2
    true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, upper_bound_rank)
    push!(fds, [fdr * k for (k, fdr) in enumerate(true_fdr)])
    push!(fdrs, true_fdr)
end
r = rank
alpha = .2
plot(1:upper_bound_rank, fdrs[1][1:upper_bound_rank], label=L"True FDR $r = 20$", line = (4, :solid), color =12, fg_legend = :transparent, legend_background_color = :transparent)
plot!(1:upper_bound_rank, fdrs[2][1:upper_bound_rank], label=L"True FDR $r = 40$", line = (4, :dot), color = 7)
plot!(ones(upper_bound_rank) * 0.2, label=L"\alpha = 0.2", color=2)
xaxis!(L"Subspace dimension $k$")
savefig("results/fdr_vs_fd/fdrs"*string(r)*"-"*string(d)*".pdf")
plot(1:upper_bound_rank, fds[1][1:upper_bound_rank], label=L"True FD $r = 20$", line = (4, :solid), color =12, fg_legend = :transparent, legend_background_color = :transparent)
plot!(1:upper_bound_rank, fds[2][1:upper_bound_rank], label=L"True FD $r = 40$", line = (4, :dot), color = 7)
plot!(ones(upper_bound_rank) * 4, label=L"\alpha = 4", color=2)
xaxis!(L"Subspace dimension $k$")
savefig("results/fdr_vs_fd/fds"*string(r)*"-"*string(d)*".pdf")
