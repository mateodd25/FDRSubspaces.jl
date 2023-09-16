import FDRControlSubspaceSelection
using Plots
using Random

Random.seed!(1)
# function main()
d = 1000
fixed_spectrum = [1+10*1.4^(-i+1) for i in 1:20]
#fixed_spectrum = [2 for i in 1:20]
rank = length(fixed_spectrum)
ensemble = FDRControlSubspaceSelection.WignerEnsemble(fixed_spectrum)
(true_signal, noisy_matrix) = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
    alpha = 0.05
    true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, 2*rank)
    println(true_fdr)
    #asymptotic_fdr = FDRControlSubspaceSelection.compute_limiting_fdr(ensemble, d, 2*rank)
    #println(asymptotic_fdr)
    results = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha)
println(results.fdr)
scatter(1:(2*rank), results.eigenvalues[1:(2*rank)],seriestype=:scatter, label="Eigenvalues")
savefig("eigenvalues.png")
scatter(1:(2*rank), results.spacings[1:(2*rank)], yscale = :log10, label="Spacings")
plot!(ones(2*rank) * results.threshold, label="Threshold")
savefig("spacings.png")
results_underestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, Int(rank*.5))
results_overestimate = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, Int(1.5*rank))
plot(1:40, results.fdr[1:40], label="estimate")
plot!(1:40, true_fdr, label="true")
plot!(1:40, results_underestimate.fdr[1:40], label="under_estimate")
plot!(1:40, results_overestimate.fdr[1:40], label="over_estimate")
savefig("fdrs.png")
# end

# main()
