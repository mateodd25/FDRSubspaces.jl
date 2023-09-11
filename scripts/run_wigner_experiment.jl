import FDRControlSubspaceSelection
using Plots

# function main()
    d = 400
    fixed_spectrum = [1+10*1.5^(-i+1) for i in 1:20]
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
    scatter(1:length(results.spacings), results.spacings)
    savefig("spacings.png")
    results_wrong_rank = FDRControlSubspaceSelection.control_fdr(noisy_matrix, alpha, 2*rank)
    plot(1:40, results.fdr[1:40], label="estimate")
    plot!(1:40, true_fdr, label="true")
    plot!(1:40, results_wrong_rank.fdr[1:40], label="over_estimate")
    savefig("fdrs.png")
# end

# main()
