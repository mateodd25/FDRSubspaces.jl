import FDRControlSubspaceSelection
using Plots

function main()
    d = 100
    fixed_spectrum = [4.0, 3.0]
    ensemble = FDRControlSubspaceSelection.WignerEnsemble(fixed_spectrum)
    true_signal, noisy_matrix = FDRControlSubspaceSelection.true_and_noisy_matrix(ensemble, d)
    alpha = 0.05
    results = FDRControlSubspaceSelection.control_FDR(noisy_matrix, alpha)
    true_fdr = FDRControlSubspaceSelection.estimate_true_fdr(ensemble, d, 8*length(fixed_spectrum))
    asymptotic_fdr = FDRControlSubspaceSelection.compute_limiting_fdr(ensemble, d, 8*length(fixed_spectrum))
end
