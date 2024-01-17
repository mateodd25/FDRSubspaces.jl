using Plots
using JSON
using Dates
using LaTeXStrings
# include("../src/estimator.jl")
using FDRControlSubspaceSelection
# using FDRControlSubspaceSelection

function general_setup()
    fntsm = font("serif-roman", pointsize=16)
    fntlg = font("serif-roman", pointsize=16)
    default(
        titlefont=fntlg,
        guidefont=fntlg,
        tickfont=fntsm,
        legendfont=fntsm,
    )
    # gr()
    # fntsm = Plots.font(pointsize=12)
    # fntlg = Plots.font(pointsize=18)
    # default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)
end

function plot_wrong_estimates(csv_path, plot_file; lower_bound=1.0e-15, upper_bound=1.0)
    # TODO: Complete
    general_setup()
    losses = DataFrame(CSV.File(csv_path))
    plot()
    for col in names(losses)
        plot!(
            losses[!, col],
            yaxis=(:log10, [lower_bound, upper_bound]),
            label=col,
            line=(2, :solid),
            # legend = :bottomleft,
        )
    end
    xaxis!("Iteration count")
    yaxis!("Objective gap")
    savefig(plot_file)
end

function plot_eigenvalues(results, output_path, rank_upper_bound)
    rank_upper_bound = min(rank_upper_bound, length(results.eigenvalues))
    scatter(1:rank_upper_bound, results.eigenvalues[1:rank_upper_bound], yscale=:log10, label="Eigenvalues")
    yaxis!(L"Eigenvalues $\lambda_{k}$")
    xaxis!(L"Index $k$")
    savefig(output_path)
end
function plot_spacings(results, output_path, rank_upper_bound)
    rank_upper_bound = min(rank_upper_bound, length(results.spacings))
    scatter(1:rank_upper_bound, results.spacings[1:rank_upper_bound], yscale=:log10, label="Spacings")
    plot!(ones(rank_upper_bound) * results.threshold, label="Threshold")
    yaxis!(L"Spacing $\lambda_{k} - \lambda_{k+1}$")
    xaxis!(L"Index $k$")
    savefig(output_path)
end

function plot_fdr(results, output_path, rank_upper_bound; threshold=-1)
    rank_upper_bound = min(rank_upper_bound, length(results.fdr))
    plot(1:rank_upper_bound, results.fdr[1:rank_upper_bound], line=(3, :dot))
    yaxis!(L"FDR estimate$")
    xaxis!(L"Subspace dimension $k$")
if threshold > 0
        plot!(ones(rank_upper_bound) * threshold, label=L"\alpha = " * string(threshold))
    end
    savefig(output_path)
end

function plot_fds(csv_path, plot_file; lower_bound=1.0e-15, upper_bound=1.0)
    # TODO Complete
end

function generate_data_time_string()::String
    return Dates.format(now(), "yyyy-mm-dd-HH-MM-SS")
end

function create_folder(output_folder::String)
    if !isdir(output_folder)
        mkpath(output_folder)
    end
end

function plot_wrong_estimates(true_fdr, results_underestimate, results, results_overestimate, rank, dimension, dimension_proportion, output_folder; rank_upper_bound=-1, add_legend=true)
    general_setup()
    if rank_upper_bound == -1
        rank_upper_bound = 2 * rank
    end
    if add_legend
        plot(1:rank_upper_bound, true_fdr, label="True FDR", line=(3, :solid))
    else
        plot(1:rank_upper_bound, true_fdr, label="True FDR", legend=false, line=(3, :solid))
    end

    xaxis!(L"Subspace dimension $k$")
    yaxis!(L"FDR")
    plot!(1:rank_upper_bound, results_underestimate.fdr[1:rank_upper_bound], label=L"FDR estimate $\hat r = r/2$", line=(3, :dash))
    plot!(1:rank_upper_bound, results.fdr[1:rank_upper_bound], label=L"FDR estimate $\hat r = r$", line=(3, :dot))
    plot!(1:rank_upper_bound, results_overestimate.fdr[1:rank_upper_bound], label=L"FDR estimate $\hat{r} = 3r/2$", line=(3, :dashdot))

    savefig(joinpath(output_folder, string(rank) * "-" * string(dimension) * "-" * string(dimension_proportion) * ".pdf"))

end

function save_results(results::FDRControlSubspaceSelection.FDRResult, alpha::Float64, data_name::String, output_folder::String)
    general_setup()
    data = Dict(
        "databse" => data_name,
        "best_k" => results.best_k,
        "rank_estimate" => results.rank_estimate,
        "threshold" => results.threshold,
        "eigenvalues" => results.eigenvalues,
        "spacings" => results.spacings,
        "fdr" => results.fdr)
    joinpath(output_folder, "results.json") |> (path -> write(path, JSON.json(data)))
    rank_upper_bound = 5 * results.rank_estimate
    plot_eigenvalues(results, joinpath(output_folder, "eigenvalues.pdf"), rank_upper_bound)
    plot_fdr(results, joinpath(output_folder, "fdr.pdf"), rank_upper_bound, threshold=alpha)
    plot_spacings(results, joinpath(output_folder, "spacings.pdf"), rank_upper_bound)
end
