using Plots
using JSON
using Dates
using LaTeXStrings
# include("../src/estimator.jl")
using FDRControlSubspaceSelection
# using FDRControlSubspaceSelection

function general_setup()
    gr()
    fntsm = font("serif-roman", pointsize=18)
    fntlg = font("serif-roman", pointsize=18)
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

function plot_eigenvalues(results, output_path, rank_upper_bound; add_legend=true)
    rank_upper_bound = min(rank_upper_bound, length(results.eigenvalues))
    println("The upper bound is $rank_upper_bound")
    scatter(
        1:rank_upper_bound,
        results.eigenvalues[1:rank_upper_bound],
        label="Eigenvalues",
        legend=add_legend,
        markerstrokewidth=0,
        markersize=4,
        fg_legend = :transparent, legend_background_color = :transparent)
    yaxis!(L"Eigenvalues $\lambda_{k}$")
    xaxis!(L"Index $k$")
    savefig(output_path)
end

function plot_spacings(results, output_path, rank_upper_bound; add_legend=true)
    rank_upper_bound = min(rank_upper_bound, length(results.spacings))
    scatter(1:rank_upper_bound, results.spacings[1:rank_upper_bound], yscale=:log10, label="Spacings", legend=add_legend, markerstrokewidth=0, markersize=4, fg_legend = :transparent, legend_background_color = :transparent)
    plot!(ones(rank_upper_bound) * results.threshold, label="Threshold")
    yaxis!(L"Spacing $\lambda_{k} - \lambda_{k+1}$")
    xaxis!(L"Index $k$")
    savefig(output_path)
end

function plot_fdr(results, output_path, rank_upper_bound; threshold=-1, true_fdr::Union{Vector{Float64},Nothing}=nothing, add_legend=true)
    rank_upper_bound = min(rank_upper_bound, length(results.fdr))

    if !isnothing(true_fdr)
        rank_upper_bound = min(length(true_fdr), rank_upper_bound)
        plot(1:rank_upper_bound, true_fdr[1:rank_upper_bound], line=(4, :solid), label="True FDR", legend=add_legend, color=12, fg_legend = :transparent, legend_background_color = :transparent)
        plot!(1:rank_upper_bound, results.fdr[1:rank_upper_bound], line=(4, :dot), label="Estimated FDR", color=1)
    else
        plot(1:rank_upper_bound, results.fdr[1:rank_upper_bound], line=(4, :dot), label="Estimated FDR", legend=add_legend, color=1, fg_legend = :transparent, legend_background_color = :transparent)
    end
    yaxis!("FDR")
    xaxis!(L"Subspace dimension $k$")
    if threshold > 0
        plot!(ones(rank_upper_bound) * threshold, label=L"\alpha = " * string(threshold), color=2)
    end
    savefig(output_path)
end

function plot_fds(csv_path, plot_file; lower_bound=1.0e-15, upper_bound=1.0)
    # TODO Complete
end

function plot_empirical_density(eigenvalues::Vector{Float64}, output_path::String; is_singular=false)
    histogram(eigenvalues[7:length(eigenvalues)], normalize=:pdf, bins=min(trunc(Int, length(eigenvalues) / 3), 200),linecolor = :transparent, legend=false)
    if is_singular
        xaxis!(L"Singular value $\sigma$")
    else
        xaxis!(L"Eigenvalue $\lambda$")
    end
    yaxis!("Normalized frequency")
    savefig(output_path)
end

function generate_data_time_string()::String
    return Dates.format(now(), "yyyy-mm-dd-HH-MM-SS")
end

function create_folder(output_folder::String)
    if !isdir(output_folder)
        mkpath(output_folder)
    end
end

function plot_wrong_estimates(true_fdr, results_underestimate, results, results_overestimate, rank, output_folder; rank_upper_bound=-1, add_legend=true, threshold=-1)
    general_setup()
    if rank_upper_bound == -1
        rank_upper_bound = 2 * rank
    end


    if !isnothing(true_fdr)
        if add_legend
            plot(1:rank_upper_bound, true_fdr, label="True FDR", yaxis =:log,  line=(4, :solid), color=12)
        else
            plot(1:rank_upper_bound, true_fdr, label="True FDR", yaxis = :log, legend=false, line=(4, :solid), color=12)
        end
        plot!(1:rank_upper_bound, results.fdr[1:rank_upper_bound], label="Default rank estimate", line=(4, :dot),color=1, fg_legend = :transparent, legend_background_color = :transparent )
    else
        plot(1:rank_upper_bound, results.fdr[1:rank_upper_bound], yaxis =:log, label="Default rank estimate", line=(4, :dot),color=1, fg_legend = :transparent, legend_background_color = :transparent)
    end
    yaxis!("FDR")
    plot!(1:rank_upper_bound, results_underestimate.fdr[1:rank_upper_bound], label="Lower rank estimate", line=(4, :dash), color=7)
    plot!(1:rank_upper_bound, results_overestimate.fdr[1:rank_upper_bound], label="Higher rank estimate", line=(4, :dashdot), color=9)

    if threshold > 0
        plot!(ones(rank_upper_bound) * threshold, label=L"\alpha = " * string(threshold), color=2)
    end
    xaxis!(L"Subspace dimension $k$")
    savefig(joinpath(output_folder, "different_rank_estimates.pdf"))

end

function save_results(results::FDRControlSubspaceSelection.FDRResult, alpha::Float64, data_name::String, output_folder::String; true_fdr::Union{Vector{Float64},Nothing}=nothing, name="", add_legend=true, rank_upper_bound=60)
    general_setup()
    name = (name == "" ? "" : "_" * name)
    data = Dict(
        "databse" => data_name,
        "best_k" => results.best_k,
        "rank_estimate" => results.rank_estimate,
        "threshold" => results.threshold,
        "eigenvalues" => results.eigenvalues,
        "spacings" => results.spacings,
        "fdr" => results.fdr)
    joinpath(output_folder, "results" * name * ".json") |> (path -> write(path, JSON.json(data)))
    if length(results.eigenvalues) > 0
        plot_eigenvalues(results, joinpath(output_folder, "eigenvalues.pdf"), rank_upper_bound, add_legend=add_legend)
        plot_spacings(results, joinpath(output_folder, "spacings.pdf"), rank_upper_bound, add_legend=add_legend)
        plot_empirical_density(results.eigenvalues, joinpath(output_folder, "empirical_density.pdf"))
    end
    plot_fdr(results, joinpath(output_folder, "fdr" * name * ".pdf"), rank_upper_bound, threshold=alpha, true_fdr=true_fdr, add_legend=add_legend)
end
