using Plots
using CSV

function general_setup()
    gr()
    fntsm = Plots.font(pointsize = 12)
    fntlg = Plots.font(pointsize = 18)
    default(titlefont = fntlg, guidefont = fntlg, tickfont = fntsm, legendfont = fntsm)
end

function plot_wrong_estimates(csv_path, plot_file; lower_bound = 1.0e-15, upper_bound = 1.0)
    # TODO: Complete
    general_setup()
    losses = DataFrame(CSV.File(csv_path))
    plot()
    for col in names(losses)
        plot!(
            losses[!, col],
            yaxis = (:log10, [lower_bound, upper_bound]),
            label = col,
            line = (2, :solid),
            # legend = :bottomleft,
        )
    end
    xaxis!("Iteration count")
    yaxis!("Objective gap")
    savefig(plot_file)
end

function plot_eigenvalues(csv_path, plot_file; lower_bound = 1.0e-15, upper_bound = 1.0)
    # TODO Complete
end

function plot_spacings(csv_path, plot_file; lower_bound = 1.0e-15, upper_bound = 1.0)
    # TODO Complete
end

function plot_fdrs(csv_path, plot_file; lower_bound = 1.0e-15, upper_bound = 1.0)
    # TODO Complete
end

function plot_fds(csv_path, plot_file; lower_bound = 1.0e-15, upper_bound = 1.0)
    # TODO Complete
end
