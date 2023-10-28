import numpy as np

from statistics_calc import calc_CI_border, calc_gaussian_KDE, calc_histogram


def boxplot(ax, data, loc, color=(0, 0, 0), perc=95, box=True, extreme=True, dash="mean"):
    if type(perc) not in (int, float) or perc < 50 or perc > 100:
        raise(ValueError("perc must be an integer between 50 and 100"))
        
    if dash == "mean":
        ax.vlines(data.mean(), loc - 0.25, loc + 0.25, lw=5, color=color, zorder=3)
    elif dash == "median":
        ax.vlines(data.median(), loc - 0.25, loc + 0.25, lw=5, color=color, zorder=3)
    else:
        raise(NotImplementedError)
        
    left, right = calc_CI_border(perc)
    
    if box:
        ax.fill_between([data.quantile(q=0.25), data.quantile(q=0.75)], loc - 0.15, loc + 0.15,
                        facecolor=(*color, 0.4), edgecolor=color, lw=3, zorder=3)
        ax.plot([data.quantile(q=left), data.quantile(q=right)], [loc, loc], color=color, 
                lw=0, marker="|", ms=12, mew=3, zorder=3)

        ax.plot([data.quantile(q=left), data.quantile(q=0.25)], [loc, loc], color=color, marker="", zorder=3)
        ax.plot([data.quantile(q=0.75), data.quantile(q=right)], [loc, loc], color=color, marker="", zorder=3)
    else:
        ax.plot([data.quantile(q=left), data.quantile(q=right)], [loc, loc], color=color, 
                marker="|", ms=12, mew=3, zorder=3)
    
    if extreme:
        lower = data.values[data < data.quantile(q=left)]
        upper = data.values[data > data.quantile(q=right)]

        ax.scatter(lower, loc * np.ones(lower.shape), marker=".", color=color, zorder=3)
        ax.scatter(upper, loc * np.ones(upper.shape), marker=".", color=color, zorder=3)
    
    return ax


def bar_and_kde(ax, data, left, right, step=0.1, kde_step=0.01, align="edge", color=(0, 0, 0)):
    bins = np.arange(left, right + step, step)
    ax.bar(bins[:-1], calc_histogram(data, bins), align=align, width=step,
           lw=2, facecolor=(*color, 0.1), edgecolor=color, zorder=2)
    
    kde_bins = np.arange(left, right + kde_step, kde_step)
    ax.plot(kde_bins, calc_gaussian_KDE(data, kde_bins), color=color, zorder=3)
    
    return ax


def log_bar(ax, data, left, right, align="edge", color=(0, 0, 0), num_bins=100):
    bins = np.logspace(np.log10(left), np.log10(right), num_bins)
    bin_width = bins[1:] - bins[:-1]
    ax.bar(bins[:-1], calc_histogram(data, bins) / bin_width, align=align, width=bin_width,
           lw=2, facecolor=(*color, 0.1), edgecolor=color, zorder=2)
    
    return ax
