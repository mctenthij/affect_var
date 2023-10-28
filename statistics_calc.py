import numpy as np
from scipy import stats


def strfnumber(number):
    if np.abs(number) > 100:
        return "{:}".format(int(number))
    elif np.abs(number) > 10:
        return "{:.1f}".format(number)
    elif np.abs(number) > 0.1:
        return "{:.2f}".format(number)
    elif np.abs(number) >= 0.001:
        return "{:.3f}".format(number)
    else:
        return ("{" + ":.{}f".format(int(np.abs(np.floor(np.log10(number))))) + "}").format(number)


def strfint(value):
    return "{}".format(int(np.round(value)))


def strfpval(pvalue):
    return "{:.3f}".format(pvalue)[1:]


def parse_pval(pvalue):
    return "p<.001" if pvalue < 0.001 else "p={}".format(strfpval(pvalue))


def print_pval(pvalue):
    return "${}$".format(parse_pval(pvalue))


def print_median(cohort):
    return "$Mdn={}$".format(strfnumber(cohort.median()))


def print_mean(cohort):
    return "$M={}$".format(strfnumber(cohort.mean()))


def print_std(cohort):
    return "$SD={}$".format(strfnumber(cohort.std()))


def print_mean_and_std(cohort):
    return "{}, {}".format(print_mean(cohort), print_std(cohort))


def print_CI(data, perc=95, ntrvl_only=False):
    left, right = calc_CI_border(perc)
    if ntrvl_only:
        return "$[{}, {}]$".format(strfnumber(data.quantile(q=left)), strfnumber(data.quantile(q=right)))
    else:
        return "{}% CI $[{}, {}]$".format(perc, strfnumber(data.quantile(q=left)), strfnumber(data.quantile(q=right)))


def print_IQR(data):
    return "IQR $={} - {}$".format(strfnumber(data.quantile(q=0.25)), strfnumber(data.quantile(q=0.75)))


def print_pearson_correlation(C1, C2):
    r, pval = calc_pearson_correlation(C1, C2)
    return "$\rho={}, {}$".format(strfnumber(r), parse_pval(pval))


def calc_pearson_correlation(C1, C2):
    return stats.pearsonr(C1, C2)


def calc_spearman_correlation(data, **kwargs):
    return stats.spearmanr(data, **kwargs)


def calc_gaussian_KDE(data, bins):
    return stats.gaussian_kde(data)(bins)


def calc_histogram(data, bins, density=True):
    return np.histogram(data, bins=bins, density=density)[0]


def calc_CI_border(perc):
    alpha = (100 - perc) / 2
    return alpha / 100, 1 - alpha / 100


def calc_pooled_var(C1, C2):
    return np.sqrt(((C1.size - 1) * C1.var() + (C2.size - 1) * C2.var()) / (C1.size + C2.size - 2))


def calc_cohens_d(C1, C2):
    return np.abs(C1.mean() - C2.mean()) / calc_pooled_var(C1, C2)


def calc_ttest_df(C1, C2):
    p1 = (C1.var() / C1.size + C2.var() / C2.size) ** 2
    p2 = (C1.var() / C1.size) ** 2 / (C1.size - 1) + (C2.var() / C2.size) ** 2 / (C2.size - 1)
    return p1 / p2


def perform_ttest(C1, C2, strfy=True, **kwargs):
    if "equal_var" in kwargs:
        if kwargs["equal_var"]:
            print("Performing an independent t-test")
        else:
            print("Performing Welch's t-test")
    T = stats.ttest_ind(C1, C2, **kwargs)
    info = (strfint(calc_ttest_df(C1, C2)), strfnumber(T.statistic), parse_pval(T.pvalue), strfnumber(calc_cohens_d(C1, C2)))
    if strfy:
        return "$t({})={}, {}, d={}$".format(*info)
    else:
        return T


def perform_shapiro(data, strfy=True):
    W = stats.shapiro(data)
    if strfy:
        return "$W({:.0f})={}, {}$".format(data.size, strfnumber(W.statistic), parse_pval(W.pvalue))
    else:
        return W


def perform_kruskal(*args, strfy=True, **kwargs):
    KW = stats.kruskal(*args, **kwargs)
    if strfy:
        return "$H={}, {}$".format(strfnumber(KW.statistic), parse_pval(KW.pvalue))
    else:
        return KW


def check_MWU_kwargs(kwargs):
    if "alternative" in kwargs:
        if not kwargs["alternative"]:
            kwargs["alternative"] = "two-sided"
    else:
        kwargs["alternative"] = "two-sided"

    return kwargs


def calc_MWU_r(C1, C2, strfy=True, **kwargs):
    kwargs = check_MWU_kwargs(kwargs)
    r = 1 - 2 * stats.mannwhitneyu(C1, C2, **kwargs).statistic / (C1.size * C2.size)
    if strfy:
        return "r={}".format(strfnumber(r))
    else:
        return r


def perform_mann_whitney_u(C1, C2, strfy=True, **kwargs):
    kwargs = check_MWU_kwargs(kwargs)
    U = stats.mannwhitneyu(C1, C2, **kwargs)
    r = calc_MWU_r(C1, C2, strfy=strfy, **kwargs)
    if strfy:
        return "$U={}, {}, {}$".format(strfnumber(U.statistic), parse_pval(U.pvalue), r)
    else:
        return U, r


def perform_bartlett(*args, strfy=True):
    B = stats.bartlett(*args)
    if strfy:
        return "$T={}, {}$".format(strfnumber(B.statistic), parse_pval(B.pvalue))
    else:
        return B


def perform_levene(*args, strfy=True, **kwargs):
    L = stats.levene(*args, **kwargs)
    if strfy:
        return "$F={}, {}$".format(strfnumber(L.statistic), parse_pval(L.pvalue))
    else:
        return L


def perform_pearson(C1, C2, strfy=True, sgn=True):
    r, pval = stats.pearsonr(C1, C2)
    if strfy:
        if sgn:
            return "$r({:.0f})={}, {}$".format(C1.size + C2.size, strfnumber(r), parse_pval(pval))
        else:
            return "$r={}$".format(strfnumber(r))
    else:
        return r, pval


def perform_spearman(C1, C2, strfy=True, sgn=True, **kwargs):
    r, pval = stats.pearsonr(C1, C2, **kwargs)
    if strfy:
        if sgn:
            return "$\rho({:.0f})={}, {}$".format(C1.size + C2.size, parse_pval(r), parse_pval(pval))
        else:
            return "$\rho={}$".format(strfnumber(r))
    else:
        return r, pval


def parse_significance(p, ncomp=1):
    if p < 0.001 / ncomp:
        return "^{***}"
    elif p < 0.01 / ncomp:
        return "^{**}"
    elif p < 0.05 / ncomp:
        return "^*"
    else:
        return "ns"
