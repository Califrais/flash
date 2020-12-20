import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="",
            **kwargs):
    """Creates a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=13)
    ax.set_yticklabels(row_labels, fontsize=13)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"), threshold=None, **textkw):
    """A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.

    data
        Data used to annotate.  If None, the image's data is used.  Optional.

    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.

    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.

    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.

    **textkw
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def gompertz_pdf(t, shape: float = .1, scale: float = .001):
    """Probability density function of a Gompertz random variable.

    Parameters
    ----------
    scale : `float`, default=.5
        Scaling parameter

    shape : `float`, default=.5
        Shape parameter

    t : `float`
        Time at which the density function is returned
    """
    return scale * shape * np.exp(scale + shape * t - scale * np.exp(shape * t))


def gompertz_survival(t, shape: float = .1, scale: float = .001):
    """Survival function of a Gompertz random variable.

    Parameters
    ----------
    scale : `float`, default=.5
        Scaling parameter

    shape : `float`, default=.5
        Shape parameter

    t : `float`
        Time at which the survival function is returned
    """
    return np.exp(- scale * (np.exp(shape * t) - 1))


def plot_history(learner, name, ax=None, **kwargs):
    """Plot an element history evolution through iterations

    Parameters
    ----------
    learner : `ligths.base.base.Learner`
        A base learner

    name : `str`
        Name of the element to be plotted

    ax : `matplotlib.axes.Axes`, default=None
        Axe instance to which the graph is plotted.  If not provided, use
        current axes or create a new one

    **kwargs
        All other arguments are forwarded to the plot function of the pandas
        DataFrame
    """
    history_keys = learner.get_history_keys()
    if name not in history_keys:
        raise ValueError("`%s` not stored in history, "
                         "must be in %s" % (name, history_keys))

    if not ax:
        ax = plt.gca()

    history = learner.get_history(name)
    if isinstance(history[0], float):
        history = pd.DataFrame(history)
    else:
        history = pd.DataFrame.from_records(history)

    n_iter = learner.get_history("n_iter")
    history.index = n_iter

    history.plot(ax=ax, **kwargs)


def visualize_vect_learning(learner, name, symbol, true_coeffs, legend_est,
                            legend_true):
    """Plots learning for a given parameter vector : objective and relative
    objective function, as well as evolution of estimators through iterations

    Parameters
    ----------
    learner : `ligths.base.base.Learner`
        A base learner

    name : `str`
        Name of the element to be plotted

    symbol : `str`
        Symbol of the element to be plotted

    true_coeffs : `np.ndarray`
        True coefficient vector to be estimated

    legend_est : `list`
        Names of each estimator coefficients

    legend_true : `list`
        Names of each true parameter coefficients
    """
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(121)
    fs = 18
    plt.title("Objective convergence", fontsize=fs + 2)
    plt.xlabel('iterations', fontsize=fs + 2)
    plt.ylabel('Obj', fontsize=fs + 2)
    plt.xticks(fontsize=fs), plt.yticks(fontsize=fs)
    plot_history(learner, name="obj", ax=ax, color='b', legend=False)

    ax = fig.add_subplot(122)
    plt.title("Relative objective convergence", fontsize=fs + 2)
    plt.xlabel('iterations', fontsize=fs + 2)
    plt.ylabel('Rel obj', fontsize=fs + 2)
    plt.xticks(fontsize=fs), plt.yticks(fontsize=fs)
    plot_history(learner, name="rel_obj", ax=ax, color='r', logy=True,
                 legend=False)
    fig.tight_layout()

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    plt.title("%s learning" % symbol, fontsize=fs + 2)
    plt.xlabel('iterations', fontsize=fs + 2)
    plt.xticks(fontsize=fs), plt.yticks(fontsize=fs)
    cm = 'Dark2'
    plot_history(learner, name=name, ax=ax, colormap=cm, alpha=.8)

    legend1 = ax.legend(legend_est, loc='center right',
                        bbox_to_anchor=(-0.1, 0.5), fontsize=fs)
    plt.gca().add_artist(legend1)

    last_iter = learner.get_history("n_iter")[-1]
    data = np.concatenate((true_coeffs, true_coeffs), axis=1).T
    df_true_coeffs = pd.DataFrame(data=data, index=[0, last_iter])
    df_true_coeffs.plot(ax=ax, colormap=cm, linestyle=':')

    lines = plt.gca().get_lines()
    to = len(true_coeffs) + 1
    plt.legend([lines[i] for i in range(to - 1, 2 * to - 2)], legend_true,
               loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs)

    fig.tight_layout()
    plt.show()
