import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def plot_all(df, y, plots_per_page):
    """
    Plots all combinations of axis from data frame.
    :param df: data frame with named columns to plot.
    :param y: A classifier or target corresponding to df.
    :param plots_per_page: The number of plots on a page, either 4, 6, or 9.
    :return: none
    """
    x = np.asarray(df)
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})
    sn_colors = sns.color_palette()
    tics = ['s', 'o', '^', 'p', '*', '.']  # a tic for each classifier
    n = 0
    if plots_per_page == 9:
        base = 330
    if plots_per_page == 6:
        base = 320
    if plots_per_page == 4:
        base = 220
    count = 1
    initial = True
    axis_names = list(df.columns.values)
    ln = len(axis_names)
    while n < ln:
        i = 1
        while i < ln:
            if (count - 1) % plots_per_page == 0:
                if n > i:
                    break
                if not initial:
                    plt.tight_layout()
                    plt.show()
                initial = False
                if i != 0:
                    fig = plt.figure()
                count = 1
            if i > n:
                ax = fig.add_subplot(base + count)
                for tcolor, tstate, tmark in zip(sn_colors, range(len(np.unique(y))), tics):
                    plt.scatter(x[y == tstate, n], x[y == tstate, i], marker=tmark, color=tcolor, lw=0.0, s=20)
                ax.set_xlabel(axis_names[n])
                ax.set_ylabel(axis_names[i])
                count += 1
            i += 1
        n += 1
    plt.show()


def plot_3d(df, y, ylabels):
    """
    Visualize the first 3 dimensions of a data frame.
    :return: none
    """
    x = np.asarray(df)
    ny = range(len(np.unique(y)))
    axis_names = list(df.columns.values)
    sns_colors = sns.color_palette()
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})
    markers = ['s', 'o', '^', 'p']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in ny:
        ax.scatter(x[y == i, 0], x[y == i, 1], x[y == i, 2], label=ylabels[i], color=sns_colors[i],
                   marker=markers[i], s=50)
    leg = plt.legend(loc='best', shadow=False, scatterpoints=1, frameon=True)
    leg.get_frame().set_linewidth(0.5)
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_zlabel(axis_names[2])
    plt.show()
    fig.savefig('data_3d.png')
    plt.close(fig)


def plot_pca(x_pca, x_lda, y, ylabels):
    """
    Plot pca and lda components of the data.
    :return: none
    """
    fig = plt.figure()
    colors = sns.color_palette()
    ax1 = fig.add_subplot(211)
    for color, i, target_name, mark, in zip(colors, [0, 1, 2], ylabels, ['s', 'o', '^']):
        ax1.scatter(x_pca[y == i, 0], x_pca[y == i, 1], color=color, alpha=.8, lw=2, marker=mark, label=target_name)
    leg = plt.legend(loc='best', shadow=False, scatterpoints=1, frameon=True)
    leg.get_frame().set_linewidth(0.5)
    plt.title('2 axis PCA projection')
    ax2 = fig.add_subplot(212)
    for color, i, target_name, mark in zip(colors, [0, 1, 2], ylabels, ['s', 'o', '^']):
        ax2.scatter(x_lda[y == i, 0], x_lda[y == i, 1], alpha=.8, color=color, lw=2, marker=mark, label=target_name)
    leg = plt.legend(loc='best', shadow=False, scatterpoints=1, frameon=True)
    leg.get_frame().set_linewidth(0.5)
    plt.title('2 axis LDA projection')
    plt.tight_layout()
    plt.show()
    fig.savefig('PCA_LDA.png')
    plt.close(fig)

