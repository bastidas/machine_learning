import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def predict_all(df, y, verbose=False):
    """
    Returns the predictions of nine different supervised machine learning algorithms.
    :param df: The input data frame should have two columns (nx2).
    :param y: The classifier should class should have length n.
    :param verbose: Print out diagnostic information with keyword.
    :return ml_pred, ml_name, x_mesh, y_mesh:
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier

    if verbose:
        import sklearn
        print('The scikit-learn version is {}.'.format(sklearn.__version__))

    method = [(GaussianNB(), "Naive Bayes"),
              (KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski'), "Nearest Neighbors"),
              (SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0, probability=True), "RBF SVM"),
              (SVC(kernel="linear", C=0.025), "Linear SVM"),
              (DecisionTreeClassifier(max_depth=5), "Decision Tree"),
              (RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), "Random Forest"),
              (AdaBoostClassifier(), "AdaBoost"),
              (MLPClassifier(alpha=1), "Neural Net"),
              (QuadraticDiscriminantAnalysis(), "QDA")]

    if verbose:
        print(df.head())

    # Split the data into a training set and test set.
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=.25, random_state=666)

    # Standard scalar standardize features by removing the mean and scaling to unit variance
    scaling = StandardScaler()
    scaling.fit(x_train)
    x_train_std = scaling.transform(x_train)
    x_test_std = scaling.transform(x_test)

    # this variable is the resolution of plots, small values take a long time to run
    mesh_resolution = 0.01
    overflow = 1.8  # make the mesh larger than the data range to make plots with more contour area
    x_mesh_range = np.arange(overflow*np.min(x_train_std[:, 0]), overflow*np.max(x_train_std[:, 0]), mesh_resolution)
    y_mesh_range = np.arange(overflow*np.min(x_train_std[:, 1]), overflow*np.max(x_train_std[:, 1]), mesh_resolution)
    x_mesh, y_mesh = np.meshgrid(x_mesh_range, y_mesh_range)
    mesh = np.array([x_mesh.ravel(), y_mesh.ravel()]).T

    ml_name = []
    ml_pred = []
    ml_score = []
    for m in method:
        if verbose:
            print("Training and fitting: ", m[1])
        tmp = m[0].fit(x_train_std, y_train)
        ml_score.append(tmp.score(x_test_std, y_test))
        prediction = tmp.predict(mesh)
        ml_pred.append(prediction.reshape(x_mesh.shape))
        ml_name.append(m[1])

    return ml_pred, ml_name, ml_score, x_mesh, y_mesh


def autoplot_pred(df, y, ml_pred, ml_name, ml_score, x_mesh, y_mesh, scaled=True):
    """
    Plot and save nine different machine learning classifier predictions.
    :param df: A two column data frame with labeled columns.
    :param y: A single column classifier.
    :param ml_pred: The sklearn predictions.
    :param ml_name: Labels for the kinds of sklearn prediction.
    :param x_mesh: The x mesh on which the prediction is made.
    :param y_mesh: The y mesh on which the prediction is made.
    :param scaled: Keyword to make scaled or non-scaled plot.
    :return: none
    """
    from sklearn.preprocessing import StandardScaler
    x = np.asarray(df)
    if scaled:
        scaling = StandardScaler()
        scaling.fit(x)
        x = scaling.transform(x)
        xmesh = x_mesh
        ymesh = y_mesh
        ex = [1.1*np.min(x[:, 0]), 1.1*np.max(x[:, 0])]
        ey = [1.1*np.min(x[:, 1]), 1.1*np.max(x[:, 1])]
    else:
        # get un-normalized data spaces
        xmesh = x_mesh * np.sqrt(np.var(x[:, 0], axis=0)) + np.mean(x[:, 0], axis=0)
        ymesh = y_mesh * np.sqrt(np.var(x[:, 1], axis=0)) + np.mean(x[:, 1], axis=0)
        ex = [np.min(x[:, 0]), 1.1*np.max(x[:, 0])]
        ey = [np.min(x[:, 1]), 1.1*np.max(x[:, 1])]
        axis_names = list(df.columns.values)
        xname = axis_names[0]
        yname = axis_names[1]

    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})
    fig = plt.figure(figsize=(9, 9))
    sn_colors = sns.color_palette()
    cm1 = plt.cm.get_cmap('ocean')
    tics = ['s', 'o', '^', 'p', '*', '.']
    for n in range(len(ml_pred)):
        ax = fig.add_subplot(331+n)
        plt.contourf(xmesh, ymesh, ml_pred[n], alpha=0.8, cmap=cm1)
        for tcolor, tstate, tmark in zip(sn_colors, range(len(np.unique(y))), tics):
            plt.scatter(x[y == tstate, 0], x[y == tstate, 1], marker=tmark, color=tcolor, lw=0.0, s=20)
        ax.set_xlim(ex)
        ax.set_ylim(ey)
        plt.title(ml_name[n])
        ax.text(ex[1] - .2, ey[0] + .2, ('%.2f' % ml_score[n]).lstrip('0'), size=15, horizontalalignment='right')
        if scaled:
            plt.axis('off')
        if not scaled:
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
    if not scaled:
        plt.tight_layout()
        fig.savefig('ml_classifiers.png')
    else:
        fig.savefig('ml_classifiers_normalized.png')
    plt.show()
    plt.close(fig)


