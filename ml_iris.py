import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ml_plot_tools import plot_pca
from ml_plot_tools import plot_3d
from ml_plot_tools import plot_all
from ml_classifier_methods import predict_all
from ml_classifier_methods import autoplot_pred

iris = datasets.load_iris()

verbose = True
if verbose:
    print(iris.keys())
    print(iris.feature_names)  # there are four feature names in the data, so the data has at least 4 dimensions
    print(iris.target_names)

# make the data into a data frame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
X = iris.data
y = iris.target

# plot the data in 3d
plot_3d(iris_df, y, iris.target_names)
# plot all columns against each other
plot_all(iris_df, y, 6)

# run pca and lda
pca = PCA(n_components=3)
X_pca = pca.fit(X).transform(X)
lda = LinearDiscriminantAnalysis(n_components=3)
X_lda = lda.fit(X, y).transform(X)
if verbose:
    print('The variance ratio explained by the first 3 LDA components: %s' % str(pca.explained_variance_ratio_))
    print('The variance ratio explained by the first 3 LDA components: %s' % str(lda.explained_variance_ratio_))

plot_pca(X_pca, X_lda, y, iris.target_names)

# run a bunch of classifiers on just two data columns
selected_properties = np.asarray([X[:, 2], X[:, 3]])
iris_df = pd.DataFrame(selected_properties.T, columns=[iris.feature_names[2], iris.feature_names[3]])
mlm, mln, mls, x_imesh, y_imesh = predict_all(iris_df, y, verbose=True)
autoplot_pred(iris_df, y, mlm, mln, mls, x_imesh, y_imesh, scaled=False)

# run a bunch of classifiers on the two principal axis returned from pca
sp = np.asarray([X_pca[:, 0], X_pca[:, 1]]).T
iris_df = pd.DataFrame(sp, columns=["pca axis 1", "pca axis 2"])
mlm, mln, mls, x_imesh, y_imesh = predict_all(iris_df, y, verbose=True)
autoplot_pred(iris_df, y, mlm, mln, mls, x_imesh, y_imesh, scaled=True)
