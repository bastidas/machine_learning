"""
Classifies three data sets (moons, circles, blobs) using a deep neural net.
"""

import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from sklearn.model_selection import train_test_split

log_dir = "/tmp/test"
blob_log_dir = log_dir + "/blob"
moon_log_dir = log_dir + "/moon"
circle_log_dir = log_dir + "/circle"
start_from_scratch = True
if start_from_scratch:
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)

random_seed = np.random.randint(0, 1000)
print("The random seed:", random_seed)
np.random.seed(random_seed)
n_points = 500
circle = datasets.make_circles(n_samples=n_points, factor=.4, noise=.1)
moon = datasets.make_moons(n_samples=n_points, noise=.1)
blob = datasets.make_blobs(n_samples=n_points, random_state=random_seed)
moon_train_data, moon_test_data, moon_train_target, moon_test_target = \
    train_test_split(moon[0], moon[1], test_size=.25, random_state=random_seed)
circle_train_data, circle_test_data, circle_train_target, circle_test_target = \
    train_test_split(circle[0], circle[1], test_size=.25, random_state=random_seed)
blob_train_data, blob_test_data, blob_train_target, blob_test_target = \
    train_test_split(blob[0], blob[1], test_size=.25, random_state=random_seed)


def input_fn_(data_set, target_set):
    data_feat_cols = {'a': tf.constant(data_set[:, 0]), 'b': tf.constant(data_set[:, 1])}
    target_cols = tf.constant(target_set)
    return data_feat_cols, target_cols

feature_cols = [tf.contrib.layers.real_valued_column("a"), tf.contrib.layers.real_valued_column("b")]


blob_classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols, hidden_units=[32, 64, 32],
                                                 n_classes=len(np.unique(blob[1])), model_dir=blob_log_dir)
blob_classifier.fit(input_fn=lambda: input_fn_(blob_train_data, blob_train_target), steps=500)
evaluation = blob_classifier.evaluate(input_fn=lambda: input_fn_(blob_test_data, blob_test_target), steps=1)
loss_score = evaluation["loss"]
accuracy_score = evaluation["accuracy"]
print("Blob loss: {0:f}".format(loss_score))
print('Blob accuracy: {0:f}'.format(accuracy_score))

moon_classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols, hidden_units=[256, 512, 256],
                                                 n_classes=len(np.unique(moon[0])), model_dir=moon_log_dir)
moon_classifier.fit(input_fn=lambda: input_fn_(moon_train_data, moon_train_target), steps=1500)
evaluation = moon_classifier.evaluate(input_fn=lambda: input_fn_(moon_test_data, moon_test_target), steps=1)
loss_score = evaluation["loss"]
accuracy_score = evaluation["accuracy"]
print("Moon loss: {0:f}".format(loss_score))
print('Moon accuracy: {0:f}'.format(accuracy_score))

circle_classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols, hidden_units=[256, 512, 256],
                                                   n_classes=len(np.unique(circle[1])), model_dir=circle_log_dir)
circle_classifier.fit(input_fn=lambda: input_fn_(circle_train_data, circle_train_target), steps=1200)
evaluation = moon_classifier.evaluate(input_fn=lambda: input_fn_(circle_test_data, circle_test_target), steps=1)
loss_score = evaluation["loss"]
accuracy_score = evaluation["accuracy"]
print("Circle loss: {0:f}".format(loss_score))
print('Circle accuracy: {0:f}'.format(accuracy_score))


def make_new_samples(x_train_space, y_train_space):
    """
    makes data that reasonably covers data space (with just the right padding) to make predictions on
    """
    samples = []
    x_cen = (max(x_train_space) + min(x_train_space))/2.0
    x_range = max(x_train_space) - min(x_train_space)
    y_cen = (max(y_train_space) + min(y_train_space)) / 2.0
    y_range = max(y_train_space) - min(y_train_space)
    axis1 = np.linspace(x_cen - x_range * .6, x_cen + x_range * .6, 350)
    axis2 = np.linspace(y_cen - y_range * .6, y_cen + y_range * .6, 350)
    for i in range(len(axis1)):
        for q in range(len(axis2)):
            samples.append([axis1[i], axis2[q]])
    samples = np.asarray(samples)
    return samples

blob_samples = make_new_samples(blob[0][:, 0], blob[0][:, 1])
blob_prediction = list(blob_classifier.predict(input_fn=lambda: input_fn_(blob_samples, []), as_iterable=True))
moon_samples = make_new_samples(moon[0][:, 0], moon[0][:, 1])
moon_prediction = list(moon_classifier.predict(input_fn=lambda: input_fn_(moon_samples, []), as_iterable=True))
circle_samples = make_new_samples(circle[0][:, 0], circle[0][:, 1])
circle_prediction = list(circle_classifier.predict(input_fn=lambda: input_fn_(circle_samples, []), as_iterable=True))

fig = plt.figure(figsize=(12, 4))
sn_colors = sns.color_palette()
cm1 = plt.cm.get_cmap('ocean')
tics = ['s', 'o', '^']
ax = fig.add_subplot(131)


def make_contour_grid(x, y, z, x_res=300, y_res=300):
    """
    uses linear interpolation to ready data for a contour plot
    """
    xi = np.linspace(min(x), max(x), x_res)
    yi = np.linspace(min(y), max(y), y_res)
    z_grid = griddata(x, y, z, xi, yi, interp='linear')
    x_grid, y_grid = np.meshgrid(xi, yi)
    return x_grid, y_grid, z_grid

X, Y, Z = make_contour_grid(blob_samples[:, 0], blob_samples[:, 1], blob_prediction, x_res=100, y_res=100)
plt.contourf(X, Y, Z, alpha=0.8, cmap=cm1)
for tcolor, tstate, tmark in zip(sn_colors, range(len(np.unique(blob[1]))), tics):
    w = np.where(blob[1] == tstate)[0]
    plt.scatter(np.take(blob[0][:, 0], w), np.take(blob[0][:, 1], w), marker=tmark, color=tcolor, lw=0.0, s=20)
ax.set_xlim([np.min(blob[0][:, 0]), np.max(blob[0][:, 0])])
ax.set_ylim([np.min(blob[0][:, 1]), np.max(blob[0][:, 1])])

ax2 = fig.add_subplot(132)
X, Y, Z = make_contour_grid(moon_samples[:, 0], moon_samples[:, 1], moon_prediction, x_res=100, y_res=100)
plt.contourf(X, Y, Z, alpha=0.8, cmap=cm1)
for tcolor, tstate, tmark in zip(sn_colors, range(len(np.unique(moon[1]))), tics):
    w = np.where(moon[1] == tstate)[0]
    plt.scatter(np.take(moon[0][:, 0], w), np.take(moon[0][:, 1], w), marker=tmark, color=tcolor, lw=0.0, s=20)
ax2.set_xlim([np.min(moon[0][:, 0]), np.max(moon[0][:, 0])])
ax2.set_ylim([np.min(moon[0][:, 1]), np.max(moon[0][:, 1])])

ax3 = fig.add_subplot(133)
X, Y, Z = make_contour_grid(circle_samples[:, 0], circle_samples[:, 1], circle_prediction, x_res=100, y_res=100)
plt.contourf(X, Y, Z, alpha=0.8, cmap=cm1)
for tcolor, tstate, tmark in zip(sn_colors, range(len(np.unique(circle[1]))), tics):
    w = np.where(circle[1] == tstate)[0]
    plt.scatter(np.take(circle[0][:, 0], w), np.take(circle[0][:, 1], w), marker=tmark, color=tcolor, lw=0.0, s=20)
ax3.set_xlim([np.min(circle[0][:, 0]), np.max(circle[0][:, 0])])
ax3.set_ylim([np.min(circle[0][:, 1]), np.max(circle[0][:, 1])])

plt.show()
fig.savefig('classifier_tensorflow.png')
plt.close(fig)
