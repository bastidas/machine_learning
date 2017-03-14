import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
label = "species"
iris_df = pd.DataFrame(iris.data, columns=cols)
# fit just petal_length and petal_width
del iris_df[cols[0]]
del iris_df[cols[1]]
cols = ["petal_length", "petal_width"]
iris_df[label] = iris.target
iris_train, iris_test = train_test_split(iris_df, test_size=.25, random_state=666)


def input_fn(data_frame):
    feature_cols = {k: tf.constant(data_frame[k].values)
                  for k in cols}
    labels = tf.constant(data_frame[label].values, shape=[data_frame[label].size], verify_shape=True)
    return feature_cols, labels

tf.logging.set_verbosity(tf.logging.ERROR)  # suppress tensor flow warnings
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in cols]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols, hidden_units=[8, 16, 8], n_classes=3,
                                            model_dir="/tmp/iris_model")

classifier.fit(input_fn=lambda: input_fn(iris_train), steps=8000)

ev = classifier.evaluate(input_fn=lambda: input_fn(iris_test), steps=1)
loss_score = ev["loss"]
accuracy_score = ev["accuracy"]
print("Loss: {0:f}".format(loss_score))
print('Accuracy: {0:f}'.format(accuracy_score))
fmt_score = '{0:f}'.format(accuracy_score)

# make grid of new prediction samples
new_samples = []
x_points = linspace(0, 7.6, 1000)
y_points = linspace(0, 2.7, 500)
for i in range(len(x_points)):
    for q in range(len(y_points)):
        new_samples.append([x_points[i], y_points[q]])

new_samples = np.asarray(new_samples)
new_samples = pd.DataFrame(new_samples, columns=cols)
new_samples[label] = ""
y_pred = list(classifier.predict(input_fn=lambda: input_fn(new_samples)))


def grid(x, y, z, x_res=1000, y_res=1000):
    xi = linspace(min(x), max(x), x_res)
    yi = linspace(min(y), max(y), y_res)
    Z = griddata(x, y, z, xi, yi, interp = 'linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z

# plot
new_samples = np.asarray(new_samples)
iris_np = np.asarray(iris_train)
y_pred = [int(y) for y in y_pred]
X, Y, Z = grid(new_samples[:, 0], new_samples[:, 1], y_pred)
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})
fig = plt.figure(figsize=(6, 6))
sn_colors = sns.color_palette()
cm1 = plt.cm.get_cmap('ocean')
tics = ['s', 'o', '^']
ax = fig.add_subplot(111)
plt.contourf(X, Y, Z, alpha=0.8, cmap=cm1)
for tcolor, tstate, tmark in zip(sn_colors, range(3), tics):
    w = np.where(iris_train[label] == tstate)[0]
    plt.scatter(np.take(iris_np[:, 0], w), np.take(iris_np[:, 1], w), marker=tmark, color=tcolor, lw=0.0, s=20)

ax.text(7, .1, '{0:f}'.format(accuracy_score), size=15, horizontalalignment='right')
ax.set_xlim([0, 7.4])
ax.set_ylim(0, 2.7)
plt.title("TensorFlow NN")

plt.show()
fig.savefig('iris_tensorflow.png')
plt.close(fig)
