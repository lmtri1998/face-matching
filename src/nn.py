import sys
from imutils import paths
import numpy as np
import argparse
import pickle
import time
import cv2
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


ap = argparse.ArgumentParser()

ap.add_argument("--data", default="data_insight.pickle",
    help='Path to data')
ap.add_argument("--folder", default="matching_out_2865_max/")
ap.add_argument("--models_out", default="nn.pickle")
args = ap.parse_args()

data = pickle.loads(open(args.folder + args.data, "rb").read())
# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(data["labels"])
print("Encoder: ", labels)

X = np.array(data['data'])
y = labels

# title = "Learning Curves (NN Classifier)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = KFold(n_splits = 5, random_state=123, shuffle=True)
# fig, axes = plt.subplots(3, 2, figsize=(10, 15))
# clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(16, 32), random_state=123)
# plot_learning_curve(clf, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=4)
# plt.savefig('nn.png')
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=123)
cv = KFold(n_splits = 5, random_state=123, shuffle=True)
i = 1
for train_idx, valid_idx in cv.split(X):
    # svclassifier = SVC(kernel='linear', random_state=123, probability=True)
    X_train, X_test, y_train, y_test = X[train_idx], X[valid_idx], y[train_idx], y[valid_idx]
    # import pdb; pdb.set_trace()
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(16, 32), random_state=1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_labels = np.where(y_pred[:, 1] > 0.5, 1, 0)
    # svclassifier.fit(X_train, y_train)
    # y_pred = svclassifier.predict(X_test)
    #print(y_pred)
    with open(args.folder + 'fold_' + str(i) + '_' + args.models_out, 'rb') as f:
        model = pickle.load(f)
    with open(args.folder + 'fold_' + str(i) + '_' + args.models_out, 'wb') as f:
        pickle.dump(clf, f)
    i += 1
    # y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    report = confusion_matrix(y_test,y_labels)
    print(report)
    print(classification_report(y_test,y_labels))

    TP = report[0][0]
    FP = report[1][0]
    TN = report[1][1]
    FN = report[0][1]
    Precision = round(TP/(TP+FP),4)
    Recall = round(TP/(TP+FN),4)
    Accuracy = round((TP+TN)/(TP+TN+FP+FN),4)
    F_core = round(2*Precision*Recall/(Precision+Recall),4)
    print("[INFO] Precision = {}".format(Precision))
    print("[INFO] Accuracy = {}".format(Accuracy))
    print("[INFO] Recall = {}".format(Recall))
    print("[INFO] F_core = {}".format(F_core))

#ROC
# fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred)

# # Plot
# plt.figure(1)
# plt.plot(fpr_rt_lm, tpr_rt_lm)
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')

# plt.legend(['ROC AUC'], loc='lower right')
# plt.savefig('nn.png')
