import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from sklearn.metrics import zero_one_loss
import time
import sys

from sklearn.model_selection import train_test_split
from acquisition_function import acquisition_function
from blackbox import blackboxfunc

from utils import *

def initialize(init_points, num_classes, num_dimensions, lattice_size, test_set_percent, model, method, opt, kernel, show_plots):
    L = lattice_size
    Nx = L
    Ny = L
    X = np.random.rand(init_points, num_dimensions) * 2 - 1
    X_train, X_test = train_test_split(X, test_size=test_set_percent, random_state=42)
    y_train = np.array([blackboxfunc(i, Nx, Ny)[0] for i in X_train])
    y_test = np.array([blackboxfunc(i, Nx, Ny)[0] for i in X_test])

    time0 = time.time()
    clf = fit_model(X_train, y_train, model=model, kernel=kernel)
    time1 = time.time()
    print(f"Time for initial fit of {init_points} datapoints is {time1 - time0}.")

    Z1 = clf.predict(X_test)
    Z1_c = encode(Z1, num_classes)
    Z = clf.predict_proba(X_test)
    Z = np.clip(Z, 1e-12, 1.0)
    Z = handle_pred(Z, y_test, num_classes)

    y_test_encoded = encode(y_test, num_classes)
    y_true_labels = np.argmax(y_test_encoded, axis=1)

    error01 = zero_one_loss(y_test_encoded, Z1_c)
    #errorLL = log_loss(y_true_labels, Z)
    errorLL = log_loss(y_true_labels, Z, labels=list(range(num_classes)))
    #brier = brier_score_loss(y_true_labels, Z[np.arange(len(Z)), y_true_labels])
    errorB = bound_measure(X_train, y_train)

    #error = ([error01], [errorLL], [errorB], [brier])
    error = ([error01], [errorLL], [errorB])
    return X_train, y_train, X_test, y_test_encoded, error, clf

def init_when_loading(X_loaded, y_loaded, num_classes, lattice_size, test_set_percent, model, kernel):
    L = lattice_size
    Nx = L
    Ny = L

    # Use a proper split from loaded data
    X_train, X_test, y_train, y_test = train_test_split(X_loaded, y_loaded, test_size=test_set_percent, random_state=42)

    clf = fit_model(X_train, y_train, model=model, kernel=kernel)

    Z1 = clf.predict(X_test)
    Z1_c = encode(Z1, num_classes)
    Z = clf.predict_proba(X_test)
    Z = np.clip(Z, 1e-12, 1.0)
    Z = handle_pred(Z, y_test, num_classes)
    y_test_encoded = encode(y_test, num_classes)
    y_true_labels = np.argmax(y_test_encoded, axis=1)

    error01 = zero_one_loss(y_test_encoded, Z1_c)
    errorLL = log_loss(y_true_labels, Z, labels=list(range(num_classes)))
    #brier = brier_score_loss(y_true_labels, Z[np.arange(len(Z)), y_true_labels])
    errorB = bound_measure(X_train, y_train)

    #error = ([error01], [errorLL], [errorB], [brier])
    error = ([error01], [errorLL], [errorB])
    return X_train, y_train, X_test, y_test_encoded, error, clf

def single_iteration(init_points, X_train, y_train, X_test, y_test, clf,
                     num_classes, num_dimensions, lattice_size,
                     kernel, model, method, opt, X_train_pass=None):
    L = lattice_size
    Nx = L
    Ny = L

    #next_x = new_opt(X_train_pass or X_train, y_train, clf, num_classes, num_dimensions, method, opt)
    X_train_input = X_train_pass if X_train_pass is not None else X_train
    next_x = new_opt(X_train_input, y_train, clf, num_classes, num_dimensions, method, opt)

    next_y = np.array([blackboxfunc(next_x, Nx, Ny)[0]])

    X = np.vstack([X_train, next_x])
    y = np.concatenate((y_train, next_y))

    clf = fit_model(X, y, model, kernel)

    Z1 = clf.predict(X_test)
    Z1_c = encode(Z1, num_classes)
    Z = clf.predict_proba(X_test)
    Z = np.clip(Z, 1e-12, 1.0)
    Z = handle_pred(Z, y_test, num_classes)
    y_true_labels = np.argmax(y_test, axis=1)

    error01 = zero_one_loss(y_test, Z1_c)
    #errorLL = log_loss(y_true_labels, Z)
    errorLL = log_loss(y_true_labels, Z,labels=list(range(num_classes)))
    #brier = brier_score_loss(y_true_labels, Z[np.arange(len(Z)), y_true_labels])
    errorB = bound_measure(X, y)

    error = ([error01], [errorLL], [errorB])
    #error = ([error01], [errorLL], [errorB], [brier])
    return X, y, clf, error

def new_opt(X_train, y_train, clf, num_classes, num_dimensions, method, opt):
    if opt == 'Minimizer':
        new_point = local_minimizer(clf, method, num_classes, num_dimensions, X_train)
    elif opt == 'Monte-Carlo':
        new_point = MCOpt(clf, method, num_classes, num_dimensions, X_train)
    else:
        sys.exit(f'Not recognized method or opt. Method: {method}, opt: {opt}.')
    return new_point

def local_minimizer(clf, method, num_classes, num_dimensions, X_train=None):
    if method == 'Random':
        return np.random.rand(num_dimensions) * 2 - 1
    else:
        X_guess = np.random.rand(num_dimensions) * 2 - 1
        bounds = [(-1, 1)] * num_dimensions
        res = minimize(acquisition_function, X_guess, args=(clf, method, num_classes, num_dimensions, X_train), method='Nelder-Mead', tol=1e-5, bounds=bounds)
        if res.success:
            return res.x
        else:
            print('Failed local minimization. Retrying.')
            print(f'failed result.x: {res.x}')
            res = minimize(acquisition_function, X_guess, args=(clf, method, num_classes, num_dimensions, X_train), method='Nelder-Mead', tol=1e-5, bounds=bounds)
            print(f'result.x after retrying: {res.x}')
            return res.x

def MCOpt(clf, method, num_classes, num_dimensions, X_train=None, MCRuns=1000):
    x = np.random.rand(MCRuns, num_dimensions)
    vals = []
    for i in range(MCRuns):
        val = acquisition_function(x[i], clf, method, num_classes, num_dimensions, X_train)
        vals.append(val)
    index = np.argmin(vals)
    x_min = x[index]
    return x_min

