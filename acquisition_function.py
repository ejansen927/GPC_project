import sys
import numpy as np
from sklearn.metrics import pairwise_distances

def acquisition_function(X_guess, clf, method, num_classes, num_dimensions, X_train=None):
    num_of_guesses = 1
    X_guess = np.reshape(X_guess, (num_of_guesses, num_dimensions))  # initial guess

    if method == 'LC':
        Z = clf.predict_proba(X_guess)
        return 1 - np.max(Z, axis=1)  # uncertainty = 1 - confidence

    elif method == 'MS':
        Z = clf.predict_proba(X_guess)
        Z_sorted = np.sort(Z, axis=1)
        Zdiff = Z_sorted[:, -1] - Z_sorted[:, -2]
        return Zdiff

    elif method == 'SE':
        Z = clf.predict_proba(X_guess)
        Z = np.clip(Z, 1e-12, 1.0)  # avoid log(0)
        Z_se = -np.sum(Z * np.log(Z), axis=1)
        return Z_se

    elif method == 'Random-MS':
        r = np.random.rand(1)
        if r < 0.3:
            return np.random.rand(1)
        else:
            Z = clf.predict_proba(X_guess)
            Z_sorted = np.sort(Z, axis=1)
            Zdiff = Z_sorted[:, -1] - Z_sorted[:, -2]
            return Zdiff

    elif method == 'VR':
        Z = clf.predict_proba(X_guess)
        return 1 - np.max(Z, axis=1)

    elif method == 'NormMS':
        Z = clf.predict_proba(X_guess)
        Z = np.clip(Z, 1e-12, 1.0)
        Z_sorted = np.sort(Z, axis=1)
        margin = Z_sorted[:, -1] - Z_sorted[:, -2]
        entropy = -np.sum(Z * np.log(Z), axis=1)
        norm_margin = margin / (entropy + 1e-6)
        return norm_margin

    elif method == 'ID':
        if X_train is None:
            sys.exit("X_train is required for Information Density (ID) method")
        Z = clf.predict_proba(X_guess)
        entropy = -np.sum(Z * np.log(Z + 1e-12), axis=1)
        dists = pairwise_distances(X_guess, X_train)
        sim = np.exp(-dists / (np.std(dists) + 1e-6))
        density = np.mean(sim, axis=1)
        return entropy * density

    else:
        sys.exit('No working acquisition function was selected. Choose from: LC, MS, SE, Random-MS, VR, NormMS, ID')

