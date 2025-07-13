import argparse
import time
import numpy as np

### import files here
from model import *
from utils import plot_prediction_initial, plot_prediction, plot_error
from acquisition_function import acquisition_function  # <- make sure you import this

parser = argparse.ArgumentParser(prog='Active Learning Classifier', description="Run a program with specified parameters.")

parser.add_argument('--init_points', type=int, default=200, help='Number of initial points')
parser.add_argument('--num_runs', type=int, default=50, help='Number of runs')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
parser.add_argument('--num_dimensions', type=int, default=2, help='Number of dimensions')
parser.add_argument('--lattice_size', type=int, default=128, help='Length of the Lattice in one dimension, so L x L lattice (Nx x Ny)')
parser.add_argument('--test_set_percent', type=float, default=0.9, help='Percentage of the test set')
parser.add_argument('--model', type=str, default='GPC', choices=['SVC', 'GPC'], help='Model to use (SVC or GPC)')
parser.add_argument('--method', type=str, default='MS', choices=['LC', 'MS', 'SE', 'Random-MS', 'Random', 'VR', 'NormMS', 'ID'], help='Acquisition method')
parser.add_argument('--opt', type=str, default='Minimizer', choices=['Minimizer', 'Monte-Carlo'], help='Optimizer type')
parser.add_argument('--kernel', type=str, default=None, choices=['None', 'RBF', 'Matern'], help='Kernel choice for GPC')
parser.add_argument('--show_plots', type=bool, default=True, help='True: show plots. False: only save plots')
parser.add_argument('--load_points', type=str, default=None, help='Load saved training points from JSON')

args = parser.parse_args()

def run_program(init_points, num_runs, num_classes, num_dimensions, lattice_size, kernel, test_set_percent, show_plots, model='GPC', method='MS', opt='Minimizer', data_file=None):
    if kernel == 'Matern':
        kernel = Matern(1.0)
    elif kernel == 'RBF':
        kernel = RBF(1.0)
    elif kernel == 'None' or kernel is None:
        kernel = None

    # Initialize
    if data_file is not None:
        X_loaded, y_loaded = load_data(data_file)
        X_train, y_train, X_test, y_test, error, clf = init_when_loading(X_loaded, y_loaded, num_classes, lattice_size, test_set_percent, model, kernel)          
    else:
        X_train, y_train, X_test, y_test, error, clf = initialize(init_points, num_classes, num_dimensions, lattice_size, test_set_percent, model, method, opt, kernel, show_plots=show_plots)

    X, y = X_train, y_train
    print(f"Initialization done: {time.time()}")
    error_list = [error]
    counter = 0
    start_time = time.time()

    for i in range(num_runs):
        counter += 1
        print(f"Iteration: {counter}, Time elapsed: {time.time() - start_time:.2f}s")

        # For 'ID' method, pass X_train to single_iteration/acquisition_function
        X_train_pass = X if method == 'ID' else None

        X, y, clf, error = single_iteration(
            init_points, X, y, X_test, y_test,
            clf, num_classes, num_dimensions,
            lattice_size, kernel, model, method, opt,
            X_train_pass
        )

        error_list.append(error)
        print(f"Error: {error}")

        if i % 25 == 0:
            print(f"Checkpoint at iteration {i}")

    print(f"Final iteration: {counter}, Total time: {time.time() - start_time:.2f}s")

    plot_all_2d_gpc(X, clf, num_classes, num_dimensions, show_plots=show_plots)
    plot_error_over_time(error_list, num_classes, num_dimensions, show_plots)
    xy_logging(X, y)
    save_data_file(error_list, num_classes, num_dimensions)
    return error_list

if __name__ == "__main__":
    run_program(
        args.init_points, args.num_runs, args.num_classes,
        args.num_dimensions, args.lattice_size, args.kernel,
        args.test_set_percent, args.show_plots,
        args.model, args.method, args.opt, args.load_points
    )
    finalize_logging(args.num_classes, args.num_dimensions, args)

