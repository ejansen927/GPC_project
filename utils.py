import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessClassifier
import sys
import os
import json
from datetime import datetime

log_data = []
start_time = datetime.now().isoformat()

def create_directory(num_classes, num_dimensions):
    directory = f'./plots/{num_classes}-phase-{num_dimensions}-dim'
    os.makedirs(directory, exist_ok=True)
    return directory

def log_metadata(clf_type, opt, acq_func, num_classes, num_dimensions, kernel, plot_type, file_path):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "classifier": clf_type,
        "optimizer": opt,
        "acq_func": acq_func,
        "num_classes": num_classes,
        "num_dimensions": num_dimensions,
        "kernel": str(kernel),
        "plot_type": plot_type,
        "file_path": file_path
    }
    log_data.append(log_entry)

def write_log_to_file(directory):
    end_time = datetime.now().isoformat()
    log_file = os.path.join(directory, f'log_{start_time.replace(":", "-")}.json')
    metadata = {
        "start_time": start_time,
        "end_time": end_time,
        "entries": log_data
    }
    with open(log_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def plot_prediction_initial(X_test, y_test_plot, clf, method, opt, num_classes, num_dimensions, kernel, show_plots):
    directory = create_directory(num_classes, num_dimensions)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(directory, f'init_phase_plot_{timestamp}.png')

    colors = np.array(['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'teal', 'cyan', 'magenta', 
                       'lavender', 'maroon', 'olive', 'mint', 'coral', 'navy', 'beige', 'taupe', 'charcoal', 'aqua'])
    xx = np.linspace(-1, 1, 100)
    yy = xx
    xx, yy = np.meshgrid(xx, yy)

    plt.clf()  # Clear the current figure

    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors[y_test_plot], s=50, edgecolors='k', label='Truth')
    points = np.c_[xx.ravel(), yy.ravel()] 
    pred = clf.predict(points)
    clf_type = model_to_string(clf)
    plt.scatter(xx.ravel(), yy.ravel(), c=colors[pred], s=5, alpha=0.5, label='Pred')
    plt.legend()
    plt.xlabel('J1')
    plt.ylabel('J2')
    plt.title(f'{num_classes} Classes, {clf_type}, {method}, {opt}')
    plt.savefig(file_path)
    if show_plots:
        plt.show(block=False)
    else:
        plt.close()

    log_metadata(clf_type, opt, method, num_classes, num_dimensions, kernel, "initial prediction", file_path)

def plot_prediction(X_test, y_test_plot, clf, method, opt, counter, num_classes, num_dimensions, kernel, show_plots):
    directory = create_directory(num_classes, num_dimensions)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(directory, f'phase_plot_{timestamp}_{counter}.png')

    colors = np.array(['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'teal', 'cyan', 'magenta', 
                       'lavender', 'maroon', 'olive', 'mint', 'coral', 'navy', 'beige', 'taupe', 'charcoal', 'aqua'])
    xx = np.linspace(-1, 1, 100)
    yy = xx
    xx, yy = np.meshgrid(xx, yy)

    plt.clf()  # Clear the current figure

    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors[y_test_plot], s=50, edgecolors='k', label='Truth')
    points = np.c_[xx.ravel(), yy.ravel()] 
    pred = clf.predict(points)
    clf_type = model_to_string(clf)
    plt.scatter(xx.ravel(), yy.ravel(), c=colors[pred], s=5, alpha=0.5, label='Pred')
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'{num_classes} Classes, {clf_type}, {method}, {opt}')
    plt.savefig(file_path)
    if show_plots:
        plt.show(block=False)
    else:
        plt.close()

    log_metadata(clf_type, opt, method, num_classes, num_dimensions, kernel, "prediction", file_path)

def plot_prediction_5d_2d(X_test, y_test_plot, clf, show_plots):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    colors = np.array(['red', 'green', 'blue'])
    
    # Define pairs of dimensions for 2D plots (all combinations for 5D)
    dimension_pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))

    for idx, (dim1, dim2) in enumerate(dimension_pairs):
        ax = axes[idx // 5, idx % 5]

        # Generate a 2D mesh grid
        xx = np.linspace(X_test[:, dim1].min(), X_test[:, dim1].max(), 100)
        yy = np.linspace(X_test[:, dim2].min(), X_test[:, dim2].max(), 100)
        xx, yy = np.meshgrid(xx, yy)

        # Scatter plot of the true test points
        ax.scatter(X_test[:, dim1], X_test[:, dim2], c=colors[y_test_plot], s=50, edgecolors='k', label='Truth')
        
        # Prepare grid points for prediction
        points = np.zeros((xx.ravel().shape[0], X_test.shape[1]))
        points[:, dim1] = xx.ravel()
        points[:, dim2] = yy.ravel()
        
        pred = clf.predict(points)
        
        # Scatter plot of the predicted points
        ax.scatter(xx.ravel(), yy.ravel(), c=colors[pred], s=5, alpha=0.5, label='Pred')
        
        ax.set_xlabel(f'x{dim1+1}')
        ax.set_ylabel(f'x{dim2+1}')
        ax.set_title(f'Plot of x{dim1+1} vs x{dim2+1}')
        ax.legend()

    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_all_2d_gpc(X, gpc,num_classes,num_dimensions,show_plots=False):
    dimensions = ['J1', 'J2', 'J3', 'H_a', 'A']
    num_dims = X.shape[1]
    
    colors = np.array(['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'teal', 'cyan', 'magenta',
                       'lavender', 'maroon', 'olive', 'mint', 'coral', 'navy', 'beige', 'taupe', 'charcoal', 'aqua'])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #directory = f'plots_{timestamp}'
    directory = create_directory(num_classes, num_dimensions)
    os.makedirs(directory, exist_ok=True)

    for i in range(num_dims):
        for j in range(i + 1, num_dims):
            x1_range = np.linspace(X[:, i].min(), X[:, i].max(), 100)
            x2_range = np.linspace(X[:, j].min(), X[:, j].max(), 100)
            x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
            
            test_points = np.zeros((x1_grid.ravel().shape[0], num_dims))
            test_points[:, i] = x1_grid.ravel()
            test_points[:, j] = x2_grid.ravel()
            
            for k in range(num_dims):
                if k != i and k != j:
                    test_points[:, k] = 0
            
            predictions = gpc.predict(test_points)
            predictions_grid = predictions.reshape(x1_grid.shape)
            
            plt.figure(figsize=(10, 8))
            plt.contourf(x1_grid, x2_grid, predictions_grid, alpha=0.7, colors=colors)
            plt.title(f'{dimensions[i]} vs {dimensions[j]}')
            plt.xlabel(dimensions[i])
            plt.ylabel(dimensions[j])
            #plt.colorbar()
            file_path = os.path.join(directory, f'{dimensions[i]}_{dimensions[j]}_{timestamp}.png')
            #plt.savefig(os.path.join(directory, f'2D_GPC_{dimensions[i]}_vs_{dimensions[j]}.png'))
            plt.savefig(file_path)
            if show_plots:
                plt.show()
            else:
                plt.close()

def plot_error(error_list, model, method, opt, num_classes, num_dimensions, kernel):
    directory = create_directory(num_classes, num_dimensions)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(directory, f'loss_{timestamp}.png')
    err_vec = np.arange(0, len(error_list))

    plt.clf()  # Clear the current figure

    plt.plot(err_vec, error_list)
    plt.grid(True)
    plt.xlabel('iterations')
    plt.ylabel('log loss')
    plt.title(f'{num_classes} Classes, {model}, {method}')
    plt.savefig(file_path)
    plt.show(block=False)

    log_metadata(model, opt, method, num_classes, num_dimensions, kernel, "error plot", file_path)

def model_to_string(clf):
    """ Used in plotting to get title for model.
    Params:
    * clf - the classifier model
    Returns:
    * res - the type of model, either GPC or SVC. 
    """
    if isinstance(clf, GaussianProcessClassifier):
        return 'GPC'
    elif isinstance(clf, svm.SVC):
        return 'SVC'
    else:
        sys.exit('Unknown classifier type detected. Either formatted incorrectly, or neither scikitlearn GPC or SVM.')

def random_class(X_guess, num_classes=2):
    """ Used to select a random class for y in MOCU AcqFunc. Dynamic by changing the number of classes.
    Params:
    * X_guess - the single initial point guessed for minimizer.
    Returns:
    * label - the random class.
    """
    for i in X_guess[0]:
        y_guess = np.random.rand()
    class_interval = 1 / num_classes
    for label in range(num_classes):
        if y_guess < (label + 1) * class_interval:
            return label

def fit_model(X_train, y_train, model, kernel):
    """ Fit chosen model (GPC or SVC) to the training set.
    Params:
    * X_train - training points np array. Shape: [N,2], where N is number of training points.
    * y_train - truth values for each point stored as np array. Shape: [N,].
    * model - selected model string passed. Tells which model to select.
    Returns:
    * clf - fitted classifier.
    """
    if model == 'GPC':
        clf = GaussianProcessClassifier(kernel=kernel, random_state=2).fit(X_train, y_train)
    elif model == 'SVC':
        clf = svm.SVC(probability=True).fit(X_train, y_train)
    else:
        sys.exit('fit_model only contains the following models: GPC, SVC')
    return clf

def encode(y, num_classes=2):
    """ One-hot encoding for y truth arrays. Used as y_truth for log_loss function.
    Params:
    * y - truth array of classes.
    * num_classes - number of classes in experiment. For dynamic handling.
    Returns:
    * y_out - one-hot encoded truth array.
    """
    y_out = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        print(f"LABEL: {label}, and NUM_CLASSES: {num_classes}")
        if label < num_classes:
            y_out[i, label] = 1
    return y_out

def log_loss_custom(y, p):
    """Custom log-loss function. I do the math manually and get the results from this. Not from Scikit learn. They are
    doing something different. Not sure how or why or what.
    """
    N = len(y)
    total_loss = 0
    for i in range(N):
        loss = -1 / N * np.sum(y[i] * np.log(p[i]))
        total_loss += loss
    return total_loss

def label_binarizer(y, num_classes=2):
    """ Deprecated. tried """
    classes = np.unique(y)  # finds unique elements in y
    n_samples = len(y)
    encoded_labels = np.zeros((n_samples, num_classes))
    class_to_index = {label: index for index, label in enumerate(classes)}
    for i, label in enumerate(y):
        class_index = class_to_index[label]
        encoded_labels[i, class_index] = 1
    return encoded_labels

def DEPRECATED_handle_pred(Z, num_classes=2):
    """ DEPRECATED. Logic is not good, redid. 
    Handles a very specific case, where once every few hundred iterations randomly test points will be generated without
    members of all classes. This crashes the program. Instead, this handles the prediction to add a zeros column in place of 
    the missing class predictions.
    """
    classes = [i for i in range(num_classes)]
    classes = np.unique(classes)
    new_Z = np.zeros((Z.shape[0], num_classes))
    ind = 0
    for i in range(num_classes):
        if i in classes:
            new_Z[:, i] = Z[:, ind]
            ind += 1
        else:
            new_Z[:, i] = np.zeros(Z.shape[0])
    return new_Z

def dep_handle_pred(Z,y,num_classes):
    """Takes prediction Z, the true labels present for the predictions (usually y_test), and the true number of classes 
    in the system to reshape arrays without predictions.
    """
    #all_classes = np.array([i for i in range(num_classes)])
    all_classes = np.arange(num_classes)
    classes_present = np.unique(y)
    #print(all_classes)
    #print(classes_present)
    set_all = set(all_classes)
    set_pres = set(classes_present)
    missing = set_all - set_pres # can subtract sets and find missing
    if missing:
        for i in missing:
            Z = np.insert(Z,i,0,axis=1)
    return Z

# attempt 3 at handling pred
def handle_pred(Z, y, num_classes):
    """Takes prediction Z, the true labels present for the predictions (usually y_test), and the true number of classes 
    in the system to reshape arrays without predictions.
    """
    all_classes = np.arange(num_classes)
    classes_present = np.unique(y)
    set_all = set(all_classes)
    set_pres = set(classes_present)
    missing = set_all - set_pres  # Find missing classes

    # Ensure Z has num_classes columns
    if Z.shape[1] < num_classes:
        # Pad Z with zero columns to have `num_classes` total columns
        Z = np.pad(Z, ((0, 0), (0, num_classes - Z.shape[1])), mode='constant')

    for i in missing:
        # Zero out missing class probabilities explicitly
        Z[:, i] = 0

    return Z


#def finalize_logging(num_classes, num_dimensions):
#    # run this in main to complete logging of run stats
#    directory = create_directory(num_classes, num_dimensions)
#    write_log_to_file(directory)

def finalize_logging(num_classes, num_dimensions, args):
    args_dict = vars(args)
    directory = create_directory(num_classes, num_dimensions)
    end_time = datetime.now().isoformat()
    log_file = os.path.join(directory, f'log_{start_time.replace(":", "-")}.json')
    metadata = {
        "start_time": start_time,
        "end_time": end_time,
        "entries": log_data,
        "command_line_args": args_dict
    }
    
    with open(log_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def xy_logging(X,y):
    # for saving model progress. wont need to iteratively train. Can refit and keep going with these as initial points.
    log_data={
        'X': X.tolist(),
        'y': y.tolist()
    }
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'plots/xy_coords/xy_data_{timestamp}.json','w') as f:
        json.dump(log_data,f,indent=4)

def load_data(path):
    # load x and y data from saved log file
    with open(path, 'r') as f:
        data=json.load(f)
    X = np.array(data['X'])
    y = np.array(data['y'])
    return X,y

def plot_error_over_time(error_list,num_classes, num_dimensions,show_plots):
    directory = create_directory(num_classes, num_dimensions)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Prepare data for plotting
    e01 = [error[0] for error in error_list]
    eLL = [error[1] for error in error_list]
    eB = [error[2] for error in error_list]

    num_runs = len(error_list)
    x_values = np.arange(num_runs)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, e01, label='Zero-One Loss')
    plt.plot(x_values, eLL, label='Log Loss')
    plt.plot(x_values, eB, label='Bound Loss')
    
    plt.title('Error Metrics Over Time')
    plt.xlabel('Iteration Number')
    plt.ylabel('Error Value')
    plt.legend()
    plt.grid(True)
    file_path = os.path.join(directory, f'error_plot_{timestamp}.png')
    plt.savefig(file_path)
    if show_plots:
        plt.show(block=False)
    else:
        plt.close()

def bound_measure(X, y):
    """
    Computes the custom error metric based on nearest neighbor distance from different classes.

    Args:
    X (np.array): Feature matrix of shape (n_samples, n_features).
    y (np.array): Labels array of shape (n_samples,).

    Returns:
    float: Normalized error metric.
    """
    cls = np.unique(y)
    distsum = 0
    for i in cls:
        # filter points belonging to class
        in_cls = (y == i)
        out_cls = (y != i)
        if np.sum(out_cls) == 0:
            continue  # skip if no other classes
        # points in current class
        X_in = X[in_cls]
        # not in current class
        X_out = X[out_cls]
        # NN not in current class
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_out)
        distances, _ = nbrs.kneighbors(X_in)
        distsum += np.sum(distances)
    err = distsum / len(X)
    return err

def save_data_file(error_list, num_classes, num_dimensions):
    directory = create_directory(num_classes, num_dimensions)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    file_path = os.path.join(directory, f'data_file_{timestamp}.npz')
    
    error_array = np.array(error_list, dtype=object)
    
    np.savez(file_path, error_list=error_array)
    
    print(f"Data saved to {file_path}")
