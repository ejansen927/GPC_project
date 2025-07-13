# Active Learning Classifier

This repository contains an active learning framework for efficiently classifying multi-class datasets on 2D or higher-dimensional grids using models such as Gaussian Process Classifier (GPC) or Support Vector Classifier (SVC). The system supports several acquisition strategies and optimization backends.

---

## Features

- Active learning loop with configurable acquisition strategies (`MS`, `LC`, `Random`, etc.)
- Applies Gaussian Process Classification (with RBF or Matern kernels) on a grid space of 2 dimensions or higher
- Simple built-in 2D visualization of decision boundaries and prediction error
- For more advanced plots, metadata such as coordinates, classes, and loss metrics are saved out to JSON files

---

## Dependencies

Requires the following Python packages installed:

- `numpy`
- `matplotlib`
- `scikit-learn`

---

## Usage

```bash
python main.py [--flags]
```

### Arguments

| Argument              | Type    | Default     | Description |
|-----------------------|---------|-------------|-------------|
| `--init_points`       | `int`   | `200`       | Number of initial training points |
| `--num_runs`          | `int`   | `50`        | Number of active learning iterations |
| `--num_classes`       | `int`   | `3`         | Number of target classes |
| `--num_dimensions`    | `int`   | `2`         | Number of input dimensions |
| `--lattice_size`      | `int`   | `128`       | Lattice side length (for a `L x L` grid) |
| `--test_set_percent`  | `float` | `0.9`       | Proportion of data used for testing |
| `--model`             | `str`   | `'GPC'`     | Classification model: `GPC` or `SVC` |
| `--method`            | `str`   | `'MS'`      | Acquisition strategy: `LC`, `MS`, `SE`, `Random`, `VR`, etc. |
| `--opt`               | `str`   | `'Minimizer'` | Optimizer type: `Minimizer` or `Monte-Carlo` |
| `--kernel`            | `str`   | `None`      | Kernel for GPC: `RBF`, `Matern`, or `None` |
| `--show_plots`        | `bool`  | `True`      | Whether to show plots (`True`) or only save them (`False`) |
| `--load_points`       | `str`   | `None`      | Path to JSON file containing initial training points |

---

## Example usage

```bash
python main.py \
  --init_points 100 \
  --num_runs 60 \
  --num_classes 3 \
  --method MS \
  --model GPC \
  --kernel RBF \
  --show_plots True
```

---

- **main.py** — Runs the training loop and handles CLI arguments.
- **model.py** — Defines models and classifier training utilities.
- **utils.py** — Includes functions for plotting, logging, and saving data.
- **acquisition_function.py** — Contains acquisition strategies for active learning.

---

## Available Acquisition Methods

- `MS` — Margin Sampling  
- `LC` — Least Confidence  
- `SE` — Shannon Entropy  
- `Random` — Random Sampling  
- `Random-MS` — Hybrid, random-margin sampling. Rolls die to determine which

---

## Output

- Saves 2D decision boundary plots and error evolution over time
- Logs training samples and test performance
- Outputs final classification accuracy and error log as JSON

---

## Contact

For questions or contributions, please contact the repository owner at ejansen@bnl.gov
