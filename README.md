# Active Learning Classifier

This repository applies the Gaussian Process and Bayesian optimization in an active learning framework for efficiently solving classification problems on a grid of two-dimensions or higher. Specifically, we apply this to a problem in physics: charting phase diagrams. We test this on a theoretical system: the Heisenberg Hamiltonian. We solve for the ground state energies of spin wave ansatzes throughout a symmetrical lattice and classify accordingly. Our active learning framework explores the phase space based on prior knowledge. Different acquisition strategies are applied for sampling the space dyanmically, as opposed to a traditional and resource intensive grid search. 

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

- `MS` — **Margin Sampling**  
  Selects the sample with the **smallest difference** between the top two predicted class probabilities:

  \[
  \text{MS}(x) = p_{\hat{1}}(x) - p_{\hat{2}}(x)
  \]

  Where \( p_{\hat{1}}(x) \) and \( p_{\hat{2}}(x) \) are the highest and second-highest predicted probabilities. Lower margin implies more uncertainty.

---

- `LC` — **Least Confidence**  
  Selects the sample where the model is **least confident** in its top prediction:

  \[
  \text{LC}(x) = 1 - \max_{c} p_c(x)
  \]

  The smaller the top class confidence, the more uncertain the prediction.

---

- `SE` — **Shannon Entropy**  
  Measures total uncertainty across all classes:

  \[
  \text{SE}(x) = -\sum_{c=1}^{C} p_c(x) \log p_c(x)
  \]

  Higher entropy indicates a more uncertain and informative point for labeling.

---

- `Random` — **Random Sampling**  
  Selects unlabeled points uniformly at random. Baseline for comparison.

- `Random-MS` — **Hybrid Random-Margin Sampling**  
  Rolls a virtual die to randomly switch between margin sampling and pure random selection. Encourages exploration and avoids local overfitting.


---

## Output

- Saves 2D decision boundary plots and error evolution over time
- Logs training samples and test performance
- Outputs final classification accuracy and error log as JSON

---

## Contact

For questions or contributions, please contact the repository owner at ejansen@bnl.gov
