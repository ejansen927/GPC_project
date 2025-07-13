# Initialization

When running multiple experiments, it is simpler to generate initial points once instead of before each trial to save time and resources when using an expensive blackbox function. The scripts `initialize.py` and `parallel_initialize.py` are used to generate labeled training data (`X`, `y`) for active learning experiments by evaluating random input points with a domain-specific `blackboxfunc`. The generated data is stored as JSON for later use in training models. Usage of this script is unique to the blackbox used in this experiment.

---

## Usage

### `initialize.py`

This script generates initial training points **serially**. It samples uniformly from the input space `[-1, 1]^d`, evaluates each point using the `blackboxfunc`, and saves the resulting `X` and `y` arrays to a JSON file.

#### **Usage**


```bash
python initialize.py --n 100 --dim 5 --lattice 12 --out init_points.json --seed 42
```

#### **Arguments**

| Argument       | Type   | Default                | Description                                             |
|----------------|--------|------------------------|---------------------------------------------------------|
| `--n`          | int    | `100`                  | Number of input points to generate                      |
| `--dim`        | int    | `5`                    | Dimensionality of the input space                       |
| `--lattice`    | int    | `12`                   | Lattice size for the blackbox evaluation                |
| `--out`        | str    | `saved_train_points.json` | Output JSON filename (saved in `saved_init_points/`) |
| `--seed`       | int    | `None`                 | Optional seed for reproducibility                       |

---

### `parallel_initialize.py`

This variant runs the same process but evaluates points **in parallel** using Python's `multiprocessing` module. It is ideal for large batches or computationally expensive blackbox functions.

#### **Usage**

```bash
python parallel_initialize.py --n 500 --dim 5 --lattice 12 --out init_points_parallel.json --seed 1337
```

#### **Performance Note**

Speedup depends on the number of cores available and the cost of evaluating `blackboxfunc`.

---

### Output

Both scripts save a JSON file in the following format:

```text
{
  "X": [[x1_1, x1_2, ..., x1_d], [x2_1, ..., x2_d], ...],
  "y": [label1, label2, ...]
}
```

The output file is saved in the `saved_init_points/` directory, created automatically if it doesn’t exist.

#### **Usage with main.py**

Load saved points using the `main.py --load_points` argument.
