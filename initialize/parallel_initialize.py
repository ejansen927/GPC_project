import argparse
import numpy as np
import json
import os
from blackbox import blackboxfunc
from multiprocessing import Pool
from functools import partial

def evaluate_point(x, Nx, Ny):
    return blackboxfunc(x, Nx, Ny)[0]

def main(n_points, dim, lattice_size, output_file, seed=None):
    if seed is not None:
        np.random.seed(seed)

    Nx = lattice_size
    Ny = lattice_size

    X = np.random.rand(n_points, dim) * 2 - 1  # Uniform in [-1, 1]

    # Create a partially applied function with Nx and Ny fixed
    eval_func = partial(evaluate_point, Nx=Nx, Ny=Ny)

    with Pool() as pool:
        y = pool.map(eval_func, X)

    data = {
        "X": X.tolist(),
        "y": y
    }

    save_dir = "saved_init_points"
    os.makedirs(save_dir, exist_ok=True)

    if not output_file.endswith(".json"):
        output_file += ".json"

    filename = os.path.join(save_dir, output_file)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved initial training points to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random points with phase labels from blackboxfunc")
    parser.add_argument('--n', type=int, default=100, help="Number of points to generate")
    parser.add_argument('--dim', type=int, default=5, help="Dimensionality of the input space")
    parser.add_argument('--lattice', type=int, default=12, help="Lattice size used by blackbox")
    parser.add_argument('--out', type=str, default="saved_train_points.json", help="Output filename")
    parser.add_argument('--seed', type=int, default=None, help="Random seed (optional)")

    args = parser.parse_args()
    main(args.n, args.dim, args.lattice, args.out, seed=args.seed)

