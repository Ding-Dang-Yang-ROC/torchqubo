# TorchQUBO

**TorchQUBO** is a PyTorch-based framework for solving QUBO (Quadratic Unconstrained Binary Optimization) problems with GPU acceleration. It provides seven solvers, both classical and modern, and can be used through a simple command-line interface (CLI) or Python API. The design is modular, making it easy to extend for research and applications.

## Installation

Requires **Python 3.8+** and **PyTorch 2.0+**.

pip install torchqubo

Or from source:

git clone https://github.com/Ding-Dang-Yang-ROC/torchqubo.git
cd torchqubo
pip install .

## Quick Start

Generate a random QUBO matrix:

torchqubo make-q --n 16 --density 0.1 --scale 1.0 --seed 42 --out q.txt

Solve with local search:

torchqubo solve-local --Q q.txt --restarts 64 --iters 300 --tabu 5 --out sol.txt --log run_local.txt

Example log (run_local.txt):

method: local_search
n: 16
device: cpu
restarts: 64
iters: 300
tabu: 5
energy: -37.25
runtime_sec: 0.001234

Compute the energy of a solution:

torchqubo energy --Q q.txt --sol sol.txt --log energy.txt

## Methods

- Local Search (solve-local): greedy / tabu search
- Simulated Annealing (solve-sa): annealing-based search
- Parallel Tempering (solve-pt): replica exchange Monte Carlo
- STE (solve-ste): gradient-based optimization (straight-through estimator)
- Gumbel-Sigmoid (solve-gumbel): continuous relaxation with Gumbel noise
- Spectral Rounding (solve-spectral): low-rank spectral approximation
- GRASP (solve-grasp): greedy randomized construction with local refinement

## Python API

import torch
from torchqubo import load_q, solve

Q = load_q("q.txt")
x, E = solve(Q, method="local")
print("Best energy:", E.item())

## License

Apache 2.0 License



👩‍💻 Author: Ding-Dang Yang | 📧 112258018@g.nccu.edu.tw

