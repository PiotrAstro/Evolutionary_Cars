# Evolutionary Cars Project

This project implements various evolutionary algorithms to train autonomous vehicles in a simulated environment. The project explores different optimization strategies including Evolutionary Strategies, Genetic Algorithms, and Differential Evolution.

I have used this work as a scientific project at my University, recommend reading **methods_comparison_report.pdf**

Currently working on Julia version of this project: https://github.com/PiotrAstro/JuliaEvolutionaryCars

## Project Overview

The project consists of several key components:

- **Evolutionary Algorithms**: Implementation of various optimization algorithms
  - Evolutionary Strategies [2]
  - Differential Evolution
  - Mutation-Only Genetic Algorithm [1] and slightly modified version
- **Environments**: Simulation environments for training
  - Basic Car Environment
  - Abstract Environment framework
- **Neural Networks**: Deep learning models for vehicle control
- **Visualization Tools**: Tools for visualizing training progress and results

Environments and Neural Networks are self written in Cython. It allows whole fitness calculation to happen in Cython using only c types. Therefore, GIL is released and these calculation can utilise true multithreading.

### Prerequisites

- Python 3.10 or higher

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Evolutionary_Cars.git
cd Evolutionary_Cars
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project

To run the main training script:
```bash
python -m src_files.main
```

Visualization tools are available in src_files/scripts/

One can also run tests of different metaparameters with src_files/scripts/metaparams_tests.py

## Project Structure

```
src_files/
├── Evolutionary_Algorithms/     # Various optimization algorithms
├── Environments/               # Simulation environments
├── Neural_Network/            # Deep learning models
├── Environments_Visualization/ # Visualization tools
└── MyMath/                    # Mathematical utilities
```

## Key Features

- Multiple evolutionary algorithm implementations
- Customizable car environment
- Neural network-based control systems
- Visualization tools
- Metaparameters tests
- Multithreading supported by using Cython for calculations

## Results and Analysis

A detailed analysis of the project's results and methodology can be found in the `methods_comparison_report.pdf` file. This report provides:
- Comparative analysis of different optimization algorithms
- Performance metrics and benchmarks
- Detailed methodology and implementation details
- Future research directions

Best method so far is modified [1].
It is very simple - I have a small population size (e.g. 400), each epoch I copy all individuals, modify copied individuals with huge mutation factor (e.g. 0.05) and then sort copied and original individuals by their fitness and take best 50%.

I have tried to use adaptive methods for mutation parameter ([3], [4], [5]) but all of them failed being too greedy. These are nice concepts though so I recommend reading articles :)

### Building Cython Extensions

To build the Cython extensions:
```bash
python setup.py clean --all build_ext --inplace
```

## Literature

1. Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for
Training Deep Neural Networks for Reinforcement Learning (https://arxiv.org/pdf/1712.06567 access: 27.03.2025)
2. Evolution Strategies as a
Scalable Alternative to Reinforcement Learning (https://arxiv.org/pdf/1703.03864 access: 27.03.2025)
3. Success-history based parameter adaptation for Differential Evolution (https://ieeexplore.ieee.org/document/6557555 access: 27.03.2025)
4. Safe Mutations for Deep and Recurrent Neural Networks
through Output Gradients (https://arxiv.org/pdf/1712.06563 access: 27.03.2025)
5. Effective Mutation Rate Adaptation
through Group Elite Selection (https://arxiv.org/pdf/2204.04817 access: 27.03.2025)
